"""Write a structural readout for the contact-to-coordinate model.

Loss curves cannot be eyeballed; structures can. Each dump writes, per validation protein:
  <id>_gen.pdb   the model's sampled all-atom structure (full 200-step rollout)
  <id>_gt.pdb    the ground truth, same atom ordering
  <id>.npz       CA-CA distance matrices for both, plus the INPUT contact map

⭐ All-atom, not CA-only: this model predicts atom14, so the PDB carries real side chains and
opens directly in PyMOL. (The proteina trunk's own outputs are CA-only and need reconstruction --
that caveat does not apply here.)
"""

import os

import numpy as np
import torch

from proteinfoundation.openfold_stub.np import residue_constants as rc


def _atom14_names(aa_idx: int):
    aa3 = rc.restype_1to3.get(rc.restypes[aa_idx], "UNK") if aa_idx < len(rc.restypes) else "UNK"
    return aa3, rc.restype_name_to_atom14_names.get(aa3, [""] * 14)


def chirality_signs(coords14, aatype, mask):
    """Per-residue handedness at the CA stereocentre: sign of (N-CA) x (C-CA) . (CB-CA).

    ⭐ Why this metric exists. A contact map is CHIRALITY-BLIND -- reflection preserves every
    pairwise distance -- so the distogram loss, smooth-lDDT and dist_mae all score a perfect mirror
    image as perfect. Measured on real samples: 4/8 and 5/8 generated chains were mirrored, sitting
    at 11-14 A CA-RMSD while their distance MAE was 1.0-2.1 A. Nothing we logged could see it.

    Glycine has no CB (atom14 slot 4 empty) and is excluded -- it has no stereocentre.
    Returns the raw signs; the CALLER compares against the native's own signs rather than assuming
    a convention, so the metric cannot be wrong-way-round.
    """
    from proteinfoundation.openfold_stub.np import residue_constants as rc
    a14 = torch.as_tensor(rc.restype_atom14_mask, device=coords14.device,
                          dtype=coords14.dtype)[aatype.long().clamp(0, 20)]
    sel = (mask > 0.5) & (a14[:, 4] > 0.5)          # slot 4 is CB
    if int(sel.sum()) == 0:
        return coords14.new_zeros(0)
    n, ca = coords14[sel, 0], coords14[sel, 1]
    c, cb = coords14[sel, 2], coords14[sel, 4]
    trip = (torch.cross(n - ca, c - ca, dim=-1) * (cb - ca)).sum(-1)
    return torch.sign(trip)


def chirality_agreement(coords_gen14, coords_gt14, aatype, mask):
    """Fraction of stereocentres whose handedness matches the NATIVE's.

    1.0 = same handedness throughout; ~0.0 = globally MIRRORED; ~0.5 = scrambled/no stereochemistry.
    Comparing against the native rather than a hard-coded sign means an inverted convention on my
    side cannot flip the conclusion.
    """
    sg = chirality_signs(coords_gen14, aatype, mask)
    st = chirality_signs(coords_gt14, aatype, mask)
    if sg.numel() == 0 or sg.numel() != st.numel():
        return float("nan")
    return float((sg == st).float().mean())


def write_atom14_pdb(path, coords14, aatype, mask):
    """coords14 [L,14,3] Angstrom, aatype [L] long, mask [L]."""
    lines, serial = [], 1
    for i in range(coords14.shape[0]):
        if float(mask[i]) < 0.5:
            continue
        aa3, names = _atom14_names(int(aatype[i]))
        for j, nm in enumerate(names):
            if not nm:
                continue
            x, y, z = (float(v) for v in coords14[i, j])
            if not all(np.isfinite([x, y, z])):
                continue
            el = nm[0]
            lines.append(
                f"ATOM  {serial:>5d} {nm:<4s}{aa3:>3s} A{i+1:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {el:>2s}"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def dump_sample(out_dir, name, coords_gen14, coords_gt14, aatype, mask, contacts):
    """Write gen/gt PDBs and the distance-matrix npz for one protein."""
    os.makedirs(out_dir, exist_ok=True)
    keep = mask.bool()
    write_atom14_pdb(os.path.join(out_dir, f"{name}_gen.pdb"), coords_gen14, aatype, mask)
    write_atom14_pdb(os.path.join(out_dir, f"{name}_gt.pdb"), coords_gt14, aatype, mask)

    ca_gen = coords_gen14[keep][:, 1, :].float()
    ca_gt = coords_gt14[keep][:, 1, :].float()
    d_gen = torch.cdist(ca_gen, ca_gen).cpu().numpy()
    d_gt = torch.cdist(ca_gt, ca_gt).cpu().numpy()
    np.savez_compressed(
        os.path.join(out_dir, f"{name}.npz"),
        dist_gen=d_gen.astype(np.float32),
        dist_gt=d_gt.astype(np.float32),
        contacts_in=contacts[keep][:, keep].cpu().numpy().astype(np.uint8),
        aatype=aatype[keep].cpu().numpy().astype(np.int8),
    )
    # A single scalar that is directly comparable to the training rmsd: how far the sampled
    # distance matrix is from the true one, which needs no alignment and so cannot be flattered
    # by a bad superposition.
    chir = chirality_agreement(coords_gen14, coords_gt14, aatype, mask)
    return float(np.abs(d_gen - d_gt).mean()), chir
