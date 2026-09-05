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
    return float(np.abs(d_gen - d_gt).mean())
