import argparse
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

# Prefer the full OpenFold repo (installed in cue_openfold), not Proteina's shim.
sys.path.insert(0, "/home/ubuntu/openfold")

from openfold.np import protein as of_protein
from openfold.np import residue_constants as rc

from proteinfoundation.utils.openfold_inference import OpenFoldDistogramOnlyInference


@dataclass(frozen=True)
class Parsed1QYS:
    sequence: str
    ca_xyz: np.ndarray  # [n, 3] in Å
    chain_id: str


def _kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """
    Computes RMSD after optimal rigid alignment (Kabsch).
    Inputs: [n, 3] arrays.
    """
    mobile_c = mobile - mobile.mean(axis=0, keepdims=True)
    target_c = target - target.mean(axis=0, keepdims=True)

    h = mobile_c.T @ target_c
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    mobile_aligned = mobile_c @ r
    diff = mobile_aligned - target_c
    return float(np.sqrt((diff * diff).sum() / mobile.shape[0]))


def _load_1qys(pdb_path: str) -> Parsed1QYS:
    with open(pdb_path, "r") as f:
        pdb_str = f.read()

    # Pick first chain ID seen in the file (fallback to "A")
    chain_id = "A"
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            cid = line[21].strip()
            chain_id = cid if cid else "A"
            break

    prot = of_protein.from_pdb_string(pdb_str, chain_id=chain_id)
    # Drop unknown residues ('X') entirely. In 1qys these correspond to water/hetero
    # residues that should not be part of the protein chain for this test.
    aatype = prot.aatype
    keep = aatype != rc.restype_num
    aatype = aatype[keep]
    atom_positions = prot.atom_positions[keep]

    seq = "".join([rc.restypes_with_x[a] for a in aatype])
    ca_xyz = atom_positions[:, rc.atom_order["CA"], :].astype(np.float64)

    return Parsed1QYS(sequence=seq, ca_xyz=ca_xyz, chain_id=chain_id)


def _distogram_probs_from_ca(
    ca_xyz: torch.Tensor,
    *,
    num_bins: int = 39,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    peak_logit: float = 10.0,
) -> torch.Tensor:
    """
    Builds distogram probabilities [1, L, L, num_bins] from CA coordinates in Å.
    """
    d = torch.cdist(ca_xyz[None, ...], ca_xyz[None, ...], p=2.0)[0]  # [L, L]
    boundaries = torch.linspace(min_bin, max_bin, num_bins - 1, device=ca_xyz.device, dtype=ca_xyz.dtype)
    b = torch.bucketize(d, boundaries)  # [L, L] in [0..num_bins-1]
    logits = torch.zeros((d.shape[0], d.shape[1], num_bins), device=ca_xyz.device, dtype=torch.float32)
    logits.scatter_(-1, b[..., None], peak_logit)
    return torch.softmax(logits, dim=-1)[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, default="/home/ubuntu/AFdistill/1qys.pdb")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model_name", type=str, default="model_1_ptm")
    parser.add_argument("--jax_params", type=str, default="/home/ubuntu/params/params_model_1_ptm.npz")
    parser.add_argument("--template_all_x", action="store_true")
    parser.add_argument("--recycles", type=int, default=3)
    args = parser.parse_args()

    parsed = _load_1qys(args.pdb)
    if parsed.ca_xyz.shape[0] == 0:
        raise ValueError(f"No CA coordinates found for {args.pdb}")

    device = torch.device(args.device)
    ca_xyz = torch.tensor(parsed.ca_xyz, dtype=torch.float32, device=device)
    dist_probs = _distogram_probs_from_ca(ca_xyz)  # [1, L, L, 39]

    # OpenFold expects residue_type indices [0..20] (20=unknown); from_pdb_string already set aatype
    with open(args.pdb, "r") as f:
        pdb_str = f.read()
    prot = of_protein.from_pdb_string(pdb_str, chain_id=parsed.chain_id)
    # Must match _load_1qys filtering
    keep = prot.aatype != rc.restype_num
    residue_type = torch.tensor(prot.aatype[keep], dtype=torch.long, device=device)[None, :]
    mask = torch.ones_like(residue_type, dtype=torch.float32)

    infer = OpenFoldDistogramOnlyInference(
        model_name=args.model_name,
        jax_params_path=args.jax_params,
        device=device,
        template_sequence_all_x=args.template_all_x,
        max_recycling_iters=args.recycles,
    )
    out = infer(dist_probs, residue_type, mask)
    pred_w_t = out["atom37"][0, :, rc.atom_order["CA"], :].detach().cpu().numpy()

    # Baseline: no template = feed uniform distogram
    uniform = torch.full_like(dist_probs, 1.0 / dist_probs.shape[-1])
    out0 = infer(uniform, residue_type, mask)
    pred_no_t = out0["atom37"][0, :, rc.atom_order["CA"], :].detach().cpu().numpy()

    rmsd_no_t = _kabsch_rmsd(pred_no_t, parsed.ca_xyz[: pred_no_t.shape[0]])
    rmsd_w_t = _kabsch_rmsd(pred_w_t, parsed.ca_xyz[: pred_w_t.shape[0]])

    print(f"n_res={parsed.ca_xyz.shape[0]}")
    print(f"RMSD (sequence-only, no template): {rmsd_no_t:.3f} Å")
    print(f"RMSD (distogram-only template)   : {rmsd_w_t:.3f} Å")

    if not (rmsd_w_t < rmsd_no_t):
        raise AssertionError(
            f"Expected template inference to improve RMSD (got {rmsd_w_t:.3f} vs {rmsd_no_t:.3f})."
        )


if __name__ == "__main__":
    main()


