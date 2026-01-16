import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np
import torch

from openfold.np import protein
from openfold.np import residue_constants as rc

from proteinfoundation.utils.openfold_inference import OpenFoldDistogramOnlyInference


@dataclass(frozen=True)
class ParsedPDB:
    sequence: str
    pseudo_beta: np.ndarray  # [n, 3] in Å
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


def _tm_score(mobile: np.ndarray, target: np.ndarray) -> float:
    raise NotImplementedError("TM-score is not implemented")

def _parse_usalign_outfmt2(stdout: str) -> dict:
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) != 1:
        raise ValueError(f"Unexpected USalign -outfmt 2 output (expected 1 data line), got {len(data_lines)} lines")
    fields = data_lines[0].split("\t")
    if len(fields) < 11:
        raise ValueError(f"Unexpected USalign -outfmt 2 columns: {len(fields)}")
    # Columns per USalign help:
    # PDBchain1, PDBchain2, TM1, TM2, RMSD, ID1, ID2, IDali, L1, L2, Lali
    return {
        "pdbchain1": fields[0],
        "pdbchain2": fields[1],
        "tm1": float(fields[2]),
        "tm2": float(fields[3]),
        "rmsd": float(fields[4]),
        "id1": float(fields[5]),
        "id2": float(fields[6]),
        "idali": float(fields[7]),
        "l1": int(fields[8]),
        "l2": int(fields[9]),
        "lali": int(fields[10]),
    }

def _load_pdb(pdb_path: str) -> ParsedPDB:
    with open(pdb_path, "r") as f:
        pdb_str = f.read()

    # Pick first chain ID seen in the file (fallback to "A")
    chain_id = "A"
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            cid = line[21].strip()
            chain_id = cid if cid else "A"
            break

    prot = protein.from_pdb_string(pdb_str, chain_id=chain_id)
    # Drop unknown residues ('X') entirely. In pdb these correspond to water/hetero
    # residues that should not be part of the protein chain for this test.
    aatype = prot.aatype
    keep = aatype != rc.restype_num
    aatype = aatype[keep]
    atom_positions = prot.atom_positions[keep]

    seq = "".join([rc.restypes_with_x[a] for a in aatype])
    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = np.where(
        np.tile(is_gly[..., None], (*((1,) * len(is_gly.shape)), 3)),
        atom_positions[..., ca_idx, :],
        atom_positions[..., cb_idx, :],
    )
    ca_xyz = atom_positions[..., ca_idx, :]

    return ParsedPDB(sequence=seq, ca_xyz=ca_xyz, pseudo_beta=pseudo_beta, chain_id=chain_id)


def _distogram_probs_from_pseudo_beta(
    pseudo_beta: torch.Tensor,
    *,
    num_bins: int = 39,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
) -> torch.Tensor:
    """
    Builds distogram probabilities [1, L, L, num_bins] from CA coordinates in Å.
    """
    d = torch.cdist(pseudo_beta[None, ...], pseudo_beta[None, ...], p=2.0)[0]  # [L, L]
    boundaries = torch.linspace(min_bin, max_bin, num_bins - 1, device=pseudo_beta.device, dtype=pseudo_beta.dtype)
    b = torch.bucketize(d, boundaries)  # [L, L] in [0..num_bins-1]
    logits = torch.zeros((d.shape[0], d.shape[1], num_bins), device=pseudo_beta.device, dtype=pseudo_beta.dtype)
    logits.scatter_(-1, b[..., None], 1.0)
    return logits[None, ...]

def _protein_from_atom37(
    *,
    atom37: np.ndarray,  # [L, 37, 3]
    aatype: np.ndarray,  # [L]
    residue_index: np.ndarray,  # [L]
    chain_index: np.ndarray,  # [L]
    remark: str,
) -> protein.Protein:
    if atom37.ndim != 3 or atom37.shape[1] != rc.atom_type_num or atom37.shape[2] != 3:
        raise ValueError(f"Expected atom37 shape [L, 37, 3], got {atom37.shape}")
    if aatype.ndim != 1 or residue_index.ndim != 1 or chain_index.ndim != 1:
        raise ValueError("Expected aatype/residue_index/chain_index to be 1D arrays")
    if not (atom37.shape[0] == aatype.shape[0] == residue_index.shape[0] == chain_index.shape[0]):
        raise ValueError("Length mismatch between atom37 and index arrays")

    # Mask atoms that don't exist for the residue type (OpenFold atom37 definition)
    aatype_clamped = np.clip(aatype.astype(np.int32), 0, rc.restype_num)
    atom_mask = rc.restype_atom37_mask[aatype_clamped].astype(np.float32)  # [L, 37]
    b_factors = np.zeros_like(atom_mask, dtype=np.float32)

    return protein.Protein(
        atom_positions=atom37.astype(np.float32),
        aatype=aatype_clamped.astype(np.int32),
        atom_mask=atom_mask,
        residue_index=residue_index.astype(np.int32),
        b_factors=b_factors,
        chain_index=chain_index.astype(np.int32),
        remark=remark,
    )


def _write_pdb(prot: protein.Protein, out_path: str) -> None:
    pdb_str = protein.to_pdb(prot)
    with open(out_path, "w") as f:
        f.write(pdb_str)

def _compare_template_batches(batch_a: dict, batch_b: dict, *, atol: float = 1e-6) -> None:
    keys = {k for k in batch_a if k.startswith("template_")} | {k for k in batch_b if k.startswith("template_")}
    for key in sorted(keys):
        if key not in batch_a:
            print(f"{key}: only in full_template_zero_coords")
            continue
        if key not in batch_b:
            print(f"{key}: only in distogram_only")
            continue
        a = batch_a[key]
        b = batch_b[key]
        if a.shape != b.shape:
            print(f"{key}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if torch.is_floating_point(a) or torch.is_floating_point(b):
            diff = (a - b).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            frac_diff = (diff > atol).float().mean().item()
            if frac_diff > 0:
                print(f"{key}: max={max_diff:.6f} mean={mean_diff:.6f} frac>{atol}={frac_diff:.4f}")
        else:
            frac_diff = (a != b).float().mean().item()
            if frac_diff > 0:
                print(f"{key}: frac_diff={frac_diff:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, default="/home/ubuntu/AFdistill/1ctf.pdb")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model_name", type=str, default="model_1_ptm")
    parser.add_argument("--jax_params", type=str, default="/home/ubuntu/params/params_model_1_ptm.npz")
    parser.add_argument(
        "--template_mode",
        type=str,
        default="distogram_only",
        choices=["distogram_only", "full_template", "full_template_zero_coords"],
        help="Template featurization mode for the template run.",
    )
    parser.add_argument(
        "--template_mmcif_path",
        type=str,
        default=None,
        help="Path to template mmCIF file or directory (required for full_template modes, optional for distogram_only).",
    )
    parser.add_argument(
        "--template_chain_id",
        type=str,
        default=None,
        help="Template chain ID for mmCIF (default: chain from input PDB).",
    )
    parser.add_argument(
        "--kalign_binary_path",
        type=str,
        default=None,
        help="Path to kalign binary (required for full_template modes, optional for distogram_only).",
    )
    parser.add_argument(
        "--template_all_x",
        action="store_true",
        help="Mask template_aatype to X (index 20) for the template run.",
    )
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="/home/ubuntu")
    parser.add_argument(
        "--assert_template_improves",
        action="store_true",
        help="If set, raise if template inference does not improve USalign-aligned RMSD vs baseline.",
    )
    parser.add_argument(
        "--compare_features",
        action="store_true",
        help="Compare template_* features between distogram_only and full_template_zero_coords modes.",
    )
    args = parser.parse_args()

    parsed = _load_pdb(args.pdb)
    if parsed.pseudo_beta.shape[0] == 0:
        raise ValueError(f"No pseudo beta coordinates found for {args.pdb}")
    if args.template_mode in ("full_template", "full_template_zero_coords"):
        if args.template_mmcif_path is None:
            raise ValueError("template_mmcif_path is required for full_template modes")
        if args.kalign_binary_path is None:
            raise ValueError("kalign_binary_path is required for full_template modes")

    device = torch.device(args.device)
    pseudo_beta = torch.tensor(parsed.pseudo_beta, dtype=torch.float32, device=device)
    dist_probs = _distogram_probs_from_pseudo_beta(pseudo_beta)  # [1, L, L, 39]
    plt.imshow(np.sum(dist_probs[0, :, :, :5].cpu().numpy(), axis=-1))
    plt.savefig("contact_dist_probs.png")

    # OpenFold expects residue_type indices [0..20] (20=unknown); from_pdb_string already set aatype
    with open(args.pdb, "r") as f:
        pdb_str = f.read()
    prot = protein.from_pdb_string(pdb_str, chain_id=parsed.chain_id)
    # Must match _load_pdb filtering
    keep = prot.aatype != rc.restype_num
    residue_type = torch.tensor(prot.aatype[keep], dtype=torch.long, device=device)[None, :]
    mask = torch.ones_like(residue_type, dtype=torch.float32)

    # Ground-truth Protein object (filtered to match our test sequence)
    prot_gt = protein.Protein(
        atom_positions=prot.atom_positions[keep],
        aatype=prot.aatype[keep],
        atom_mask=prot.atom_mask[keep],
        residue_index=prot.residue_index[keep],
        b_factors=prot.b_factors[keep],
        chain_index=prot.chain_index[keep],
        remark="GT from input PDB (filtered: drop X/UNK residues)",
    )

    infer = OpenFoldDistogramOnlyInference(
        model_name=args.model_name,
        jax_params_path=args.jax_params,
        device=device,
        rm_template_sequence=args.template_all_x,
        max_recycling_iters=args.recycles,
    )
    template_chain_id = args.template_chain_id or parsed.chain_id
    if args.compare_features:
        if args.template_mmcif_path is None:
            raise ValueError("template_mmcif_path is required to compare features")
        if args.kalign_binary_path is None:
            raise ValueError("kalign_binary_path is required to compare features")
        dist_batch = infer.build_batch(
            dist_probs,
            residue_type,
            mask,
            template_mode="distogram_only",
            mask_template_aatype=args.template_all_x,
        )
        zero_batch = infer.build_batch(
            dist_probs,
            residue_type,
            mask,
            template_mode="full_template_zero_coords",
            template_mmcif_path=args.template_mmcif_path,
            template_chain_id=template_chain_id,
            kalign_binary_path=args.kalign_binary_path,
            mask_template_aatype=args.template_all_x,
        )
        print("\nTemplate feature diffs: distogram_only vs full_template_zero_coords")
        _compare_template_batches(dist_batch, zero_batch)

    out = infer(
        dist_probs,
        residue_type,
        mask,
        template_mode=args.template_mode,
        template_mmcif_path=args.template_mmcif_path,
        template_chain_id=template_chain_id,
        kalign_binary_path=args.kalign_binary_path,
        mask_template_aatype=args.template_all_x,
    )
    atom37_w_t = out["atom37"]
    if atom37_w_t.ndim != 4 or atom37_w_t.shape[0] != 1 or atom37_w_t.shape[2] != rc.atom_type_num or atom37_w_t.shape[3] != 3:
        raise ValueError(f"Unexpected atom37 shape from template run: {tuple(atom37_w_t.shape)}")

    # Baseline: no template = feed uniform distogram
    uniform = torch.full_like(dist_probs, 1.0 / dist_probs.shape[-1])
    out0 = infer(
        uniform,
        residue_type,
        mask,
        template_mode="distogram_only",
        mask_template_aatype=args.template_all_x,
    )
    atom37_no_t = out0["atom37"]
    if atom37_no_t.ndim != 4 or atom37_no_t.shape[0] != 1 or atom37_no_t.shape[2] != rc.atom_type_num or atom37_no_t.shape[3] != 3:
        raise ValueError(f"Unexpected atom37 shape from baseline run: {tuple(atom37_no_t.shape)}")

    print(f"n_res={parsed.ca_xyz.shape[0]}")

    # Save PDBs + run USalign vs GT (more interpretable than RMSD here)
    pdb_id = os.path.splitext(os.path.basename(args.pdb))[0]
    tag = f"{pdb_id}_chain{parsed.chain_id}_recycles{args.recycles}"
    template_tag = args.template_mode
    if args.template_all_x:
        template_tag += "_templateAllX"

    gt_path = os.path.join(args.out_dir, f"{tag}_gt.pdb")
    pred_template_path = os.path.join(args.out_dir, f"{tag}_pred_{template_tag}.pdb")
    pred_baseline_path = os.path.join(args.out_dir, f"{tag}_pred_no_template.pdb")

    _write_pdb(prot_gt, gt_path)

    prot_pred_template = _protein_from_atom37(
        atom37=atom37_w_t[0].detach().cpu().numpy(),
        aatype=prot_gt.aatype,
        residue_index=prot_gt.residue_index,
        chain_index=prot_gt.chain_index,
        remark=f"Predicted by OpenFold with template mode: {args.template_mode}",
    )
    _write_pdb(prot_pred_template, pred_template_path)

    prot_pred_baseline = _protein_from_atom37(
        atom37=atom37_no_t[0].detach().cpu().numpy(),
        aatype=prot_gt.aatype,
        residue_index=prot_gt.residue_index,
        chain_index=prot_gt.chain_index,
        remark="Predicted by OpenFold with uniform distogram (no template)",
    )
    _write_pdb(prot_pred_baseline, pred_baseline_path)

    print(f"Saved GT PDB:        {gt_path}")
    print(f"Saved baseline PDB:  {pred_baseline_path}")
    print(f"Saved template PDB:  {pred_template_path}")

    usalign = "/home/ubuntu/.local/bin/USalign"
    baseline_metrics = None
    template_metrics = None
    template_label = f"template_{args.template_mode}"
    for label, pred_path in [("baseline", pred_baseline_path), (template_label, pred_template_path)]:
        r = subprocess.run(
            [usalign, pred_path, gt_path, "-mol", "prot", "-outfmt", "2", "-ter", "2"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"\nUSalign ({label} vs GT) [-outfmt 2]:")
        print(r.stdout.strip())
        metrics = _parse_usalign_outfmt2(r.stdout)
        print(f"Parsed: TM={metrics['tm1']:.4f} aligned_RMSD={metrics['rmsd']:.3f}Å Lali={metrics['lali']}")
        if label == "baseline":
            baseline_metrics = metrics
        else:
            template_metrics = metrics

    if baseline_metrics is None or template_metrics is None:
        raise ValueError("Failed to parse USalign outputs for baseline/template")

    if args.assert_template_improves and not (template_metrics["rmsd"] < baseline_metrics["rmsd"]):
        raise AssertionError(
            "Expected template inference to improve USalign-aligned RMSD "
            f"(got {template_metrics['rmsd']:.3f} vs {baseline_metrics['rmsd']:.3f})."
        )


if __name__ == "__main__":
    main()


