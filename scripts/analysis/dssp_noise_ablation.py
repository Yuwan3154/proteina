"""
DSSP Noise-Level Ablation Script

Tests DSSP prediction accuracy at various noise levels with a 4-condition ablation:
  1. seq+CATH  — both sequence and CATH provided
  2. seq only  — sequence provided, CATH masked
  3. CATH only — sequence masked (all UNK=20), CATH provided
  4. neither   — both masked

For each condition, the script:
  - Takes ground-truth protein structures
  - Applies noise via flow-matching interpolation x_t = (1-t)*x_0 + t*x_1
  - Runs the model forward to get DSSP logits
  - Compares to ground-truth DSSP from full backbone coordinates

Usage:
    python scripts/analysis/dssp_noise_ablation.py \
        --ckpt data/weights/<checkpoint>.ckpt \
        --n_proteins 20 \
        --output_dir results/dssp_ablation
"""

import argparse
import gzip
import os
import sys

# Disable torch.compile/dynamo to avoid bool mask issues in pair_bias_attn
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

import glob
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_float32_matmul_precision("medium")

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.dssp_utils import compute_dssp_target
from proteinfoundation.utils.coors_utils import ang_to_nm
from proteinfoundation.datasets.cath_utils import (
    load_cath_mapping,
    cath_code_strings_to_indices_for_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cath_code_lookup(cathdata_dir):
    """Parse cath-b-newest-all.gz to build {pdb_chain: [cath_code, ...]} mapping."""
    lookup = defaultdict(list)
    gz_path = os.path.join(cathdata_dir, "cath-b-newest-all.gz")
    if not os.path.exists(gz_path):
        print(f"WARNING: {gz_path} not found — CATH codes will be unavailable")
        return lookup
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            domain_id = parts[0]      # e.g. "101mA00"
            cath_code = parts[2]      # e.g. "1.10.490.10"
            # Extract pdb + chain from domain id (first 4 chars = pdb, 5th = chain)
            if len(domain_id) < 5:
                continue
            pdb = domain_id[:4].lower()
            chain = domain_id[4].upper()
            key = f"{pdb}_{chain}"
            if cath_code not in lookup[key]:
                lookup[key].append(cath_code)
    return lookup


def build_batch(pt_data, t_val, x_0, model, device,
                seq_provided, cath_indices, cath_mask, ext_lig_provided):
    """Build a batch dictionary for a single protein at a given noise level."""
    coords = torch.as_tensor(pt_data.coords, dtype=torch.float32)  # [n, 37, 3] in Å
    residue_type = torch.as_tensor(pt_data.residue_type, dtype=torch.long)  # [n]
    n = residue_type.shape[0]

    # Mask (must be bool for pair_bias_attn)
    coord_mask = torch.as_tensor(pt_data.coord_mask)  # [n, 37]
    mask = (coord_mask.sum(dim=-1) > 0)  # [n] bool

    # Clean CA in nm
    x_1 = ang_to_nm(coords[:, 1, :])  # [n, 3]

    # Center
    valid = mask.bool()
    com = x_1[valid].mean(dim=0, keepdim=True)
    x_1 = x_1 - com
    x_1 = x_1 * mask[:, None].float()

    # Interpolation: x_t = (1-t)*x_0 + t*x_1
    t_tensor = torch.tensor([t_val], dtype=torch.float32)
    x_t = (1.0 - t_val) * x_0[:n, :] + t_val * x_1  # [n, 3]
    x_t = x_t * mask[:, None].float()

    # Self-conditioning: oracle clean coordinates (x_1 already centered, in nm)
    x_sc = x_1.clone()  # [n, 3]

    # Sequence
    if seq_provided:
        seq = residue_type.clone()
        seq = torch.where(seq == -1, torch.tensor(20, dtype=torch.long), seq)
    else:
        seq = torch.full_like(residue_type, 20)  # All UNK

    # ext_lig
    if ext_lig_provided and hasattr(pt_data, "ext_lig"):
        ext_lig = torch.as_tensor(pt_data.ext_lig, dtype=torch.long)
    else:
        ext_lig = torch.full((n,), 2, dtype=torch.long)  # All unknown

    # Sequence position
    if hasattr(pt_data, "seq_pos"):
        seq_pos = torch.as_tensor(pt_data.seq_pos, dtype=torch.long)
        if seq_pos.dim() == 1:
            seq_pos = seq_pos.unsqueeze(-1)
    else:
        seq_pos = torch.arange(n, dtype=torch.long).unsqueeze(-1)

    # Build batch (add batch dim)
    batch = {
        "x_t": x_t.unsqueeze(0).to(device),
        "t": t_tensor.to(device),
        "mask": mask.unsqueeze(0).to(device),
        "x_sc": x_sc.unsqueeze(0).to(device),
        "residue_type": seq.unsqueeze(0).to(device),
        "ext_lig": ext_lig.unsqueeze(0).to(device),
        "seq_pos": seq_pos.unsqueeze(0).to(device),
        "_zero_idx_emb": True,
    }

    # CATH indices
    if cath_indices is not None:
        batch["cath_code_indices"] = cath_indices.to(device)
        if cath_mask is not None:
            batch["cath_code_indices_mask"] = cath_mask.to(device)

    return batch, mask


def get_cath_indices_for_protein(protein_id, cath_lookup, cath_code_dir, multilabel_mode, device):
    """Look up CATH codes for a protein and convert to model indices."""
    cath_codes = cath_lookup.get(protein_id, [])
    if not cath_codes:
        cath_codes = ["x.x.x.x"]
    # cath_code_strings_to_indices_for_model expects List[List[str]]
    cath_code_list = [cath_codes]
    indices, mask = cath_code_strings_to_indices_for_model(
        cath_code_list, cath_code_dir, multilabel_mode, device=device
    )
    return indices, mask


def get_null_cath_indices(cath_code_dir, multilabel_mode, device):
    """Get fully-masked CATH indices (all null)."""
    return cath_code_strings_to_indices_for_model(
        [["x.x.x.x"]], cath_code_dir, multilabel_mode, device=device
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DSSP noise-level ablation")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--n_proteins", type=int, default=20,
                        help="Number of proteins to evaluate")
    parser.add_argument("--output_dir", default="results/dssp_ablation",
                        help="Output directory for results")
    parser.add_argument("--data_dir", default="data/pdb_train/processed",
                        help="Directory with processed .pt files")
    parser.add_argument("--cathdata_dir", default="data/cathdata",
                        help="Directory with CATH data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    model = Proteina.load_from_checkpoint(args.ckpt, strict=False)
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}")

    # Get model config
    cath_code_dir = model.cfg_exp.model.nn.get("cath_code_dir", "")
    if cath_code_dir and "${" in cath_code_dir:
        # Resolve env var
        cath_code_dir = os.path.expandvars(cath_code_dir.replace("${oc.env:", "$").replace("}", ""))
    if not os.path.isabs(cath_code_dir):
        cath_code_dir = os.path.join(root, cath_code_dir)
    multilabel_mode = model.cfg_exp.model.nn.get("multilabel_mode", "sample")

    # Load CATH lookup
    cath_lookup = load_cath_code_lookup(os.path.join(root, args.cathdata_dir))
    print(f"Loaded CATH lookup with {len(cath_lookup)} entries")

    # Null CATH indices (for masked condition)
    null_cath_indices, null_cath_mask = get_null_cath_indices(
        cath_code_dir, multilabel_mode, device
    )

    # Select proteins
    pt_files = sorted(glob.glob(os.path.join(root, args.data_dir, "*.pt")))
    np.random.seed(args.seed)
    np.random.shuffle(pt_files)
    pt_files = pt_files[:args.n_proteins]
    print(f"Selected {len(pt_files)} proteins for evaluation")

    # Noise levels: t=0 (pure noise) to t=0.95 (mostly clean)
    t_values = [0.0] + [round(0.05 * i, 2) for i in range(1, 21)]  # 0.0, 0.05, ..., 1.0

    # Conditions: (name, seq_provided, cath_provided)
    conditions = [
        ("seq+CATH", True, True),
        ("seq_only", True, False),
        ("CATH_only", False, True),
        ("neither", False, False),
    ]

    # Results storage
    results = []

    # Pre-sample a large Gaussian reference for reuse (largest possible protein)
    max_n = 768
    x_0_base = torch.randn(max_n, 3, dtype=torch.float32)
    # Center it
    x_0_base = x_0_base - x_0_base.mean(dim=0, keepdim=True)

    for pi, pt_file in enumerate(pt_files):
        protein_id = os.path.basename(pt_file).replace(".pt", "")
        print(f"\n[{pi+1}/{len(pt_files)}] Processing {protein_id}")

        pt_data = torch.load(pt_file, weights_only=False)
        n = pt_data.residue_type.shape[0]

        # Ground-truth DSSP from full backbone coordinates
        coords_full = torch.as_tensor(pt_data.coords, dtype=torch.float32).unsqueeze(0)  # [1, n, 37, 3]
        coord_mask_full = torch.as_tensor(pt_data.coord_mask).unsqueeze(0)  # [1, n, 37]
        res_mask = (coord_mask_full.sum(dim=-1) > 0).squeeze(0)  # [n]
        dssp_target = compute_dssp_target(
            coords_full, res_mask.unsqueeze(0), coord_mask_full.bool()
        )
        if dssp_target is None:
            print(f"  Skipping {protein_id}: DSSP target is None (CA-only data?)")
            continue
        dssp_target = dssp_target.squeeze(0).to(device)  # [n]
        valid = dssp_target >= 0

        if valid.sum() == 0:
            print(f"  Skipping {protein_id}: no valid DSSP residues")
            continue

        # CATH indices for this protein
        protein_cath_indices, protein_cath_mask = get_cath_indices_for_protein(
            protein_id, cath_lookup, cath_code_dir, multilabel_mode, device
        )

        # Gaussian reference x_0 for this protein (take first n coords, recenter)
        x_0 = x_0_base[:n].clone()
        x_0_valid = x_0[res_mask.bool()]
        x_0 = x_0 - x_0_valid.mean(dim=0, keepdim=True)
        x_0 = x_0 * res_mask[:, None].float()

        for t_val in t_values:
            for cond_name, seq_on, cath_on in conditions:
                cath_idx = protein_cath_indices if cath_on else null_cath_indices
                cath_msk = protein_cath_mask if cath_on else null_cath_mask

                batch, mask_cpu = build_batch(
                    pt_data, t_val, x_0, model, device,
                    seq_provided=seq_on,
                    cath_indices=cath_idx,
                    cath_mask=cath_msk,
                    ext_lig_provided=True,
                )

                with torch.no_grad():
                    nn_out = model.nn(batch)

                dssp_logits = nn_out.get("dssp_logits")
                if dssp_logits is None:
                    print(f"  WARNING: No dssp_logits in output for {protein_id}")
                    continue

                dssp_pred = dssp_logits.squeeze(0).argmax(dim=-1)  # [n]
                correct = (dssp_pred == dssp_target) & valid
                acc = correct.sum().float() / valid.sum().float()

                results.append({
                    "protein_id": protein_id,
                    "n_residues": int(n),
                    "t": t_val,
                    "condition": cond_name,
                    "dssp_accuracy": float(acc.item()),
                    "n_valid": int(valid.sum().item()),
                })

            # Print progress for this t
            accs_at_t = {r["condition"]: r["dssp_accuracy"]
                        for r in results[-4:]}
            print(f"  t={t_val:.2f}: " + " | ".join(
                f"{k}={v:.3f}" for k, v in accs_at_t.items()))

    # Save CSV
    csv_path = os.path.join(args.output_dir, "dssp_noise_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["protein_id", "n_residues", "t",
                                                "condition", "dssp_accuracy", "n_valid"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Plot
    plot_results(results, args.output_dir)


def plot_results(results, output_dir):
    """Plot DSSP accuracy vs noise level for each condition."""
    conditions = ["seq+CATH", "seq_only", "CATH_only", "neither"]
    colors = {"seq+CATH": "tab:blue", "seq_only": "tab:orange",
              "CATH_only": "tab:green", "neither": "tab:red"}

    # Aggregate by (t, condition) -> list of accuracies
    agg = defaultdict(list)
    for r in results:
        agg[(r["t"], r["condition"])].append(r["dssp_accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        t_vals = sorted(set(r["t"] for r in results))
        means = []
        stds = []
        for t in t_vals:
            vals = agg.get((t, cond), [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(float("nan"))
                stds.append(0)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(t_vals, means, "o-", label=cond, color=colors[cond], markersize=4)
        ax.fill_between(t_vals, means - stds, means + stds,
                        alpha=0.15, color=colors[cond])

    ax.set_xlabel("t (noise level: 0=pure noise, 1=clean)", fontsize=12)
    ax.set_ylabel("DSSP Prediction Accuracy", fontsize=12)
    ax.set_title("DSSP Accuracy vs Noise Level — Ablation Matrix", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1/3, color="gray", linestyle=":", alpha=0.5, label="random (1/3)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plot_path = os.path.join(output_dir, "dssp_noise_ablation.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
