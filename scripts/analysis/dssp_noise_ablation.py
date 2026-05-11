"""
DSSP Noise-Level Ablation Script.

Tests DSSP prediction accuracy at varying noise levels with a 4-condition
ablation matrix:
  1. seq+CATH  — both sequence and CATH provided
  2. seq only  — sequence provided, CATH masked (all-null)
  3. CATH only — sequence masked (all UNK=20), CATH provided
  4. neither   — both masked

For each (protein, t, condition) triple the script runs the trainer's
training-step forward path (deterministic t, no SC bootstrap) and reports
DSSP accuracy against TWO ground-truth versions:

- `broken_GT`: targets from the live data pipeline's DSSPTargetTransform.
  Until the atom-index bug fix in transforms.py is shipped to the model
  weights via retraining, this matches what `val/dssp_acc` reports during
  training. After retraining, the two GTs should converge.
- `fixed_GT`: targets recomputed inline using N/CA/C/O at the correct atom37
  indices (0,1,2,4). This is the real DSSP signal.

Usage:
    CUTLASS_PATH=/home/ubuntu/openfold/cutlass \
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
        proteina/scripts/analysis/dssp_noise_ablation.py \
        --ckpt data/weights/<ckpt>.ckpt \
        [--n_proteins 20] [--output_dir results/dssp_ablation] \
        [--t_values 0.0 0.05 ... 1.0]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# Disable torch.compile/dynamo to avoid bool-mask issue in pair_bias_attn
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")

from proteinfoundation.proteinflow.proteina import Proteina


# ---------------------------------------------------------------------------
# Data module bootstrap (same path as scripts/analysis/dssp_validation_reproduce.py)
# ---------------------------------------------------------------------------

def load_data_module(cfg_exp, num_workers: int = 0):
    """Instantiate the same PDBLightningDataModule training uses."""
    dataset_subdir = cfg_exp.get("dataset_config_subdir", None)
    rel_config_path = os.path.relpath(
        os.path.join(ROOT, "configs", "datasets_config")
        + (f"/{dataset_subdir}" if dataset_subdir else ""),
        os.path.dirname(__file__),
    )
    with hydra.initialize(config_path=rel_config_path, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp["dataset"])
    cfg_data.datamodule.num_workers = num_workers
    if num_workers == 0:
        cfg_data.datamodule.prefetch_factor = None
        cfg_data.datamodule.pin_memory = False
    multilabel_mode = OmegaConf.select(cfg_exp, "model.nn.multilabel_mode")
    if multilabel_mode is not None:
        cfg_data.datamodule.multilabel_mode = multilabel_mode
    cfg_data.datamodule.batch_size = 1
    dm = hydra.utils.instantiate(cfg_data.datamodule)
    if hasattr(dm, "use_multiprocessing"):
        dm.use_multiprocessing = False
    return dm


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def clone_batch(batch):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = {kk: (vv.clone() if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Forward path mirroring the trainer (deterministic t, no SC bootstrap, no
# random seq/extlig masking). Produces nn_out["dssp_logits"] for the given
# batch at noise level t_val.
# ---------------------------------------------------------------------------

def run_forward_at_t(model, batch, t_val: float, use_self_cond: bool = False):
    x_1, mask, batch_shape, n, dtype = model.extract_clean_sample(batch)
    x_1 = model.fm._mask_and_zero_com(x_1, mask)
    t = torch.full(batch_shape, float(t_val), device=model.device, dtype=x_1.dtype)
    x_0 = model.fm.sample_reference(
        n=n, shape=batch_shape, device=model.device, dtype=dtype, mask=mask,
        modality="coordinates",
    )
    x_t = model.fm.interpolate(x_0, x_1, t, modality="coordinates")
    batch["t"] = t
    batch["mask"] = mask
    batch["x_t"] = x_t
    if model.cfg_exp.training.seq_cond:
        seq = batch["residue_type"]
        seq = torch.where(seq == -1, torch.tensor(20, device=seq.device, dtype=seq.dtype), seq)
        if "residue_type_unmasked" not in batch:
            batch["residue_type_unmasked"] = seq.clone().detach()
        batch["residue_type"] = seq
    else:
        batch.pop("residue_type", None)
    batch.pop("cath_code", None)
    if model.cfg_exp.training.get("zero_sin_pos_emb", False):
        batch["_zero_idx_emb"] = True
    model._set_self_cond(batch, x_1, contact_map_mode=False, use_self_cond=use_self_cond)
    nn_out = model.predict_clean(batch)
    return nn_out, x_1, mask, t


# ---------------------------------------------------------------------------
# Ablation overrides (applied after batch arrives, before forward)
# ---------------------------------------------------------------------------

def apply_ablation(batch, *, mask_seq: bool, mask_cath: bool):
    """Return a clone of `batch` with ablations applied.

    - mask_seq=True   → all residue_type set to UNK (20)
    - mask_cath=True  → cath_code_indices/_mask zeroed (all-null fold)
    """
    b = clone_batch(batch)
    if mask_seq and "residue_type" in b:
        b["residue_type"] = torch.full_like(b["residue_type"], 20)
    if mask_cath:
        if "cath_code_indices" in b:
            b["cath_code_indices"] = torch.zeros_like(b["cath_code_indices"])
        if "cath_code_indices_mask" in b:
            b["cath_code_indices_mask"] = torch.zeros_like(b["cath_code_indices_mask"])
    return b


# ---------------------------------------------------------------------------
# Dual-GT helpers
# ---------------------------------------------------------------------------

def _pydssp_at_indices(coords, mask, coord_mask, idx_o):
    """Run pydssp using atom indices [0,1,2,idx_o] for N/CA/C/O. Returns [b,n]
    with -1 for invalid, or None if coords don't have enough atoms.

    idx_o:
      - 4 → real N/CA/C/O on atom37-ordered coords (FIXED behavior)
      - 3 → BUG path: passes CB to pydssp in place of O (matches what training
        actually saw, given DSSPTargetTransform's [0,1,2,3] indexing on
        atom37-ordered coords).
    """
    if coords.shape[2] < idx_o + 1:
        return None
    import pydssp
    ncao = coords[:, :, [0, 1, 2, idx_o], :].float()
    if coord_mask is not None and coord_mask.shape[2] >= idx_o + 1:
        ncao_valid = (coord_mask[:, :, 0] & coord_mask[:, :, 1]
                      & coord_mask[:, :, 2] & coord_mask[:, :, idx_o])
        valid = mask & ncao_valid
    else:
        valid = mask
    out = pydssp.assign(ncao, out_type="index").long()
    return torch.where(valid, out, torch.full_like(out, -1))


def compute_dssp_atom37_inline(coords, mask, coord_mask=None):
    """Real N/CA/C/O via atom37 indices [0,1,2,4]. None if <5 atoms."""
    return _pydssp_at_indices(coords, mask, coord_mask, idx_o=4)


def compute_dssp_buggy_atom3_inline(coords, mask, coord_mask=None):
    """N/CA/C/CB via atom37 indices [0,1,2,3] — matches the pre-fix
    DSSPTargetTransform behavior. Useful for confirming the model learned the
    broken supervision signal regardless of whether transforms.py is fixed."""
    return _pydssp_at_indices(coords, mask, coord_mask, idx_o=3)


def acc_with_confusion(dssp_logits, dssp_target):
    valid = dssp_target >= 0
    if valid.sum().item() == 0:
        return float("nan"), 0, {"pred": {0: 0, 1: 0, 2: 0}, "gt": {0: 0, 1: 0, 2: 0}}
    pred = dssp_logits.argmax(dim=-1)
    correct = (pred == dssp_target) & valid
    acc = correct.sum().float() / valid.sum().float().clamp(min=1)
    pred_counts = {c: int((pred[valid] == c).sum().item()) for c in (0, 1, 2)}
    gt_counts = {c: int((dssp_target[valid] == c).sum().item()) for c in (0, 1, 2)}
    return float(acc.item()), int(valid.sum().item()), {
        "pred": pred_counts, "gt": gt_counts,
    }


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
    parser.add_argument("--t_values", nargs="+", type=float, default=None,
                        help="Override the default 0..1 in 0.05 steps")
    parser.add_argument("--split", choices=["val", "train"], default="val",
                        help="Dataset split to draw proteins from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    load_dotenv(os.path.join(ROOT, ".env"))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    print(f"[load] {args.ckpt}")
    model = Proteina.load_from_checkpoint(args.ckpt, strict=False)
    model.eval().to(device)

    # Sanity: dssp head present and loaded
    raw_sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    head_w_loaded = model.nn.dssp_head[1].weight.detach().cpu()
    head_w_disk = raw_sd["nn.dssp_head.1.weight"].detach().cpu()
    assert torch.allclose(head_w_loaded, head_w_disk), \
        "dssp_head Linear weight differs between loaded model and ckpt on disk"
    print(f"[verify] dssp_head loaded OK; "
          f"bias={[round(b, 4) for b in model.nn.dssp_head[1].bias.tolist()]}")

    # --- Build data module + dataloader ---
    dm = load_data_module(model.cfg_exp, num_workers=0)
    if hasattr(dm, "prepare_data"):
        dm.prepare_data()
    dm.setup(stage="fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()
    print(f"[data] using {args.split} loader (batch_size=1)")

    # --- Configure t schedule ---
    t_values = (args.t_values if args.t_values is not None
                else [0.0] + [round(0.05 * i, 2) for i in range(1, 21)])

    # --- 4-condition ablation ---
    conditions = [
        ("seq+CATH",  False, False),  # mask_seq, mask_cath
        ("seq_only",  False, True),
        ("CATH_only", True,  False),
        ("neither",   True,  True),
    ]

    # --- Loop ---
    results = []
    proteins_seen = 0
    loader_iter = iter(loader)
    while proteins_seen < args.n_proteins:
        try:
            raw_batch = next(loader_iter)
        except StopIteration:
            print(f"[data] loader exhausted after {proteins_seen} proteins")
            break

        if "dssp_target" not in raw_batch:
            continue

        raw_batch = move_to_device(raw_batch, device)
        protein_id = raw_batch.get("protein_id", raw_batch.get("id", ["?"]))
        if isinstance(protein_id, (list, tuple)):
            protein_id = str(protein_id[0])
        coords = raw_batch.get("coords")
        coord_mask = raw_batch.get("coord_mask")
        # Mask used by the trainer (extract_clean_sample reads mask_dict)
        if "mask_dict" in raw_batch and isinstance(raw_batch["mask_dict"], dict):
            res_mask = raw_batch["mask_dict"]["coords"][..., 0, 0].bool()
        else:
            res_mask = (coord_mask.sum(dim=-1) > 0).bool()

        # Compute BOTH GT versions inline (don't trust raw_batch["dssp_target"]
        # — its correctness depends on whether transforms.py has been fixed).
        gt_broken = compute_dssp_buggy_atom3_inline(
            coords, res_mask,
            coord_mask.bool() if coord_mask is not None else None,
        )
        gt_fixed = compute_dssp_atom37_inline(
            coords, res_mask,
            coord_mask.bool() if coord_mask is not None else None,
        )
        if gt_fixed is None or gt_broken is None:
            print(f"[skip] {protein_id}: no atom37 backbone (CA-only?)")
            continue

        n_valid_b = int((gt_broken >= 0).sum().item())
        n_valid_f = int((gt_fixed >= 0).sum().item())
        broken_dist = {c: int((gt_broken == c).sum().item()) for c in (0, 1, 2)}
        fixed_dist = {c: int((gt_fixed == c).sum().item()) for c in (0, 1, 2)}
        print(f"\n[{proteins_seen+1}/{args.n_proteins}] {protein_id} "
              f"n_valid(broken={n_valid_b}, fixed={n_valid_f})")
        print(f"  broken_GT: L={broken_dist[0]} H={broken_dist[1]} E={broken_dist[2]}")
        print(f"  fixed_GT : L={fixed_dist[0]} H={fixed_dist[1]} E={fixed_dist[2]}")

        for t_val in t_values:
            for cond_name, mask_seq, mask_cath in conditions:
                wbatch = apply_ablation(raw_batch, mask_seq=mask_seq, mask_cath=mask_cath)
                with torch.no_grad():
                    nn_out, _, _, _ = run_forward_at_t(
                        model, wbatch, t_val, use_self_cond=False,
                    )
                logits = nn_out.get("dssp_logits")
                if logits is None:
                    continue
                acc_b, nv_b, conf_b = acc_with_confusion(logits, gt_broken)
                acc_f, nv_f, conf_f = acc_with_confusion(logits, gt_fixed)
                results.append({
                    "protein_id": protein_id,
                    "t": t_val,
                    "condition": cond_name,
                    "broken_GT_acc": acc_b,
                    "fixed_GT_acc": acc_f,
                    "n_valid_broken": nv_b,
                    "n_valid_fixed": nv_f,
                    "pred_L": conf_b["pred"][0],
                    "pred_H": conf_b["pred"][1],
                    "pred_E": conf_b["pred"][2],
                })
            row = results[-len(conditions):]
            print(f"  t={t_val:.2f}: " + "  ".join(
                f"{r['condition']}=brok {r['broken_GT_acc']:.3f}/fix {r['fixed_GT_acc']:.3f}"
                for r in row
            ))
        proteins_seen += 1

    if not results:
        print("[abort] no results collected")
        return

    # --- CSV ---
    csv_path = os.path.join(args.output_dir, "dssp_noise_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[csv] {csv_path}")

    # --- Aggregate + plot ---
    plot_results(results, args.output_dir, t_values, conditions)


def plot_results(results, output_dir, t_values, conditions):
    """Two side-by-side panels: broken-GT acc vs t, and fixed-GT acc vs t."""
    cond_names = [c[0] for c in conditions]
    colors = {"seq+CATH": "tab:blue", "seq_only": "tab:orange",
              "CATH_only": "tab:green", "neither": "tab:red"}

    def aggregate(field):
        agg = defaultdict(list)
        for r in results:
            agg[(r["t"], r["condition"])].append(r[field])
        return agg

    agg_b = aggregate("broken_GT_acc")
    agg_f = aggregate("fixed_GT_acc")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, agg, title in [
        (axes[0], agg_b, "broken_GT (matches val/dssp_acc — atom-index bug)"),
        (axes[1], agg_f, "fixed_GT  (real DSSP via N/CA/C/O at atom37 [0,1,2,4])"),
    ]:
        for cond in cond_names:
            means, stds = [], []
            for t in t_values:
                vals = agg.get((t, cond), [])
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                else:
                    means.append(float("nan"))
                    stds.append(0.0)
            means = np.array(means); stds = np.array(stds)
            ax.plot(t_values, means, "o-", label=cond, color=colors[cond], markersize=4)
            ax.fill_between(t_values, means - stds, means + stds,
                            alpha=0.15, color=colors[cond])
        ax.set_xlabel("t (0=noise, 1=clean)", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.05)
        ax.axhline(y=1/3, color="gray", linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("DSSP accuracy", fontsize=12)
    fig.suptitle("DSSP accuracy vs noise level — broken vs fixed GT", fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "dssp_noise_ablation.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {plot_path}")


if __name__ == "__main__":
    main()
