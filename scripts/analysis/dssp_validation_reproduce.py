"""
DSSP Validation Reproduction Harness.

Goal: decide whether the "all-loop" pathology in dssp_noise_ablation.py is a
batch-construction bug (H1) or a missing-EMA-weights problem (H2).

Strategy: load a real validation batch via the same Hydra-instantiated
PDBLightningDataModule that training uses, then run the forward path with a
deterministic `t` and compute dssp_acc the same way training does
(proteina.py:662-664). If raw weights reach >=0.85 at any tested t -> H1
(input pipeline bug, fix dssp_noise_ablation.py). If they stay <=0.45 at every
t -> H2 (raw weights inadequate, EMA needed but not on disk).

Usage:
    CUTLASS_PATH=/home/ubuntu/openfold/cutlass \
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
        proteina/scripts/analysis/dssp_validation_reproduce.py \
        --ckpt proteina/data/weights/<ckpt>.ckpt \
        [--t_values 0.3 0.55 0.8 1.0] [--n_batches 3] [--split val|train]
"""

import argparse
import os
import sys

# Disable torch.compile/dynamo to avoid known bool-mask issue in pair_bias_attn
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import hashlib
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")

from proteinfoundation.proteinflow.proteina import Proteina


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data_module(cfg_exp, num_workers: int = 0):
    """Instantiate the same PDBLightningDataModule that training uses."""
    dataset_subdir = cfg_exp.get("dataset_config_subdir", None)
    config_path = (
        f"../configs/datasets_config/{dataset_subdir}"
        if dataset_subdir is not None
        else "../configs/datasets_config/"
    )
    # Compose the data config relative to this script. Hydra path is relative to
    # the *file*, not to cwd, so this works regardless of cwd.
    rel_config_path = os.path.relpath(
        os.path.join(ROOT, "configs", "datasets_config")
        + (f"/{dataset_subdir}" if dataset_subdir else ""),
        os.path.dirname(__file__),
    )
    with hydra.initialize(
        config_path=rel_config_path, version_base=hydra.__version__
    ):
        cfg_data = hydra.compose(config_name=cfg_exp["dataset"])
    # Mirror train.py's overrides for parity with training
    cfg_data.datamodule.num_workers = num_workers
    if num_workers == 0:
        # PyTorch DataLoader rejects prefetch_factor when not multiprocessing
        cfg_data.datamodule.prefetch_factor = None
        cfg_data.datamodule.pin_memory = False
    multilabel_mode = OmegaConf.select(cfg_exp, "model.nn.multilabel_mode")
    if multilabel_mode is not None:
        cfg_data.datamodule.multilabel_mode = multilabel_mode
    cfg_data.datamodule.batch_size = max(
        int(cfg_exp.opt.get("batch_size", 1) or 1), 1
    )
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    if hasattr(datamodule, "use_multiprocessing"):
        datamodule.use_multiprocessing = False
    return datamodule


def move_batch_to_device(batch, device):
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


def hash_tensor(t: torch.Tensor) -> str:
    """Stable short hash for verifying weight loading."""
    arr = t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
    return hashlib.sha1(arr).hexdigest()[:12]


def dssp_acc_from_logits(dssp_logits, dssp_target):
    """Same formula as proteina.py:662-664."""
    valid = dssp_target >= 0
    if valid.sum().item() == 0:
        return float("nan"), 0, None
    pred = dssp_logits.argmax(dim=-1)
    correct = (pred == dssp_target) & valid
    acc = correct.sum().float() / valid.sum().float().clamp(min=1)
    # Per-class confusion (tp counts only)
    tp = {c: int(((pred == c) & (dssp_target == c)).sum().item())
          for c in (0, 1, 2)}
    pred_counts = {c: int((pred[valid] == c).sum().item()) for c in (0, 1, 2)}
    gt_counts = {c: int((dssp_target[valid] == c).sum().item()) for c in (0, 1, 2)}
    return float(acc.item()), int(valid.sum().item()), {
        "tp": tp, "pred": pred_counts, "gt": gt_counts,
    }


def compute_fixed_dssp_target(coords, mask, coord_mask=None):
    """Compute DSSP using the CORRECT atom indices for OpenFold-ordered coords.

    OpenFold's atom37 ordering puts CB at index 3 and O at index 4. The existing
    DSSPTargetTransform / compute_dssp_target use indices [0,1,2,3] = N/CA/C/CB,
    which is wrong: pydssp needs N/CA/C/O. With CB substituted for O the
    hydrogen-bond pattern is destroyed and pydssp returns ~all-loop.

    coords: [b, n, atoms, 3] (OpenFold-ordered)
    mask:   [b, n] bool
    coord_mask: optional [b, n, atoms] bool
    Returns: [b, n] long, with -1 for invalid.
    """
    if coords.shape[2] < 5:
        return None
    import pydssp
    # Correct indices in OpenFold/atom37 ordering: N=0, CA=1, C=2, O=4 (NOT 3=CB)
    ncao = coords[:, :, [0, 1, 2, 4], :]  # [b, n, 4, 3]
    if coord_mask is not None and coord_mask.shape[2] >= 5:
        ncao_valid = (
            coord_mask[:, :, 0]
            & coord_mask[:, :, 1]
            & coord_mask[:, :, 2]
            & coord_mask[:, :, 4]
        )
        valid_mask = mask & ncao_valid
    else:
        valid_mask = mask
    dssp_out = pydssp.assign(ncao.float(), out_type="index")
    dssp_target = dssp_out.long().clone()
    dssp_target = torch.where(
        valid_mask, dssp_target, torch.full_like(dssp_target, -1)
    )
    return dssp_target


def run_forward_at_t(model, batch, t_val: float, use_self_cond: bool = False):
    """Replicate the inner body of training_step with deterministic t.

    Mirrors model_trainer_base.training_step lines 906-1078, but skips:
    - the `random.random() > 0.5` self-cond bootstrap (use_self_cond explicit)
    - the residue/ext_lig random masking (we want eval-mode determinism)
    - the fold-mask random dropouts (preserve given cath_code_indices as-is)

    Returns (nn_out, x_1, mask, t).
    """
    # Extract clean sample (proteina.py:195-, model_trainer_base:573-)
    x_1, mask, batch_shape, n, dtype = model.extract_clean_sample(batch)
    x_1 = model.fm._mask_and_zero_com(x_1, mask)

    # Deterministic t
    t = torch.full(batch_shape, float(t_val), device=model.device, dtype=x_1.dtype)

    # Sample x_0 reference + interpolate (no contact_map mode for these ckpts)
    x_0 = model.fm.sample_reference(
        n=n, shape=batch_shape, device=model.device, dtype=dtype, mask=mask,
        modality="coordinates",
    )
    x_t = model.fm.interpolate(x_0, x_1, t, modality="coordinates")

    batch["t"] = t
    batch["mask"] = mask
    batch["x_t"] = x_t

    # Sequence conditional path (proteina.py training-step lines 1038-1053).
    # Crucial: do NOT random-mask residues; we want full sequence at test time.
    if model.cfg_exp.training.seq_cond:
        seq = batch["residue_type"]
        seq = torch.where(
            seq == -1,
            torch.tensor(20, device=seq.device, dtype=seq.dtype),
            seq,
        )
        if "residue_type_unmasked" not in batch:
            batch["residue_type_unmasked"] = seq.clone().detach()
        batch["residue_type"] = seq
    else:
        if "residue_type" in batch:
            batch.pop("residue_type")

    # Drop cath_code raw strings; keep cath_code_indices as-is (no fold-mask dropout)
    if "cath_code" in batch:
        batch.pop("cath_code")

    # zero_sin_pos_emb (line 1068-1070 of trainer base)
    if model.cfg_exp.training.get("zero_sin_pos_emb", False):
        batch["_zero_idx_emb"] = True

    # Self-conditioning: zero out, optionally bootstrap once
    target_clean = x_1
    model._set_self_cond(batch, target_clean, contact_map_mode=False,
                         use_self_cond=use_self_cond)

    nn_out = model.predict_clean(batch)
    return nn_out, x_1, mask, t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DSSP validation reproduction harness (H1 vs H2)"
    )
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--t_values", nargs="+", type=float,
        default=[0.3, 0.55, 0.8, 1.0],
        help="t values to evaluate dssp_acc at (1.0 = clean structure)",
    )
    parser.add_argument(
        "--n_batches", type=int, default=3,
        help="Number of batches to average over",
    )
    parser.add_argument(
        "--split", choices=["val", "train"], default="val",
        help="Which split to draw batches from (val matches val/dssp_acc reading)",
    )
    parser.add_argument(
        "--use_self_cond", action="store_true",
        help="Bootstrap self-cond (mirrors the 50% of train batches that ran SC)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model -------------------------------------------------------
    print(f"[load] {args.ckpt}")
    model = Proteina.load_from_checkpoint(args.ckpt, strict=False)
    model.eval().to(device)

    # Verify the dssp head is in the loaded weights (catches strict=False
    # silently swallowing keys).
    raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw_sd = raw_ckpt["state_dict"]
    head_keys = ["nn.dssp_head.0.weight", "nn.dssp_head.0.bias",
                 "nn.dssp_head.1.weight", "nn.dssp_head.1.bias"]
    for k in head_keys:
        assert k in raw_sd, f"checkpoint missing dssp head key {k!r}"
    head_w_loaded = model.nn.dssp_head[1].weight.detach().cpu()
    head_w_disk = raw_sd["nn.dssp_head.1.weight"].detach().cpu()
    assert torch.allclose(head_w_loaded, head_w_disk), \
        "dssp_head Linear weight differs between loaded model and ckpt on disk!"
    print(f"[verify] dssp head loaded; "
          f"weight hash={hash_tensor(head_w_loaded)} "
          f"bias={[round(b, 4) for b in model.nn.dssp_head[1].bias.tolist()]}")

    # Confirm checkpoint EMA situation (informational)
    opt0 = raw_ckpt.get("optimizer_states", [{}])[0]
    has_ema = "ema" in opt0
    print(f"[verify] checkpoint optimizer_states[0] has 'ema' key: {has_ema}")

    # --- Build data module ------------------------------------------------
    print(f"[data] composing dataset config: {model.cfg_exp.get('dataset')}")
    datamodule = load_data_module(model.cfg_exp, num_workers=0)
    print(f"[data] data_dir={datamodule.data_dir}")
    if hasattr(datamodule, "prepare_data"):
        datamodule.prepare_data()
    datamodule.setup(stage="fit")  # populates both train and val datasets
    if args.split == "val":
        loader = datamodule.val_dataloader()
    else:
        loader = datamodule.train_dataloader()
    print(f"[data] using {args.split} loader")

    # --- Run forward at each t over n_batches ----------------------------
    per_t = {t: [] for t in args.t_values}
    per_t_confusion = {t: [] for t in args.t_values}
    n_seen = 0

    loader_iter = iter(loader)
    for bi in range(args.n_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            print(f"[data] loader exhausted at batch {bi}")
            break
        # Some loaders return a single item; ensure dict
        if hasattr(batch, "keys"):
            batch_keys = list(batch.keys())
        else:
            batch_keys = ["<non-dict batch>"]
        print(f"[batch {bi}] keys: {batch_keys}")
        batch = move_batch_to_device(batch, device)
        if "dssp_target" not in batch:
            print(f"[batch {bi}] no dssp_target! (DSSPTargetTransform missing?) -- skipping")
            continue

        for t_val in args.t_values:
            # Make a working copy so the batch state stays clean across t's
            wbatch = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items()}
            # Re-deepcopy mask_dict (nested)
            if "mask_dict" in batch and isinstance(batch["mask_dict"], dict):
                wbatch["mask_dict"] = {
                    kk: (vv.clone() if isinstance(vv, torch.Tensor) else vv)
                    for kk, vv in batch["mask_dict"].items()
                }
            with torch.no_grad():
                nn_out, _x1, mask, _t = run_forward_at_t(
                    model, wbatch, t_val, use_self_cond=args.use_self_cond,
                )
            dssp_logits = nn_out.get("dssp_logits")
            if dssp_logits is None:
                print(f"  t={t_val}: NO dssp_logits in nn_out -- "
                      "checkpoint head not wired? predict_dssp=False?")
                continue
            dssp_target_broken = wbatch["dssp_target"]
            # Compute corrected GT inline (uses atom indices [0,1,2,4] — N/CA/C/O
            # in OpenFold ordering — instead of [0,1,2,3] which gives N/CA/C/CB).
            coords = wbatch.get("coords")
            cm = wbatch.get("coord_mask")
            mask_for_gt = wbatch.get("mask")
            if mask_for_gt is None and "mask_dict" in wbatch:
                mask_for_gt = wbatch["mask_dict"]["coords"][..., 0, 0].bool()
            dssp_target_fixed = compute_fixed_dssp_target(
                coords, mask_for_gt.bool() if mask_for_gt is not None else None,
                cm.bool() if cm is not None else None,
            )
            acc_b, nv_b, conf_b = dssp_acc_from_logits(dssp_logits, dssp_target_broken)
            if dssp_target_fixed is not None:
                acc_f, nv_f, conf_f = dssp_acc_from_logits(dssp_logits, dssp_target_fixed)
            else:
                acc_f, nv_f, conf_f = float("nan"), 0, None
            per_t[t_val].append((acc_b, nv_b, acc_f, nv_f))
            per_t_confusion[t_val].append((conf_b, conf_f))
            print(
                f"  t={t_val:.2f}:  broken_GT_acc={acc_b:.4f} "
                f"(n={nv_b}, gt={conf_b['gt']}, pred={conf_b['pred']})"
            )
            if conf_f is not None:
                print(
                    f"            fixed_GT_acc ={acc_f:.4f} "
                    f"(n={nv_f}, gt={conf_f['gt']}, pred={conf_f['pred']})"
                )
        n_seen += 1

    if n_seen == 0:
        print("[result] no batches processed -- abort")
        return

    print("\n=== Aggregate ===")
    best_b = {}
    best_f = {}
    for t_val in args.t_values:
        rows = per_t[t_val]
        if not rows:
            print(f"  t={t_val:.2f}: no data")
            continue
        # Broken GT (what training reports)
        tot_b_correct = sum(a * n for a, n, _, _ in rows)
        tot_b_valid = sum(n for _, n, _, _ in rows)
        wb = tot_b_correct / max(tot_b_valid, 1)
        # Fixed GT (what the model actually achieves on real DSSP labels)
        tot_f_correct = sum(a * n for _, _, a, n in rows if not (a != a))
        tot_f_valid = sum(n for _, _, a, n in rows if not (a != a))
        wf = (tot_f_correct / max(tot_f_valid, 1)) if tot_f_valid else float("nan")
        print(
            f"  t={t_val:.2f}: broken_GT_acc={wb:.4f} (n={tot_b_valid})  |  "
            f"fixed_GT_acc={wf:.4f} (n={tot_f_valid})"
        )
        best_b[t_val] = wb
        best_f[t_val] = wf

    if not best_b:
        print("[result] no aggregated values -- abort")
        return

    print("\n=== Verdict ===")
    best_acc_b = max(best_b.values())
    best_acc_f = max([v for v in best_f.values() if v == v], default=float("nan"))
    print(
        f"Best broken-GT acc across t: {best_acc_b:.4f}  "
        f"(this is what `val/dssp_acc` measures during training)"
    )
    if best_acc_f == best_acc_f:
        print(
            f"Best fixed-GT  acc across t: {best_acc_f:.4f}  "
            f"(this is what the model actually achieves on REAL DSSP labels)"
        )

    if best_acc_b >= 0.85 and (best_acc_f != best_acc_f or best_acc_f <= 0.50):
        print(
            "\nROOT CAUSE: DSSPTargetTransform is feeding pydssp the wrong atoms.\n"
            "  - pdb_data.py:695 reorders coords to OpenFold's atom37 layout\n"
            "    (atom 3 = CB, atom 4 = O).\n"
            "  - DSSPTargetTransform (transforms.py:101) then extracts indices\n"
            "    [0,1,2,3], passing N/CA/C/CB to pydssp instead of N/CA/C/O.\n"
            "  - With CB swapped for O, pydssp's H-bond computation fails and\n"
            "    returns ~all-loop. The model dutifully memorized that signal.\n"
            "  - The 99% val/dssp_acc agrees with broken targets, NOT with reality.\n"
            "Fix: change indices to [0,1,2,4] in DSSPTargetTransform (and verify\n"
            "compute_dssp_target callers — its docstring says O(3) but atom37\n"
            "puts O at 4). Then RETRAIN: existing weights are unsalvageable for\n"
            "DSSP because they were supervised on broken labels."
        )
    elif best_acc_b >= 0.85:
        print(
            "Raw weights reproduce high broken-GT acc AND non-trivial fixed-GT acc.\n"
            "The model has learned something despite the label bug. Investigate further."
        )
    else:
        print(
            f"Ambiguous broken-GT acc ({best_acc_b:.3f}). Try more batches "
            "(--n_batches), the train split (--split train), or self-cond "
            "(--use_self_cond)."
        )


if __name__ == "__main__":
    main()
