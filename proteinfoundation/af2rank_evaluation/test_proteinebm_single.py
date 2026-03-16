#!/usr/bin/env python3
"""
Isolated debug script: run ProteinEBM on a single PDB with extensive logging.

Usage (from protebm conda env):
  python test_proteinebm_single.py [--pdb PATH] [--config PATH] [--checkpoint PATH] [--t FLOAT]

Defaults point to a known example so it can be run with no arguments.
"""

import argparse
import sys
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIG     = "/home/ubuntu/ProteinEBM/protein_ebm/config/proteinebm_v2_cathmd_config.yaml"
DEFAULT_CHECKPOINT = "/home/ubuntu/ProteinEBM/weights/proteinebm_v2_cathmd_weights.pt"
DEFAULT_PDB = (
    "/home/ubuntu/proteina/inference/"
    "inference_seq_cond_sampling_ca_beta-2.5-2.0_finetune-all_v1.4_default-fold_4-seq-S25_64-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise/"
    "1a2y_C/1a2y_C_0.pdb"
)


def banner(msg: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  {msg}", flush=True)
    print('='*60, flush=True)


def tensor_stats(name: str, t: torch.Tensor) -> None:
    arr = t.detach().float().cpu().numpy().ravel()
    n_nan  = int(np.isnan(arr).sum())
    n_inf  = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    if len(finite):
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype} "
              f"min={finite.min():.4g} max={finite.max():.4g} mean={finite.mean():.4g} "
              f"nan={n_nan} inf={n_inf}", flush=True)
    else:
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype} ALL NaN/Inf "
              f"nan={n_nan} inf={n_inf}", flush=True)


def check_model_weights(model: torch.nn.Module) -> None:
    banner("Model weight NaN/Inf check")
    n_params = 0
    n_nan_params = 0
    n_inf_params = 0
    for name, p in model.named_parameters():
        n_params += 1
        arr = p.detach().float().cpu().numpy().ravel()
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())
        if has_nan or has_inf:
            n_nan_params += has_nan
            n_inf_params += has_inf
            print(f"  PROBLEM param: {name}  nan={has_nan} inf={has_inf}  shape={list(p.shape)}", flush=True)
    print(f"  Total params: {n_params}  params_with_nan: {n_nan_params}  params_with_inf: {n_inf_params}", flush=True)
    if n_nan_params == 0 and n_inf_params == 0:
        print("  All weights look finite.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ProteinEBM single-example debug script")
    parser.add_argument("--pdb",        default=DEFAULT_PDB,        help="Path to a decoy PDB file")
    parser.add_argument("--config",     default=DEFAULT_CONFIG,     help="ProteinEBM config YAML")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="ProteinEBM checkpoint .pt")
    parser.add_argument("--t",          type=float, default=0.05,   help="Diffusion time t (default: 0.05)")
    parser.add_argument("--no-template-self-condition", dest="template_self_condition",
                        action="store_false", default=True)
    args = parser.parse_args()

    banner("Configuration")
    print(f"  PDB:        {args.pdb}", flush=True)
    print(f"  Config:     {args.config}", flush=True)
    print(f"  Checkpoint: {args.checkpoint}", flush=True)
    print(f"  t:          {args.t}", flush=True)
    print(f"  template_self_condition: {args.template_self_condition}", flush=True)

    for path, label in [(args.pdb, "PDB"), (args.config, "config"), (args.checkpoint, "checkpoint")]:
        if not os.path.exists(path):
            print(f"\nERROR: {label} file not found: {path}", flush=True)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        mem_free, mem_total = torch.cuda.mem_get_info(device)
        print(f"  GPU: {props.name}  VRAM free/total: {mem_free/1e9:.1f}/{mem_total/1e9:.1f} GB", flush=True)

    # ── Load model ──────────────────────────────────────────────────────────
    banner("Loading ProteinEBM model")
    sys.path.insert(0, "/home/ubuntu/ProteinEBM")
    from proteinfoundation.af2rank_evaluation.proteinebm_scorer import (
        load_proteinebm_model,
        build_input_feats_from_pdb,
    )

    model, config = load_proteinebm_model(args.config, args.checkpoint, device)
    print(f"  Model loaded. Parameter count: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    check_model_weights(model)

    # ── Build input features ─────────────────────────────────────────────────
    banner("Building input features from PDB")
    feats = build_input_feats_from_pdb(
        pdb_path=args.pdb,
        model=model,
        t=args.t,
        template_self_condition=args.template_self_condition,
        device=device,
    )

    print("  Input feature tensors:", flush=True)
    for key, val in feats.items():
        tensor_stats(key, val)

    # ── Forward pass ────────────────────────────────────────────────────────
    banner("Running model.compute_energy()")

    # Register forward hooks on all submodules to catch the first NaN
    nan_detected = []

    def make_hook(name):
        def hook(module, input, output):
            if nan_detected:
                return
            outs = output if isinstance(output, (list, tuple)) else [output]
            for o in outs:
                if isinstance(o, torch.Tensor):
                    if o.isnan().any() or o.isinf().any():
                        nan_detected.append(name)
                        print(f"\n  !!! First NaN/Inf detected in output of: {name}", flush=True)
                        tensor_stats(f"  output of {name}", o)
        return hook

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            out = model.compute_energy(feats)
    finally:
        for h in hooks:
            h.remove()

    banner("Output")
    if isinstance(out, dict):
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                tensor_stats(key, val)
            else:
                print(f"  {key}: {val}", flush=True)
        energy_val = out.get("energy")
        if energy_val is not None:
            e = float(energy_val.detach().cpu().item()) if energy_val.numel() == 1 else energy_val.detach().cpu().numpy()
            print(f"\n  >>> ENERGY = {e}", flush=True)
            if isinstance(e, float) and (e != e):  # NaN check
                print("  >>> RESULT: NaN energy!", flush=True)
            else:
                print("  >>> RESULT: energy is finite.", flush=True)
    else:
        print(f"  Output type: {type(out)}", flush=True)
        print(f"  Output: {out}", flush=True)

    if nan_detected:
        print(f"\n  First NaN/Inf was produced in module: {nan_detected[0]}", flush=True)
    else:
        print("\n  No NaN/Inf detected in any intermediate module output.", flush=True)


if __name__ == "__main__":
    main()
