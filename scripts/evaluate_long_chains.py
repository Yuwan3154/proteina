#!/usr/bin/env python3
"""Evaluate Frame2ConFind checkpoint on longer protein chains [385, 768].
Selects 128 protein chains stratified by length, runs ConFind in parallel,
performs model inference, and computes evaluation metrics.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add paths to sys.path
sys.path.insert(0, "/home/ubuntu/Frame2ConFind")
sys.path.insert(0, "/home/ubuntu/proteina")

from Frame2ConFind.data.dataset import ConfindPDBDataset, collate
from Frame2ConFind.models.frame2seq_contact import Frame2seqContact
from Frame2ConFind.train.common import binary_auroc_aupr, get_amp_dtype


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/home/ubuntu/Frame2ConFind/runs/f2s_ft_max384_pair_ebs16_no-sin-pos-emb/best.pt",
        help="Path to the checkpoint best.pt",
    )
    parser.add_argument(
        "--csv-path",
        default="/home/ubuntu/proteina/data/pdb_train/df_pdb_f1_minl50_maxl768_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.csv",
        help="Path to the full PDB dataset CSV metadata",
    )
    parser.add_argument(
        "--processed-dir",
        default="/home/ubuntu/proteina/data/pdb_train/processed",
        help="Directory with processed .pt files",
    )
    parser.add_argument(
        "--rotlib",
        default="/home/ubuntu/proteina/data/rotlibs",
        help="MSL-formatter rotamer library path",
    )
    parser.add_argument(
        "--confind-bin",
        default="/home/ubuntu/.local/bin/confind",
        help="Path to confind executable",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/ubuntu/Frame2ConFind/runs/f2s_ft_max384_pair_ebs16_no-sin-pos-emb/evaluation_385_768",
        help="Output directory for CSV and JSON summary",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.01, help="Contact probability threshold")
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16", help="Precision mode")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel ConFind workers")
    return parser.parse_args()


def load_model(checkpoint_path: str, device):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg", {})
    
    # Instantiate f2s model using config from checkpoint
    model = Frame2seqContact(
        zero_absolute_positional_embedding=bool(
            cfg.get("zero_absolute_positional_embedding", True)
        ),
        contact_head_mode=cfg.get("contact_head_mode", "pair"),
    )
    
    sd = state["model_state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod.") :]] = v
        else:
            new_sd[k] = v
            
    model.load_state_dict(new_sd, strict=True)
    model = model.to(device).eval()
    print(f"Loaded checkpoint from {checkpoint_path}", flush=True)
    print(f"Model positional embedding config: zero_absolute_positional_embedding={model.zero_absolute_positional_embedding}", flush=True)
    print(f"Model contact head config: contact_head_mode={model.contact_head_mode}", flush=True)
    return model


def worker_confind(pid, pt_path, rotlib, confind_bin):
    """Worker process to load a protein graph and run ConFind ground-truth map calculation."""
    import sys
    sys.path.insert(0, "/home/ubuntu/proteina")
    import torch
    from proteinfoundation.utils.confind_utils import confind_raw_contact_map
    
    try:
        graph = torch.load(pt_path, map_location="cpu", weights_only=False)
        target_map = confind_raw_contact_map(
            graph,
            rotlib_path=rotlib,
            confind_bin=confind_bin,
            omp_threads=1,
            renumber=True,
        )
        return pid, target_map, None
    except Exception as e:
        return pid, None, str(e)


def per_protein_metrics(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    L_valid: int,
    threshold: float = 0.01,
) -> dict:
    probs = probs.float()
    target = target.float()
    pairs_mask_2d = mask.bool()
    flat_mask = pairs_mask_2d.reshape(-1)
    p = probs.reshape(-1)[flat_mask]
    t = target.reshape(-1)[flat_mask]

    mse = float(((p - t) ** 2).mean())
    mae = float((p - t).abs().mean())
    
    # Pearson Correlation
    pc = p - p.mean()
    tc = t - t.mean()
    denom = (pc.pow(2).sum().sqrt() * tc.pow(2).sum().sqrt()).clamp(min=1e-8)
    pearson = float((pc * tc).sum() / denom)

    # AUROC & AUPR
    y = (t >= threshold).to(torch.float32)
    auroc, aupr = binary_auroc_aupr(p, y)

    out = {"mse": mse, "mae": mae, "pearson": pearson, "auroc": auroc, "aupr": aupr}

    L = probs.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    sep = (jj - ii).to(probs.device)

    for range_name, min_sep in [("medium", 12), ("long", 24)]:
        consider = (sep >= min_sep) & pairs_mask_2d
        if consider.sum() == 0:
            for k in (1, 2, 5):
                out[f"P@L/{k}_{range_name}"] = float("nan")
            continue
        p_range = probs[consider]
        y_range = (target[consider] >= threshold).to(torch.float32)
        for k in (1, 2, 5):
            topk = min(max(1, L_valid // k), p_range.numel())
            idx = torch.topk(p_range, topk).indices
            if y_range.sum() == 0:
                out[f"P@L/{k}_{range_name}"] = float("nan")
            else:
                out[f"P@L/{k}_{range_name}"] = float(y_range[idx].mean().item())
    return out


def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = get_amp_dtype(args.amp)
    
    processed_dir = Path(args.processed_dir)
    
    print("Step 1: Selecting 128 stratified protein chains in range [385, 768]...", flush=True)
    
    # Define length bins
    bins = [
        ("385-480", 385, 480),
        ("481-576", 481, 576),
        ("577-672", 577, 672),
        ("673-768", 673, 768),
    ]
    
    bin_candidates = {name: [] for name, _, _ in bins}
    
    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            length = int(row["length"])
            
            # Check if corresponding .pt file exists
            pt_path = processed_dir / f"{pid}.pt"
            if not pt_path.exists():
                continue
                
            for name, low, high in bins:
                if low <= length <= high:
                    bin_candidates[name].append((pid, length))
                    break
                    
    selected_ids = []
    selected_lengths = {}
    selected_bins = {}
    
    for name, low, high in bins:
        candidates = bin_candidates[name]
        print(f"  Bin {name}: found {len(candidates)} candidates.", flush=True)
        if len(candidates) < 32:
            raise RuntimeError(f"Not enough candidates in bin {name} (need 32, found {len(candidates)})")
            
        sampled = random.sample(candidates, 32)
        for pid, length in sampled:
            selected_ids.append(pid)
            selected_lengths[pid] = length
            selected_bins[pid] = name
            
    print(f"Successfully selected {len(selected_ids)} stratified protein chains.", flush=True)
    
    # Precompute ConFind ground-truth contact maps in parallel
    print(f"Step 2: Precomputing ConFind ground-truth maps in parallel with {args.workers} workers...", flush=True)
    confind_maps = {}
    failed_pids = []
    
    worker_args = [
        (pid, processed_dir / f"{pid}.pt", args.rotlib, args.confind_bin)
        for pid in selected_ids
    ]
    
    confind_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(worker_confind, *arg): arg[0]
            for arg in worker_args
        }
        
        completed = 0
        for future in as_completed(futures):
            pid, target_map, err = future.result()
            completed += 1
            if err is not None:
                print(f"  [{completed}/128] ERROR: ConFind failed for {pid}: {err}", flush=True)
                failed_pids.append(pid)
            else:
                confind_maps[pid] = target_map
                elapsed = time.perf_counter() - confind_start
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (len(selected_ids) - completed) / rate if rate > 0 else 0.0
                print(f"  [{completed}/128] ConFind completed for {pid} ({elapsed:.1f}s elapsed, rate={rate:.2f}/s, eta={eta/60.0:.1f}m)", flush=True)
                
    total_confind_sec = time.perf_counter() - confind_start
    print(f"ConFind precomputation finished in {total_confind_sec/60.0:.2f}m. Failed: {len(failed_pids)}", flush=True)
    
    # Filter selected_ids to successful ones
    successful_ids = [pid for pid in selected_ids if pid in confind_maps]
    if not successful_ids:
        raise RuntimeError("All ConFind runs failed! Cannot proceed with evaluation.")
        
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load dataset with require_confind=False
    ds = ConfindPDBDataset(
        ids=successful_ids,
        processed_dir=str(processed_dir),
        max_len=1000, # Large enough to avoid any cropping
        random_crop=False,
        require_confind=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    
    rows = []
    print("Step 3: Run model prediction and compute metrics...", flush=True)
    
    for bi, batch in enumerate(loader):
        pid = batch["ids"][0]
        L = int(batch["lengths"][0].item())
        bin_name = selected_bins[pid]
        
        target_map = confind_maps[pid]
        target_tensor = torch.as_tensor(target_map, dtype=torch.float32)
        
        # Run forward pass of Frame2ConFind model
        model_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            logits, probs = model(x_f2s=batch["x_f2s"].to(device), mask=batch["mask"].to(device))
            
        model_sec = time.perf_counter() - model_start
        
        # Post-process outputs
        probs_single = probs[0, :L, :L].float().cpu()
        target_single = target_tensor[:L, :L]
        mask_single = batch["contact_mask"][0, :L, :L]
        mask_1d = batch["mask"][0, :L].bool()
        L_valid = int(mask_1d.sum().item())
        
        # Calculate metrics
        mets = per_protein_metrics(probs_single, target_single, mask_single, L_valid, args.threshold)
        mets["id"] = pid
        mets["L"] = L
        mets["L_valid"] = L_valid
        mets["bin"] = bin_name
        mets["model_time_sec"] = model_sec
        
        rows.append(mets)
        if (bi + 1) % 10 == 0 or bi == len(loader) - 1:
            print(f"  [{bi+1}/{len(loader)}] Evaluated {pid}: Pearson={mets['pearson']:.3f} AUROC={mets['auroc']:.3f}", flush=True)
        
    # Write CSV
    keys = sorted({k for r in rows for k in r.keys()})
    ordered = ["id", "bin", "L", "L_valid", "model_time_sec"] + [k for k in keys if k not in {"id", "bin", "L", "L_valid", "model_time_sec"}]
    
    csv_path = out_dir / "per_protein_long.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote detailed CSV to {csv_path}", flush=True)
    
    # Aggregate summary
    numeric_keys = [k for k in ordered if k not in {"id", "bin", "L", "L_valid"}]
    summary = {}
    for k in numeric_keys:
        vals = np.array([r[k] for r in rows if not (r[k] is None or (isinstance(r[k], float) and np.isnan(r[k])))])
        if vals.size == 0:
            summary[f"{k}_mean"] = float("nan")
            summary[f"{k}_median"] = float("nan")
        else:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_median"] = float(np.median(vals))
            
    # Calculate stratified summary per bin
    summary["stratified"] = {}
    for name, _, _ in bins:
        bin_rows = [r for r in rows if r["bin"] == name]
        summary["stratified"][name] = {}
        for k in ["pearson", "auroc", "aupr", "P@L/1_long", "P@L/5_long", "mse", "mae"]:
            vals = np.array([r[k] for r in bin_rows if not (r[k] is None or (isinstance(r[k], float) and np.isnan(r[k])))])
            if vals.size == 0:
                summary["stratified"][name][f"{k}_mean"] = float("nan")
            else:
                summary["stratified"][name][f"{k}_mean"] = float(np.mean(vals))
                
    summary["n_proteins"] = len(rows)
    summary["checkpoint"] = args.checkpoint
    summary["threshold"] = args.threshold
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    json_path = out_dir / "summary_long.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary JSON to {json_path}", flush=True)
    
    # Print overall mean metrics
    print("\n=== Overall Evaluation Summary ===", flush=True)
    print(f"Total evaluated: {len(rows)} proteins", flush=True)
    print(f"Mean Pearson Correlation: {summary['pearson_mean']:.4f}", flush=True)
    print(f"Mean AUROC:               {summary['auroc_mean']:.4f}", flush=True)
    print(f"Mean AUPR:                {summary['aupr_mean']:.4f}", flush=True)
    print(f"Mean MSE:                 {summary['mse_mean']:.6f}", flush=True)
    print(f"Mean MAE:                 {summary['mae_mean']:.6f}", flush=True)
    print(f"Mean P@L/1 Long-range:    {summary['P@L/1_long_mean']:.4f}", flush=True)
    
    print("\n=== Stratified Bin Mean Pearson ===", flush=True)
    for name, _, _ in bins:
        print(f"  Bin {name}: {summary['stratified'][name]['pearson_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
