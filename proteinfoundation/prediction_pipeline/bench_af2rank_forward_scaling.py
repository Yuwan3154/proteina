#!/usr/bin/env python3
"""Microbenchmark: AF2Rank/OpenFold model_1_ptm forward-pass wall-clock vs sequence length.

Times only the inner AlphaFold forward call (`scorer.model.model(batch)`) with
`torch.cuda.synchronize()` fencing. Synthetic poly-alanine CA-only inputs at
representative lengths. Single GPU, no concurrent shards. Recycles=6, deepspeed
evo attn disabled (production knobs).
"""
import argparse
import csv
import gc
import math
import sys
import time
import statistics

import numpy as np
import torch

sys.path.insert(0, "/home/jupyter-chenxi/proteina")
from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank


def make_synth_pdb(L, out_path):
    lines = []
    for i in range(L):
        x = 1.5 * i
        y = 0.5 * math.sin(i * 0.6)
        z = 0.5 * math.cos(i * 0.6)
        lines.append(
            "ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C" % (i+1, i, x, y, z)
        )
    lines.append("TER")
    lines.append("END")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[64, 96, 128, 192, 256, 320, 384, 448, 512])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--recycles", type=int, default=6)
    p.add_argument("--model_name", default="model_1_ptm")
    p.add_argument("--out_csv", default="/tmp/af2rank_forward_scaling.csv")
    p.add_argument("--summary_csv", default="/tmp/af2rank_forward_summary.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bootstrap_L = args.lengths[0]
    bootstrap_pdb = "/tmp/synth_ref_bootstrap.pdb"
    make_synth_pdb(bootstrap_L, bootstrap_pdb)

    print(f"Loading OpenFold {args.model_name} (recycles={args.recycles}) ...")
    scorer = OpenFoldAF2Rank(
        reference_pdb=bootstrap_pdb,
        chain="A",
        model_name=args.model_name,
        recycles=args.recycles,
        debug=False,
        chunk_size=None,
        use_deepspeed_evoformer_attention=False,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
        skip_ref_metrics=True,
    )
    print("Model loaded. Starting length sweep.\n")

    rows = []
    summary_rows = []

    for L in args.lengths:
        ref_pdb = f"/tmp/synth_ref_L{L}.pdb"
        make_synth_pdb(L, ref_pdb)
        scorer.reset_reference(ref_pdb, chain="A")

        try:
            batch, _ = scorer._featurize(ref_pdb, decoy_chain="A", seed=args.seed)
        except Exception as e:
            print(f"L={L}: featurize failed: {type(e).__name__}: {e}")
            continue

        warmup_failed = False
        for w in range(args.warmup):
            try:
                with torch.no_grad():
                    _ = scorer.model.model(batch)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as e:
                print(f"L={L}: OOM during warmup {w}: {e}")
                warmup_failed = True
                break
        if warmup_failed:
            torch.cuda.empty_cache()
            gc.collect()
            continue

        rep_times = []
        rep_peaks = []
        for rep in range(args.reps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    out = scorer.model.model(batch)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as e:
                print(f"L={L}: OOM during rep {rep}: {e}")
                break
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            rep_times.append(elapsed_ms)
            rep_peaks.append(peak_mb)
            rows.append({"L": L, "rep": rep, "forward_ms": elapsed_ms, "peak_gpu_mb": peak_mb})

        if rep_times:
            med = statistics.median(rep_times)
            mean = statistics.mean(rep_times)
            std = statistics.stdev(rep_times) if len(rep_times) > 1 else 0.0
            print(f"L={L:>4d}  n={len(rep_times):>2d}  median={med:>9.1f} ms  mean={mean:>9.1f} ms  "
                  f"std={std:>7.1f} ms ({100*std/mean:>4.1f}%)  peak={max(rep_peaks):>6.0f} MB")
            summary_rows.append({
                "L": L, "n": len(rep_times),
                "median_ms": med, "mean_ms": mean, "std_ms": std,
                "min_ms": min(rep_times), "max_ms": max(rep_times),
                "peak_mb": max(rep_peaks),
            })

        del batch
        try: del out
        except NameError: pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["L", "rep", "forward_ms", "peak_gpu_mb"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-rep -> {args.out_csv} ({len(rows)} rows)")

    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["L", "n", "median_ms", "mean_ms", "std_ms",
                                          "min_ms", "max_ms", "peak_mb"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote summary  -> {args.summary_csv} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
