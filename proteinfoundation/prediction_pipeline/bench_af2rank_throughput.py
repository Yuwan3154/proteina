#!/usr/bin/env python3
"""AF2Rank throughput benchmark: featurize vs GPU-forward split, fp32 vs fp16.

Goal: decide whether af2rank can scale from top_k=8 to ~2000 samples/protein.
The oracle-convergence run showed ~29 s wall PER TEMPLATE (median gap between
"Featurizing template" log lines) — this script attributes that time to its two
phases so we know what to optimise:

  phase A = featurize  : OpenFold template pipeline (CIF parse -> kalign ->
                          build template feats).  PURE CPU, NOT amortizable by
                          XLA/torch.compile, runs once per template.
  phase B = forward    : AlphaFold forward (recycles=3) on GPU.  THIS is what a
                          compiled-JAX / fp16 path would speed up.

It loops over N real or synthetic CA-only templates with ONE protein/reference
(model + reference loaded once), timing each phase separately, then reports
samples/sec and the extrapolated wall for 2000 samples/protein.

fp16: wraps only the GPU forward in torch.autocast(float16); featurization is
left fp32 (it is I/O + numpy and not the bottleneck).
"""
import argparse
import gc
import math
import os
import sys
import time
import statistics
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/proteina"))
sys.path.insert(0, os.path.expanduser("~/openfold"))

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank


def make_synth_pdb(L, out_path, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(L):
        x = 1.5 * i + jitter * rng.standard_normal()
        y = 4.0 * math.sin(i * 0.35) + jitter * rng.standard_normal()
        z = 4.0 * math.cos(i * 0.35) + jitter * rng.standard_normal()
        lines.append("ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C"
                     % (i + 1, i, x, y, z))
    lines += ["TER", "END"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=64, help="number of templates to score")
    p.add_argument("--L", type=int, default=200, help="sequence length (synthetic)")
    p.add_argument("--recycles", type=int, default=3)
    p.add_argument("--model_name", default="model_1_ptm")
    p.add_argument("--fp16", action="store_true", help="autocast(float16) the GPU forward")
    p.add_argument("--use_deepspeed_evoformer_attention", action="store_true", default=False)
    p.add_argument("--use_cuequivariance_attention", action="store_true", default=False)
    p.add_argument("--use_cuequivariance_multiplicative_update", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tmpdir = tempfile.mkdtemp(prefix="af2rank_bench_")
    ref_pdb = os.path.join(tmpdir, "ref.pdb")
    make_synth_pdb(args.L, ref_pdb, jitter=0.0, seed=0)

    # Distinct templates (jittered) so featurization can't trivially cache.
    tmpl_pdbs = []
    for i in range(args.N):
        tp = os.path.join(tmpdir, f"tmpl_{i:04d}.pdb")
        make_synth_pdb(args.L, tp, jitter=1.0, seed=i + 1)
        tmpl_pdbs.append(tp)

    print(f"Loading OpenFold {args.model_name} (recycles={args.recycles}, fp16={args.fp16}) ...")
    t_load0 = time.perf_counter()
    scorer = OpenFoldAF2Rank(
        reference_pdb=ref_pdb,
        chain="A",
        model_name=args.model_name,
        recycles=args.recycles,
        debug=False,
        chunk_size=None,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        use_cuequivariance_attention=args.use_cuequivariance_attention,
        use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
        skip_ref_metrics=True,  # isolate model cost: no USalign subprocess
    )
    t_load = time.perf_counter() - t_load0
    print(f"Model loaded in {t_load:.1f}s. Scoring {args.N} templates (L={args.L}).\n")

    feat_times, fwd_times = [], []
    peak_mb = 0.0
    for i, tp in enumerate(tmpl_pdbs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch, _tc, seg_ids = scorer._featurize(tp, decoy_chain="A", seed=args.seed)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if args.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _out = scorer.model.model(batch)
            else:
                _out = scorer.model.model(batch)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / (1024 ** 2))
        if i >= 2:  # drop first 2 as warmup
            feat_times.append(t1 - t0)
            fwd_times.append(t2 - t1)
        del batch, _out
        gc.collect()
        torch.cuda.empty_cache()

    fmed, ffwd = statistics.median(feat_times), statistics.median(fwd_times)
    per_tmpl = fmed + ffwd
    print("=== per-template (median, warmup-dropped) ===")
    print(f"  featurize : {fmed:6.2f} s  (CPU, not GPU-amortizable)")
    print(f"  forward   : {ffwd:6.2f} s  (GPU, recycles={args.recycles}, fp16={args.fp16})")
    print(f"  total     : {per_tmpl:6.2f} s   -> {1.0/per_tmpl:.3f} templates/sec")
    print(f"  peak GPU  : {peak_mb:.0f} MB")
    print()
    print("=== extrapolation to 2000 samples/protein (1 model, this backend) ===")
    print(f"  per protein : {2000*per_tmpl/3600:6.2f} GPU-hr")
    print(f"  (feat-only) : {2000*fmed/3600:6.2f} GPU-hr   (forward-only): {2000*ffwd/3600:6.2f} GPU-hr")
    print(f"  8 proteins  : {8*2000*per_tmpl/3600:6.1f} GPU-hr  on 8xV100 -> {8*2000*per_tmpl/3600/8:.1f} hr wall")


if __name__ == "__main__":
    main()
