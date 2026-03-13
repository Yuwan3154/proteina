#!/usr/bin/env python
"""Focused benchmark: deepspeed MSA + cueq pair ops vs deepspeed+cue_mul only.

Tests at seq_len 128, 256, 512 (production-relevant sizes).
Reports time, speedup, and peak GPU memory per config.

Usage:
    CUTLASS_PATH=/home/ubuntu/openfold/cutlass \\
    python -u test_focused_benchmark.py [--seq_len 128 256 512]
"""
import argparse
import copy
import os
import sys
import time

import torch

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference

JAX_PARAMS = os.path.expanduser("~/openfold/openfold/resources/params/params_model_1_ptm.npz")

CONFIGS = {
    "baseline": {
        "use_deepspeed_evoformer_attention": False,
    },
    # deepspeed for ALL attn (row/col + triangle) + cueq for tri_mul only
    "deepspeed+cue_mul_only": {
        "use_deepspeed_evoformer_attention": True,
        "use_cuequivariance_multiplicative_update": True,
    },
    # deepspeed for MSA row/col only + cueq for tri_attn + cueq for tri_mul
    # (monkey-patch clears use_deepspeed_evo_attention in PairStack so cueq wins)
    "deepspeed_msa+cue_both": {
        "use_deepspeed_evoformer_attention": True,
        "use_cuequivariance_triangle_attention": True,
        "use_cuequivariance_multiplicative_update": True,
    },
}


def p(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_dummy_inputs(seq_len, device):
    residue_type = torch.randint(0, 20, (1, seq_len), device=device)
    mask = torch.ones(1, seq_len, dtype=torch.float32, device=device)
    distogram_probs = torch.randn(1, seq_len, seq_len, 39, device=device).softmax(dim=-1)
    return distogram_probs, residue_type, mask


def benchmark_config(config_name, model_name, jax_params, device, seq_len,
                     max_recycling_iters, n_warmup=3, n_measure=5, **model_kwargs):
    """Benchmark a single configuration. Returns (avg_seconds, peak_gpu_mb) or (None, None)."""
    p(f"  [{config_name}] Loading model...")
    try:
        infer = OpenFoldTemplateInference(
            model_name=model_name,
            jax_params_path=jax_params,
            device=device,
            max_recycling_iters=max_recycling_iters,
            use_mlm=False,
            **model_kwargs,
        )
    except Exception as e:
        p(f"  [{config_name}] FAILED to load: {e}")
        return None, None

    dgram, rtype, mask = make_dummy_inputs(seq_len, device=device)
    batch = infer.build_batch(dgram, rtype, mask)

    try:
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                infer.model(copy.deepcopy(batch))
        _sync()

        # Reset peak memory stats before measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Measure
        t0 = time.time()
        for _ in range(n_measure):
            with torch.no_grad():
                infer.model(copy.deepcopy(batch))
            _sync()
        avg = (time.time() - t0) / n_measure

        peak_mb = None
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

        p(f"  [{config_name}] seq_len={seq_len}: {avg:.3f}s/iter, peak GPU {peak_mb:.0f} MB")
    except Exception as e:
        p(f"  [{config_name}] FAILED during forward: {e}")
        avg, peak_mb = None, None

    del infer
    torch.cuda.empty_cache()
    return avg, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--model_name", type=str, default="model_1_ptm")
    parser.add_argument("--jax_params", type=str, default=JAX_PARAMS)
    parser.add_argument("--max_recycling_iters", type=int, default=1)
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_measure", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"Device: {device}")
    if torch.cuda.is_available():
        p(f"GPU: {torch.cuda.get_device_name(device)}")
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2
        p(f"Total GPU memory: {total_mb:.0f} MB")

    for sl in args.seq_len:
        p(f"\n{'='*60}")
        p(f"  Sequence length = {sl}")
        p(f"{'='*60}")

        results = {}
        for name, kwargs in CONFIGS.items():
            avg, peak_mb = benchmark_config(
                name, args.model_name, args.jax_params, device, sl,
                args.max_recycling_iters, args.n_warmup, args.n_measure,
                **kwargs,
            )
            results[name] = (avg, peak_mb)

        baseline_time = results.get("baseline", (None, None))[0]
        if baseline_time:
            p(f"\n  --- Summary (seq_len={sl}) ---")
            p(f"  {'Config':35s} {'Time':>8s} {'Speedup':>8s} {'Peak GPU MB':>12s}")
            p(f"  {'-'*67}")
            for name, (avg, peak_mb) in results.items():
                if avg is not None:
                    speedup = baseline_time / avg
                    mem_str = f"{peak_mb:>10.0f}" if peak_mb is not None else "        N/A"
                    p(f"  {name:35s} {avg:8.3f}s {speedup:7.2f}x {mem_str}")

    p("\n=== BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
