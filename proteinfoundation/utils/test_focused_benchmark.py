#!/usr/bin/env python
"""Focused benchmark: deepspeed MSA + cueq pair ops vs deepspeed+cue_mul only.

Tests at seq_len 128, 256, 512 (production-relevant sizes).

Usage:
    python -u test_focused_benchmark.py [--seq_len 128 256 512]
"""
import os
import sys

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from test_native_speedups import benchmark_config, p

import argparse
import torch

import os
JAX_PARAMS = os.path.expanduser("~/openfold/openfold/resources/params/params_model_1_ptm.npz")

CONFIGS = {
    "baseline": {
        "use_deepspeed_evoformer_attention": False,
    },
    # deepspeed for ALL attention (row/col + triangle) + cueq for tri_mul only
    "deepspeed+cue_mul_only": {
        "use_deepspeed_evoformer_attention": True,
        "use_cuequivariance_multiplicative_update": True,
    },
    # deepspeed for MSA row/col only + cueq for tri_attn + cueq for tri_mul
    # (requires fixed monkey-patch that clears use_deepspeed_evo_attention in PairStack)
    "deepspeed_msa+cue_both": {
        "use_deepspeed_evoformer_attention": True,
        "use_cuequivariance_triangle_attention": True,
        "use_cuequivariance_multiplicative_update": True,
    },
}


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

    for sl in args.seq_len:
        p(f"\n{'='*60}")
        p(f"  Sequence length = {sl}")
        p(f"{'='*60}")

        results = {}
        for name, kwargs in CONFIGS.items():
            avg = benchmark_config(
                name, args.model_name, args.jax_params, device, sl,
                args.max_recycling_iters, args.n_warmup, args.n_measure,
                **kwargs,
            )
            results[name] = avg

        baseline = results.get("baseline")
        if baseline:
            p(f"\n  --- Summary (seq_len={sl}) ---")
            p(f"  {'Config':35s} {'Time':>8s} {'Speedup':>8s}")
            p(f"  {'-'*53}")
            for name, avg in results.items():
                if avg is not None:
                    speedup = baseline / avg
                    p(f"  {name:35s} {avg:8.3f}s {speedup:7.2f}x")

    p("\n=== BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
