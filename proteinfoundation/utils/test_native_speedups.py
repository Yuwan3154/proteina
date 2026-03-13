#!/usr/bin/env python
"""Benchmark OpenFold native speedups: deepspeed evoformer attention, cuequivariance ops.

Tests each flag individually and in combination, measuring model forward time.

Usage:
    python -u test_native_speedups.py [--seq_len 50 100] [--max_recycling_iters 1]
"""
import argparse
import copy
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


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
    """Benchmark a single configuration, return avg seconds per forward."""
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
        return None

    dgram, rtype, mask = make_dummy_inputs(seq_len, device=device)
    batch = infer.build_batch(dgram, rtype, mask)

    try:
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                infer.model(copy.deepcopy(batch))
        _sync()

        # Measure
        t0 = time.time()
        for _ in range(n_measure):
            with torch.no_grad():
                infer.model(copy.deepcopy(batch))
            _sync()
        avg = (time.time() - t0) / n_measure

        p(f"  [{config_name}] seq_len={seq_len}: {avg:.3f}s/iter")
    except Exception as e:
        p(f"  [{config_name}] FAILED during forward: {e}")
        avg = None

    del infer
    torch.cuda.empty_cache()
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--model_name", type=str, default="model_1_ptm")
    parser.add_argument("--jax_params", type=str,
                        default=os.path.expanduser("~/openfold/openfold/resources/params/params_model_1_ptm.npz"))
    parser.add_argument("--max_recycling_iters", type=int, default=1)
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_measure", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"Device: {device}")

    # Configurations to benchmark
    configs = {
        # True baseline: all optimisations disabled.
        "baseline": {
            "use_deepspeed_evoformer_attention": False,
        },
        "deepspeed_attn": {
            "use_deepspeed_evoformer_attention": True,
        },
        "cue_attn": {
            "use_cuequivariance_attention": True,
        },
        "cue_mul": {
            "use_cuequivariance_multiplicative_update": True,
        },
        "cue_both": {
            "use_cuequivariance_attention": True,
            "use_cuequivariance_multiplicative_update": True,
        },
        # NOTE: when both deepspeed AND cue_attn are enabled, primitives.py gives
        # deepspeed priority for ALL attention (MSA row/col + triangle).  So this config
        # is effectively deepspeed for all attention + cueq for multiplicative update only
        # (NOT cueq for triangle attention).  See deepspeed_msa+cue_both below for the
        # true "deepspeed MSA + cueq pair" config recommended by OpenFold2 docs.
        "deepspeed+cue_both": {
            "use_deepspeed_evoformer_attention": True,
            "use_cuequivariance_attention": True,
            "use_cuequivariance_multiplicative_update": True,
        },
        # deepspeed for MSA attention (row/col) only, cuequivariance for
        # triangle multiplicative updates only.  No cueq triangle attention.
        "deepspeed+cue_mul_only": {
            "use_deepspeed_evoformer_attention": True,
            "use_cuequivariance_multiplicative_update": True,
        },
        # deepspeed priority for ALL attention — verifies the priority flip in primitives.py.
        # Equivalent to deepspeed+cue_both (cueq attention is bypassed everywhere).
        "deepspeed_priority_all": {
            "use_deepspeed_evoformer_attention": True,
            "use_cuequivariance_attention": True,  # deepspeed wins due to priority flip
            "use_cuequivariance_multiplicative_update": True,
        },
        # Recommended by OpenFold2 docs: deepspeed for MSA row/col attention,
        # cuequivariance for BOTH pair operations (triangle attention + multiplicative update).
        # Achieved by patching PairStack.forward (use_cuequivariance_triangle_attention)
        # while global cue_attn=False keeps MSA attention on deepspeed.
        "deepspeed_msa+cue_both": {
            "use_deepspeed_evoformer_attention": True,
            "use_cuequivariance_triangle_attention": True,
            "use_cuequivariance_multiplicative_update": True,
        },
        "compile_dynamic_true": {
            "compile_model": False,  # we'll manually compile with dynamic=True
        },
        "all_speedups": {
            "use_deepspeed_evoformer_attention": True,
            "use_cuequivariance_attention": True,
            "use_cuequivariance_multiplicative_update": True,
            "compile_model": False,  # we'll manually compile with dynamic=True
        },
    }

    for sl in args.seq_len:
        p(f"\n{'='*60}")
        p(f"  Sequence length = {sl}")
        p(f"{'='*60}")

        results = {}
        for name, kwargs in configs.items():
            # For compile configs, we manually enable after loading
            needs_compile = name in ("compile_dynamic_true", "all_speedups")
            avg = benchmark_config(
                name, args.model_name, args.jax_params, device, sl,
                args.max_recycling_iters, args.n_warmup, args.n_measure,
                **kwargs,
            )
            if avg is not None and needs_compile:
                # Re-run with compilation
                p(f"  [{name}] Re-running with torch.compile(dynamic=True)...")
                try:
                    infer = OpenFoldTemplateInference(
                        model_name=args.model_name,
                        jax_params_path=args.jax_params,
                        device=device,
                        max_recycling_iters=args.max_recycling_iters,
                        use_mlm=False,
                        **kwargs,
                    )
                    infer.enable_compilation(dynamic=True)
                    dgram, rtype, mask = make_dummy_inputs(sl, device=device)
                    batch = infer.build_batch(dgram, rtype, mask)
                    # Warmup (extra for compilation)
                    for _ in range(args.n_warmup + 2):
                        with torch.no_grad():
                            infer.model(copy.deepcopy(batch))
                    _sync()
                    t0 = time.time()
                    for _ in range(args.n_measure):
                        with torch.no_grad():
                            infer.model(copy.deepcopy(batch))
                        _sync()
                    avg = (time.time() - t0) / args.n_measure
                    p(f"  [{name}] seq_len={sl}: {avg:.3f}s/iter (compiled)")
                    del infer
                    torch.cuda.empty_cache()
                except Exception as e:
                    p(f"  [{name}] FAILED during compiled forward: {e}")
                    avg = None

            results[name] = avg

        # Summary table
        baseline = results.get("baseline")
        if baseline:
            p(f"\n  --- Summary (seq_len={sl}) ---")
            p(f"  {'Config':30s} {'Time':>8s} {'Speedup':>8s}")
            p(f"  {'-'*48}")
            for name, avg in results.items():
                if avg is not None:
                    speedup = baseline / avg
                    p(f"  {name:30s} {avg:8.3f}s {speedup:7.2f}x")

    p("\n=== BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
