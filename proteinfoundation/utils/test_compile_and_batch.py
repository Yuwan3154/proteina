#!/usr/bin/env python
"""Tests and benchmarks for OpenFoldTemplateInference compilation, tracing, and batching.

Usage:
    python -u test_compile_and_batch.py [--skip-compile] [--skip-trace] [--skip-batch]
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

import openfold.np.residue_constants as rc
from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


def make_dummy_inputs(seq_len: int, device: torch.device = torch.device("cpu")):
    """Create dummy inputs for a given sequence length."""
    residue_type = torch.randint(0, 20, (1, seq_len), device=device)
    mask = torch.ones(1, seq_len, dtype=torch.float32, device=device)
    distogram_probs = torch.randn(1, seq_len, seq_len, 39, device=device).softmax(dim=-1)
    return distogram_probs, residue_type, mask


def p(*args, **kwargs):
    """print with flush=True."""
    print(*args, flush=True, **kwargs)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------

def test_backward_compat(infer: OpenFoldTemplateInference):
    """Original single-sample forward still works."""
    p("  Testing backward compatibility (single-sample forward)...")
    dgram, rtype, mask = make_dummy_inputs(30, device=infer.device)
    out = infer(dgram, rtype, mask)
    assert "final_atom_positions" in out, "Missing final_atom_positions"
    pos = out["final_atom_positions"]
    assert pos.shape[-3] >= 30, f"Unexpected shape {pos.shape}"
    assert pos.shape[-2] == 37
    assert pos.shape[-1] == 3
    p(f"    OK: output shape = {tuple(pos.shape)}")


def test_determinism(infer: OpenFoldTemplateInference):
    """With use_mlm=False the feature pipeline should be deterministic."""
    p("  Testing determinism (use_mlm=False)...")
    dgram, rtype, mask = make_dummy_inputs(30, device=infer.device)
    out1 = infer(dgram, rtype, mask)
    out2 = infer(dgram, rtype, mask)
    d = (out1["final_atom_positions"] - out2["final_atom_positions"]).abs().max().item()
    p(f"    Max diff between two runs: {d:.6f} Angstrom")
    assert d == 0.0, f"Expected deterministic output, got max diff = {d}"
    p("    OK: fully deterministic")


def test_compilation_toggle(infer: OpenFoldTemplateInference):
    """Test enable/disable compilation and verify output consistency."""
    p("  Testing compilation toggle (dynamic=True to handle variable shapes)...")
    dgram, rtype, mask = make_dummy_inputs(30, device=infer.device)
    batch = infer.build_batch(dgram, rtype, mask)

    # Run without compilation
    assert not infer.compiled
    with torch.no_grad():
        out_base = infer.model(copy.deepcopy(batch))
    pos_base = out_base["final_atom_positions"].detach().clone()

    # Enable compilation with dynamic=True (handles shape variation incl. batching)
    infer.enable_compilation(dynamic=True)
    assert infer.compiled

    # First compiled forward (triggers JIT compilation)
    p("    First compiled forward (triggers JIT compilation)...")
    t0 = time.time()
    with torch.no_grad():
        out_compiled = infer.model(copy.deepcopy(batch))
    p(f"    First compiled forward: {time.time() - t0:.2f}s")

    pos_compiled = out_compiled["final_atom_positions"].detach().clone()
    max_diff = (pos_base - pos_compiled).abs().max().item()
    mean_diff = (pos_base - pos_compiled).abs().mean().item()
    p(f"    Max diff (base vs compiled):  {max_diff:.2e} Angstrom")
    p(f"    Mean diff (base vs compiled): {mean_diff:.2e} Angstrom")
    assert max_diff < 1.5, f"Compiled output diverged too much: max_diff={max_diff}"
    assert mean_diff < 1.0, f"Compiled output mean diff too large: mean_diff={mean_diff}"

    # Second compiled forward (uses cached kernel)
    t0 = time.time()
    with torch.no_grad():
        infer.model(copy.deepcopy(batch))
    p(f"    Second compiled forward: {time.time() - t0:.2f}s")

    # Disable and verify restoration
    infer.disable_compilation()
    assert not infer.compiled

    with torch.no_grad():
        out_restored = infer.model(copy.deepcopy(batch))
    max_diff_restored = (pos_base - out_restored["final_atom_positions"]).abs().max().item()
    p(f"    Max diff (base vs restored): {max_diff_restored:.2e}")
    assert max_diff_restored < 1e-4, f"Restored output diverged: {max_diff_restored}"
    p("    OK: compilation toggle works")


def test_tracing(model_name, jax_params, max_recycling_iters):
    """Test enable_tracing via trace_model_ and verify output consistency."""
    p("  Testing trace_model_ (fresh model instance, trace_interval=64)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline (no tracing)
    infer_base = OpenFoldTemplateInference(
        model_name=model_name, jax_params_path=jax_params, device=device,
        max_recycling_iters=max_recycling_iters, use_mlm=False,
    )
    dgram, rtype, mask = make_dummy_inputs(50, device=device)
    batch = infer_base.build_batch(dgram, rtype, mask, _pad_to_length=64)
    with torch.no_grad():
        out_base = infer_base.model(copy.deepcopy(batch))
    pos_base = out_base["final_atom_positions"].detach().clone()

    # Traced model (separate instance — trace_model_ modifies in-place)
    infer_traced = OpenFoldTemplateInference(
        model_name=model_name, jax_params_path=jax_params, device=device,
        max_recycling_iters=max_recycling_iters, use_mlm=False,
    )
    # Build batch padded to 64 (same as baseline) then trace explicitly
    batch_t = infer_traced.build_batch(dgram, rtype, mask, _pad_to_length=64)
    p("    Calling enable_tracing (one-time tracing cost)...")
    t0 = time.time()
    infer_traced.enable_tracing(batch_t)
    p(f"    Tracing overhead: {time.time() - t0:.2f}s")
    p(f"    Traced length: {infer_traced._traced_length}")

    p("    Running traced forward...")
    t0 = time.time()
    with torch.no_grad():
        out_traced = infer_traced.model(copy.deepcopy(batch_t))
    p(f"    First traced forward: {time.time() - t0:.2f}s")

    pos_traced = out_traced["final_atom_positions"].detach().clone()
    # Both were padded to 64 — compare first 50 residues
    max_diff = (pos_base[:50] - pos_traced[:50]).abs().max().item()
    mean_diff = (pos_base[:50] - pos_traced[:50]).abs().mean().item()
    p(f"    Max diff (base vs traced):  {max_diff:.2e} Angstrom")
    p(f"    Mean diff (base vs traced): {mean_diff:.2e} Angstrom")
    assert max_diff < 1.5, f"Traced output diverged too much: max_diff={max_diff}"
    assert mean_diff < 1.0, f"Traced output mean diff too large: mean_diff={mean_diff}"

    # Second call — should reuse traced model, same batch
    t0 = time.time()
    with torch.no_grad():
        out_traced2 = infer_traced.model(copy.deepcopy(batch_t))
    p(f"    Second traced forward (cached): {time.time() - t0:.2f}s")

    d2 = (out_traced["final_atom_positions"] - out_traced2["final_atom_positions"]).abs().max().item()
    p(f"    Determinism check: {d2:.2e}")
    assert d2 == 0.0, f"Traced model non-deterministic: {d2}"
    p("    OK: tracing works and is deterministic")
    return infer_traced


def test_build_batch_padding(infer: OpenFoldTemplateInference):
    """Test that _pad_to_length creates correctly shaped features."""
    p("  Testing build_batch padding...")
    dgram, rtype, mask = make_dummy_inputs(30, device=infer.device)

    batch_nopad = infer.build_batch(dgram, rtype, mask)
    batch_pad = infer.build_batch(dgram, rtype, mask, _pad_to_length=50)

    for key in batch_nopad:
        if key not in batch_pad:
            continue
        s_nopad = batch_nopad[key].shape
        s_pad = batch_pad[key].shape
        if s_pad != s_nopad:
            p(f"    {key}: {s_nopad} -> {s_pad}")

    sm_pad = batch_pad.get("seq_mask")
    if sm_pad is not None:
        # seq_mask shape: [N_res, R], first dim is residues
        mask_slice = sm_pad[..., 0] if sm_pad.dim() >= 2 else sm_pad
        n_real = int(mask_slice.sum().item())
        n_total = mask_slice.numel()
        p(f"    seq_mask: {n_real}/{n_total} real residues")
        assert n_real == 30, f"Expected 30 real residues, got {n_real}"
    p("    OK: padding works correctly")


def test_batching_same_length(infer: OpenFoldTemplateInference):
    """Test batched inference with same-length sequences."""
    p("  Testing batching (same length)...")
    seq_len = 30
    n_samples = 3

    dgrams, rtypes, masks = [], [], []
    for _ in range(n_samples):
        d, r, m = make_dummy_inputs(seq_len, device=infer.device)
        dgrams.append(d)
        rtypes.append(r)
        masks.append(m)

    out_batched = infer.forward_batched(dgrams, rtypes, masks)
    pos_batched = out_batched["final_atom_positions"]
    p(f"    Batched output shape: {tuple(pos_batched.shape)}")
    assert pos_batched.shape[0] == n_samples
    assert pos_batched.shape[-2] == 37
    assert pos_batched.shape[-1] == 3

    # Batched samples should be distinct
    d01 = (pos_batched[0] - pos_batched[1]).abs().max().item()
    p(f"    Sample 0 vs 1 max diff = {d01:.2e}")
    assert d01 > 0.1, "Batched samples appear identical"

    # Model deterministic on same batch
    batch_multi = infer.build_batch_multi(dgrams, rtypes, masks)
    with torch.no_grad():
        out1 = infer.model(copy.deepcopy(batch_multi))
        out2 = infer.model(copy.deepcopy(batch_multi))
    d_det = (out1["final_atom_positions"] - out2["final_atom_positions"]).abs().max().item()
    p(f"    Determinism check: max diff = {d_det:.2e}")
    assert d_det == 0.0
    p("    OK: same-length batching works")


def test_batching_variable_length(infer: OpenFoldTemplateInference):
    """Test batched inference with different-length sequences."""
    p("  Testing batching (variable length)...")
    lengths = [25, 30, 35]

    dgrams, rtypes, masks = [], [], []
    for seq_len in lengths:
        d, r, m = make_dummy_inputs(seq_len, device=infer.device)
        dgrams.append(d)
        rtypes.append(r)
        masks.append(m)

    out_batched = infer.forward_batched(dgrams, rtypes, masks)
    pos_batched = out_batched["final_atom_positions"]
    p(f"    Batched output shape: {tuple(pos_batched.shape)}")
    assert pos_batched.shape[0] == len(lengths)
    assert pos_batched.shape[-3] >= max(lengths)
    p("    OK: variable-length batching works")


# ------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------

def benchmark_compilation(infer: OpenFoldTemplateInference, seq_len: int = 50,
                          n_warmup: int = 3, n_measure: int = 5):
    """Benchmark torch.compile vs uncompiled model forward (single sample)."""
    p(f"  Benchmarking torch.compile (seq_len={seq_len}, single sample)...")
    dgram, rtype, mask = make_dummy_inputs(seq_len, device=infer.device)
    batch = infer.build_batch(dgram, rtype, mask)

    # Uncompiled baseline
    infer.disable_compilation()
    for _ in range(n_warmup):
        with torch.no_grad():
            infer.model(copy.deepcopy(batch))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer.model(copy.deepcopy(batch))
        _sync()
    base_time = (time.time() - t0) / n_measure
    p(f"    Uncompiled: {base_time:.3f}s/iter")

    # Compiled (dynamic=True so it also works when we later benchmark with batching)
    infer.enable_compilation(dynamic=True)
    # Warm up: first call compiles, subsequent calls use cache
    p("    Warming up compiled model (first call triggers JIT)...")
    for _ in range(n_warmup + 2):
        with torch.no_grad():
            infer.model(copy.deepcopy(batch))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer.model(copy.deepcopy(batch))
        _sync()
    compiled_time = (time.time() - t0) / n_measure

    speedup = base_time / compiled_time if compiled_time > 0 else float("inf")
    p(f"    torch.compile: {compiled_time:.3f}s/iter  speedup={speedup:.2f}x")

    infer.disable_compilation()
    return base_time, compiled_time


def benchmark_tracing(model_name, jax_params, max_recycling_iters,
                      seq_len: int = 50, n_warmup: int = 3, n_measure: int = 5):
    """Benchmark trace_model_ (JIT trace) vs uncompiled model forward.

    Uses a fresh model instance because tracing is in-place and irreversible.
    Builds the batch explicitly at seq_len, then calls enable_tracing directly
    before measuring repeated model forwards.
    """
    p(f"  Benchmarking trace_model_ (seq_len={seq_len}, single sample)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Uncompiled baseline (new instance) ----
    infer_base = OpenFoldTemplateInference(
        model_name=model_name, jax_params_path=jax_params, device=device,
        max_recycling_iters=max_recycling_iters, use_mlm=False,
    )
    dgram, rtype, mask = make_dummy_inputs(seq_len, device=device)
    batch = infer_base.build_batch(dgram, rtype, mask, _pad_to_length=seq_len)

    for _ in range(n_warmup):
        with torch.no_grad():
            infer_base.model(copy.deepcopy(batch))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer_base.model(copy.deepcopy(batch))
        _sync()
    base_time = (time.time() - t0) / n_measure
    p(f"    Uncompiled: {base_time:.3f}s/iter")

    # ---- Traced (new instance) ----
    infer_traced = OpenFoldTemplateInference(
        model_name=model_name, jax_params_path=jax_params, device=device,
        max_recycling_iters=max_recycling_iters, use_mlm=False,
    )
    # Build the batch at seq_len on the traced instance (same weights, same batch)
    batch_t = infer_traced.build_batch(dgram, rtype, mask, _pad_to_length=seq_len)

    p("    Tracing model (one-time cost)...")
    t_trace_start = time.time()
    infer_traced.enable_tracing(batch_t)   # explicit trace call
    p(f"    Tracing overhead: {time.time() - t_trace_start:.2f}s")
    p(f"    Traced at length: {infer_traced._traced_length}")

    # Warmup on traced model
    for _ in range(n_warmup):
        with torch.no_grad():
            infer_traced.model(copy.deepcopy(batch_t))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer_traced.model(copy.deepcopy(batch_t))
        _sync()
    traced_time = (time.time() - t0) / n_measure

    speedup = base_time / traced_time if traced_time > 0 else float("inf")
    p(f"    trace_model_: {traced_time:.3f}s/iter  speedup={speedup:.2f}x")

    del infer_base, infer_traced
    return base_time, traced_time


def benchmark_batching(infer: OpenFoldTemplateInference, seq_len: int = 50,
                       n_samples: int = 4, n_warmup: int = 1, n_measure: int = 3):
    """Benchmark batched vs sequential model forward only (no feature pipeline)."""
    p(f"  Benchmarking model-only batching (seq_len={seq_len}, n_samples={n_samples})...")

    dgrams, rtypes, masks = [], [], []
    for _ in range(n_samples):
        d, r, m = make_dummy_inputs(seq_len, device=infer.device)
        dgrams.append(d)
        rtypes.append(r)
        masks.append(m)

    # Pre-build all batches (exclude feature pipeline time)
    single_batches = [infer.build_batch(d, r, m) for d, r, m in zip(dgrams, rtypes, masks)]
    multi_batch = infer.build_batch_multi(dgrams, rtypes, masks)

    # Sequential model forward
    for _ in range(n_warmup):
        for b in single_batches:
            with torch.no_grad():
                infer.model(copy.deepcopy(b))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        for b in single_batches:
            with torch.no_grad():
                infer.model(copy.deepcopy(b))
        _sync()
    seq_time = (time.time() - t0) / n_measure

    # Batched model forward
    for _ in range(n_warmup):
        with torch.no_grad():
            infer.model(copy.deepcopy(multi_batch))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer.model(copy.deepcopy(multi_batch))
        _sync()
    batch_time = (time.time() - t0) / n_measure

    speedup = seq_time / batch_time if batch_time > 0 else float("inf")
    p(f"    Sequential ({n_samples}x single fwd): {seq_time:.3f}s")
    p(f"    Batched    ({n_samples}x in 1 fwd):   {batch_time:.3f}s")
    p(f"    Speedup: {speedup:.2f}x  ({seq_time/n_samples:.3f}s/sample seq vs {batch_time/n_samples:.3f}s/sample batched)")
    return seq_time, batch_time


def benchmark_trace_plus_batch(model_name, jax_params, max_recycling_iters,
                                seq_len: int = 50, n_samples: int = 4,
                                n_warmup: int = 3, n_measure: int = 5):
    """Benchmark trace_model_ + batching vs sequential uncompiled (model forward only)."""
    p(f"  Benchmarking trace_model_ + batching (seq_len={seq_len}, n_samples={n_samples})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dgrams = []; rtypes = []; masks = []
    for _ in range(n_samples):
        d, r, m = make_dummy_inputs(seq_len, device=device)
        dgrams.append(d); rtypes.append(r); masks.append(m)

    # ---- Baseline: sequential uncompiled ----
    infer_base = OpenFoldTemplateInference(
        model_name=model_name, jax_params_path=jax_params, device=device,
        max_recycling_iters=max_recycling_iters, use_mlm=False,
    )
    single_batches = [infer_base.build_batch(d, r, m) for d, r, m in zip(dgrams, rtypes, masks)]

    for _ in range(n_warmup):
        for b in single_batches:
            with torch.no_grad():
                infer_base.model(copy.deepcopy(b))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        for b in single_batches:
            with torch.no_grad():
                infer_base.model(copy.deepcopy(b))
        _sync()
    baseline = (time.time() - t0) / n_measure
    p(f"    Baseline (seq uncompiled, {n_samples}x): {baseline:.3f}s")

    # ---- Traced + batched ----
    infer_traced = OpenFoldTemplateInference(
        model_name=model_name, jax_params_path=jax_params, device=device,
        max_recycling_iters=max_recycling_iters, use_mlm=False,
    )
    multi_batch = infer_traced.build_batch_multi(dgrams, rtypes, masks, _pad_to_length=seq_len)
    # Use first sample for tracing (trace_model_ expects no batch dim)
    single_for_trace = {k: v[0] for k, v in multi_batch.items()}

    p("    Tracing model (one-time cost)...")
    t_trace = time.time()
    infer_traced.enable_tracing(single_for_trace)   # explicit trace
    p(f"    Tracing overhead: {time.time() - t_trace:.2f}s")

    for _ in range(n_warmup):
        with torch.no_grad():
            infer_traced.model(copy.deepcopy(multi_batch))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer_traced.model(copy.deepcopy(multi_batch))
        _sync()
    combined = (time.time() - t0) / n_measure

    total_speedup = baseline / combined if combined > 0 else float("inf")
    p(f"    trace_model_+batched ({n_samples}x): {combined:.3f}s")
    p(f"    Total speedup: {total_speedup:.2f}x  ({baseline/n_samples:.3f}s/sample seq vs {combined/n_samples:.3f}s/sample)")

    del infer_base, infer_traced
    return baseline, combined


def benchmark_combined(infer: OpenFoldTemplateInference, seq_len: int = 50,
                       n_samples: int = 4, n_warmup: int = 3, n_measure: int = 5):
    """Benchmark torch.compile + batching combined vs sequential uncompiled (model forward only).

    IMPORTANT: compile with dynamic=True so the same compiled kernels work for
    both single-sample (rank-3 tensors) and batched (rank-4 tensors) inputs.
    Warm up the compiled model exclusively on batched inputs to ensure the
    rank-4 specialization is cached before measuring.
    """
    p(f"  Benchmarking torch.compile + batching (seq_len={seq_len}, n_samples={n_samples})...")

    dgrams, rtypes, masks = [], [], []
    for _ in range(n_samples):
        d, r, m = make_dummy_inputs(seq_len, device=infer.device)
        dgrams.append(d)
        rtypes.append(r)
        masks.append(m)

    single_batches = [infer.build_batch(d, r, m) for d, r, m in zip(dgrams, rtypes, masks)]
    multi_batch = infer.build_batch_multi(dgrams, rtypes, masks)

    # --- Baseline: sequential uncompiled ---
    infer.disable_compilation()
    for _ in range(n_warmup):
        for b in single_batches:
            with torch.no_grad():
                infer.model(copy.deepcopy(b))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        for b in single_batches:
            with torch.no_grad():
                infer.model(copy.deepcopy(b))
        _sync()
    baseline = (time.time() - t0) / n_measure
    p(f"    Baseline (seq uncompiled, {n_samples}x): {baseline:.3f}s")

    # --- Combined: compiled + batched ---
    # Compile with dynamic=True so kernels handle the rank-4 batched tensors.
    infer.enable_compilation(dynamic=True)
    p("    Warming up compiled model on batched inputs...")
    for _ in range(n_warmup + 2):
        with torch.no_grad():
            infer.model(copy.deepcopy(multi_batch))
    _sync()

    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer.model(copy.deepcopy(multi_batch))
        _sync()
    combined = (time.time() - t0) / n_measure

    total_speedup = baseline / combined if combined > 0 else float("inf")
    p(f"    torch.compile+batched ({n_samples}x): {combined:.3f}s")
    p(f"    Total speedup: {total_speedup:.2f}x  ({baseline/n_samples:.3f}s/sample seq vs {combined/n_samples:.3f}s/sample)")

    infer.disable_compilation()
    return baseline, combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--model_name", type=str, default="model_1_ptm")
    parser.add_argument("--jax_params", type=str, default=os.path.expanduser("~/openfold/openfold/resources/params/params_model_1_ptm.npz"))
    parser.add_argument("--max_recycling_iters", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"Device: {device}")
    p(f"Loading model: {args.model_name}...")

    infer = OpenFoldTemplateInference(
        model_name=args.model_name,
        jax_params_path=args.jax_params,
        device=device,
        max_recycling_iters=args.max_recycling_iters,
        use_mlm=False,
    )
    p("Model loaded.")

    # ----------------------------------------------------------------
    # Correctness tests
    # ----------------------------------------------------------------
    p("\n=== Backward Compatibility ===")
    test_backward_compat(infer)

    p("\n=== Determinism (use_mlm=False) ===")
    test_determinism(infer)

    p("\n=== Padding ===")
    test_build_batch_padding(infer)

    if not args.skip_batch:
        p("\n=== Batching (same length) ===")
        test_batching_same_length(infer)

        p("\n=== Batching (variable length) ===")
        test_batching_variable_length(infer)

    if not args.skip_compile:
        p("\n=== Compilation Toggle (torch.compile) ===")
        test_compilation_toggle(infer)

    if not args.skip_trace:
        p("\n=== Tracing (trace_model_) ===")
        test_tracing(args.model_name, args.jax_params, args.max_recycling_iters)

    # ----------------------------------------------------------------
    # Benchmarks
    # ----------------------------------------------------------------
    p("\n" + "=" * 60)
    p("BENCHMARKS  (max_recycling_iters=" + str(args.max_recycling_iters) + ")")
    p("=" * 60)

    if not args.skip_compile:
        for sl in [50, 100, 200]:
            p(f"\n--- torch.compile speedup (seq_len={sl}) ---")
            benchmark_compilation(infer, seq_len=sl)

    if not args.skip_trace:
        for sl in [50, 100, 200]:
            p(f"\n--- trace_model_ speedup (seq_len={sl}) ---")
            benchmark_tracing(args.model_name, args.jax_params, args.max_recycling_iters,
                              seq_len=sl)

    if not args.skip_batch:
        for n in [2, 4, 8]:
            p(f"\n--- Batching speedup (n_samples={n}, seq_len=50) ---")
            benchmark_batching(infer, seq_len=50, n_samples=n)

    if not args.skip_compile and not args.skip_batch:
        for sl, n in [(50, 4), (50, 8), (100, 4)]:
            p(f"\n--- torch.compile+batch (seq_len={sl}, n_samples={n}) ---")
            benchmark_combined(infer, seq_len=sl, n_samples=n)

    if not args.skip_trace and not args.skip_batch:
        for sl, n in [(50, 4), (50, 8), (100, 4)]:
            p(f"\n--- trace_model_+batch (seq_len={sl}, n_samples={n}) ---")
            benchmark_trace_plus_batch(args.model_name, args.jax_params,
                                       args.max_recycling_iters, seq_len=sl, n_samples=n)

    p("\n=== ALL TESTS AND BENCHMARKS COMPLETE ===")


if __name__ == "__main__":
    main()
