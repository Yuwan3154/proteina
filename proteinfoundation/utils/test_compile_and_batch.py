#!/usr/bin/env python
"""Tests and benchmarks for OpenFoldTemplateInference compilation and batching.

Usage:
    python -u test_compile_and_batch.py [--skip-compile] [--skip-batch]
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


def _make_synthetic_allatom_pdb(seq_len: int) -> str:
    """Write a minimal all-atom backbone PDB (N, CA, C, O) and return its path.

    Used to test the full-template featurization path (which requires a real
    3D structure, not just a sequence stub).  Coordinates are placed on a
    straight helix-like backbone to avoid degenerate geometry.
    """
    import tempfile, math
    lines = ["MODEL     1"]
    atom_idx = 1
    residues = ["ALA"] * seq_len
    for res_idx, resname in enumerate(residues):
        # Simple helical coordinates
        angle = res_idx * (100.0 * math.pi / 180.0)
        ca_x = 2.3 * math.cos(angle)
        ca_y = 2.3 * math.sin(angle)
        ca_z = res_idx * 1.5
        # Place N, CA, C, O at approximate backbone positions
        for atom_name, dx, dy, dz in [(" N  ", -1.2, 0.0, -0.5),
                                       (" CA ", 0.0, 0.0, 0.0),
                                       (" C  ", 1.1, 0.6, 0.0),
                                       (" O  ", 1.4, 1.7, 0.0)]:
            # 0-indexed residue numbers so that convert_pdb_to_cif's +1 offset
            # produces 1..seq_len which fits within the CIF entity definition.
            lines.append(
                f"ATOM  {atom_idx:5d} {atom_name} {resname} A{res_idx:4d}    "
                f"{ca_x+dx:8.3f}{ca_y+dy:8.3f}{ca_z+dz:8.3f}  1.00  0.00           C"
            )
            atom_idx += 1
    lines.append("ENDMDL")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write("\n".join(lines) + "\n")
    tmp.close()
    return tmp.name


def benchmark_featurization(infer: OpenFoldTemplateInference, seq_len: int = 50,
                            n_warmup: int = 1, n_measure: int = 3):
    """Benchmark featurization (build_batch) vs model forward to identify bottleneck.

    Tests BOTH paths used in production:
      A) distogram_only: stub template (no mmcif, no kalign) — used in Proteina diffusion
      B) full_template: real CIF + kalign alignment — used in AF2Rank scoring

    Model forward is benchmarked once (same in both paths).
    """
    import tempfile, os
    sys.path.insert(0, "/home/ubuntu/proteina")
    sys.path.insert(0, "/home/ubuntu/openfold")

    p(f"  Benchmarking featurization vs model forward (seq_len={seq_len})...")
    dgram, rtype, mask = make_dummy_inputs(seq_len, device=infer.device)

    # ---- Path A: distogram_only (stub template, no mmcif) ----
    for _ in range(n_warmup):
        infer.build_batch(dgram, rtype, mask, template_mode="distogram_only")
    t0 = time.time()
    for _ in range(n_measure):
        infer.build_batch(dgram, rtype, mask, template_mode="distogram_only")
    feat_distogram = (time.time() - t0) / n_measure
    p(f"    build_batch distogram_only:    {feat_distogram:.3f}s/iter")

    # ---- Path B: full_template (real CIF, skip_template_alignment=True) ----
    # Production AF2Rank uses UNK query (residue index 20) and skips kalign.
    # infer.skip_template_alignment must be True (set at construction in main()).
    rtype_unk = torch.full((1, seq_len), 20, dtype=torch.long, device=infer.device)  # UNK=20
    if not getattr(infer, "skip_template_alignment", False):
        p("    WARNING: infer.skip_template_alignment is False; full_template path will call kalign")
    try:
        from proteinfoundation.af2rank_evaluation.af2rank_openfold_scorer import convert_pdb_to_cif
        pdb_path = _make_synthetic_allatom_pdb(seq_len)
        cif_path = convert_pdb_to_cif(pdb_path, chain_id="A")

        full_kwargs = dict(
            template_mode="full_template",
            template_mmcif_path=cif_path,
            template_chain_id="A",
            kalign_binary_path="/usr/bin/kalign",
            mask_template_aatype=True,   # mask out template sequence (AF2Rank protocol)
            zero_template_unit_vector=True,
            zero_template_torsion_angles=True,
        )

        # Smoke-test: verify features look correct on first call
        feat_check = infer.build_batch(None, rtype_unk, mask, **full_kwargs)
        # template_aatype should be all-UNK (index 20) when mask_template_aatype=True
        tpl_aa = feat_check.get("template_aatype")
        if tpl_aa is not None:
            n_unk = int((tpl_aa == 20).sum().item())
            n_total = int(tpl_aa.numel())
            p(f"    [check] template_aatype UNK(20) entries: {n_unk}/{n_total} (expect all-UNK with mask_template_aatype=True)")
        # template_all_atom_positions should be non-zero (real backbone coords)
        tpl_pos = feat_check.get("template_all_atom_positions")
        if tpl_pos is not None:
            pos_max = float(tpl_pos.abs().max().item())
            p(f"    [check] template_all_atom_positions max abs: {pos_max:.2f} Angstrom (expect >0 for real coords)")

        for _ in range(n_warmup):
            infer.build_batch(None, rtype_unk, mask, **full_kwargs)
        t0 = time.time()
        for _ in range(n_measure):
            infer.build_batch(None, rtype_unk, mask, **full_kwargs)
        feat_full = (time.time() - t0) / n_measure
        p(f"    build_batch full_template:     {feat_full:.3f}s/iter  (skip_alignment=True, no kalign)")
        os.unlink(pdb_path)
        os.unlink(cif_path)
    except Exception as e:
        feat_full = None
        p(f"    build_batch full_template:     FAILED: {e}")

    # ---- Model forward (GPU) ----
    batch = infer.build_batch(dgram, rtype, mask, template_mode="distogram_only")
    for _ in range(n_warmup):
        with torch.no_grad():
            infer.model(copy.deepcopy(batch))
    _sync()
    t0 = time.time()
    for _ in range(n_measure):
        with torch.no_grad():
            infer.model(copy.deepcopy(batch))
        _sync()
    model_time = (time.time() - t0) / n_measure
    p(f"    model forward (GPU):           {model_time:.3f}s/iter")

    if feat_full is not None:
        total_full = feat_full + model_time
        p(f"    Total (full_template path):    {total_full:.3f}s/iter")
        p(f"      featurization fraction:      {feat_full/total_full*100:.1f}%")
        p(f"      model fraction:              {model_time/total_full*100:.1f}%")
    return feat_distogram, feat_full, model_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--skip-feat", action="store_true")
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
        # deepspeed attention is now the default (1.4-1.6x speedup).
        # It is mutually exclusive with compile_model.
        use_deepspeed_evoformer_attention=True,
        # Production AF2Rank scorer always skips kalign alignment.
        skip_template_alignment=True,
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

    # ----------------------------------------------------------------
    # Benchmarks
    # ----------------------------------------------------------------
    p("\n" + "=" * 60)
    p("BENCHMARKS  (max_recycling_iters=" + str(args.max_recycling_iters) + ")")
    p("=" * 60)

    if not args.skip_feat:
        p("\n=== Featurization vs Model Forward (bottleneck analysis) ===")
        for sl in [50, 100, 200]:
            p(f"\n--- Featurization breakdown (seq_len={sl}) ---")
            benchmark_featurization(infer, seq_len=sl)

    if not args.skip_compile:
        for sl in [50, 100, 200]:
            p(f"\n--- torch.compile speedup (seq_len={sl}) ---")
            benchmark_compilation(infer, seq_len=sl)

    if not args.skip_batch:
        for n in [2, 4, 8]:
            p(f"\n--- Batching speedup (n_samples={n}, seq_len=50) ---")
            benchmark_batching(infer, seq_len=50, n_samples=n)

    if not args.skip_compile and not args.skip_batch:
        for sl, n in [(50, 4), (50, 8), (100, 4)]:
            p(f"\n--- torch.compile+batch (seq_len={sl}, n_samples={n}) ---")
            benchmark_combined(infer, seq_len=sl, n_samples=n)

    p("\n=== ALL TESTS AND BENCHMARKS COMPLETE ===")


if __name__ == "__main__":
    main()
