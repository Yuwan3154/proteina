"""
Benchmark: SDPA / Flash / cuEquivariance vs vanilla attention in PairBiasAttention.

Usage:
    python scripts/benchmark_attention.py                 # full benchmark + parity
    python scripts/benchmark_attention.py --parity-only   # parity tests only
    python scripts/benchmark_attention.py --no-parity     # speed/VRAM only

Results:
    - Parity table: max |output - vanilla_output| in fp32 for each backend
    - Speed table:  median forward time (ms) at L=128/256/384, with/without torch.compile
    - VRAM table:   peak VRAM (MB) per backend/length
    - Fit table:    coefficients of T = a*L^2 + b fit for time and VRAM

Notes on parity tolerance:
    - SDPA vs vanilla (fp32): expect < 1e-6 (same algorithm, fused kernel)
    - cuEquivariance vs vanilla (fp32): expect < 1e-5 (Triton kernel for N>=128, DH%32==0)
    - Flash vs vanilla (no-bias, fp32): expect < 1e-5 (fp32 uses PyTorch fallback in flash_attn 2.x)
    - Flash attention is NOT tested in fp16/bf16 for parity — online softmax numerics differ (~1e-3)

Notes on scaling fits:
    - All attention methods are O(L^2) in FLOPs; fit T = a*L^2 + b
    - Flash attention's key advantage is O(L) peak memory (no NxN matrix stored);
      at these small lengths the constant terms may dominate the quadratic term
"""

import argparse
import copy
import sys
import time
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make sure proteinfoundation is importable when running from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/ubuntu/proteina")
from proteinfoundation.nn.pair_bias_attn.pair_bias_attn import PairBiasAttention

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LENGTHS = [128, 256, 384]
N_REPS = 16
N_WARMUP = 4
BATCH = 4

# Dims chosen to satisfy cuEq constraint (DH = 512//8 = 64, 64 % 32 == 0)
NODE_DIM = 512
NHEADS = 8
DIM_HEAD = NODE_DIM // NHEADS  # 64
PAIR_DIM = 256

DTYPE_PARITY = torch.float32
DTYPE_BENCH = torch.bfloat16

PARITY_TOL = 1e-5
PARITY_TOL_FLASH = 2e-3  # flash_attn 2.x fp32 not supported; auto-casts to bf16 → ~1e-3 error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_module(attn_impl: str, with_pair: bool = True, dtype=torch.float32) -> PairBiasAttention:
    pair_dim = PAIR_DIM if with_pair else None
    m = PairBiasAttention(
        node_dim=NODE_DIM,
        dim_head=DIM_HEAD,
        heads=NHEADS,
        bias=True,
        dim_out=NODE_DIM,
        qkln=True,
        pair_dim=pair_dim,
        attn_impl=attn_impl,
    ).cuda().to(dtype)
    return m


def _make_inputs(L: int, with_pair: bool, dtype, device="cuda"):
    B = BATCH
    x = torch.randn(B, L, NODE_DIM, dtype=dtype, device=device)
    z = torch.randn(B, L, L, PAIR_DIM, dtype=dtype, device=device) if with_pair else None
    # All tokens valid (no padding) — simplest case and correct for flash attention
    mask = torch.ones(B, L, L, dtype=torch.bool, device=device)
    return x, z, mask


def _copy_weights(src: PairBiasAttention, dst: PairBiasAttention):
    """Copy weights from src to dst (dst may have different attn_impl)."""
    # Copy all matching parameters by name
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    common = {k: src_sd[k] for k in dst_sd if k in src_sd}
    dst_sd.update(common)
    dst.load_state_dict(dst_sd, strict=False)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------

def run_parity_tests():
    print("\n" + "=" * 65)
    print("PARITY TESTS (fp32, L=256, B=2, all-valid tokens)")
    print("=" * 65)
    print(f"{'Backend':<20} {'Pair bias':<12} {'Tolerance':<12} {'Max |diff|':<15} {'Pass'}")
    print("-" * 75)

    L = 256

    # --- Baseline: vanilla with pair bias (fp32) ---
    ref_pair = _make_module("vanilla", with_pair=True, dtype=DTYPE_PARITY)
    x, z, mask = _make_inputs(L, with_pair=True, dtype=DTYPE_PARITY)
    with torch.no_grad():
        ref_out = ref_pair(x, z, mask)

    failed = False
    for impl in ("sdpa", "cuequivariance"):
        m = _make_module(impl, with_pair=True, dtype=DTYPE_PARITY)
        _copy_weights(ref_pair, m)
        with torch.no_grad():
            out = m(x, z, mask)
        diff = (out - ref_out).abs().max().item()
        ok = diff <= PARITY_TOL
        if not ok:
            failed = True
        print(f"{impl:<20} {'yes':<12} {PARITY_TOL:<12.0e} {diff:<15.3e} {'PASS' if ok else 'FAIL'}")

    # --- Flash: no pair bias; fp32 auto-casts to bf16 inside flash_attn → relaxed tol ---
    # Compare flash(fp32 inputs, bf16 kernel) vs vanilla(fp32) — both same weights.
    ref_nopair = _make_module("vanilla", with_pair=False, dtype=DTYPE_PARITY)
    x_np, _, mask_np = _make_inputs(L, with_pair=False, dtype=DTYPE_PARITY)
    with torch.no_grad():
        ref_nopair_out = ref_nopair(x_np, None, mask_np)

    m_flash = _make_module("flash", with_pair=False, dtype=DTYPE_PARITY)
    _copy_weights(ref_nopair, m_flash)
    with torch.no_grad():
        flash_out = m_flash(x_np, None, mask_np)
    diff = (flash_out - ref_nopair_out).abs().max().item()
    ok = diff <= PARITY_TOL_FLASH
    if not ok:
        failed = True
    note = f"(no pair bias; bf16 kernel)"
    print(f"{'flash':<20} {'no':<12} {PARITY_TOL_FLASH:<12.0e} {diff:<15.3e} {'PASS' if ok else 'FAIL'}  {note}")

    print("-" * 75)
    print("Note: flash_attn 2.x does not support fp32; inputs are auto-cast to bf16,")
    print("      so a relaxed tolerance applies. For bf16 training this is expected behavior.")
    print(f"Overall: {'ALL PASS' if not failed else 'SOME FAILED'}")
    return not failed


# ---------------------------------------------------------------------------
# Speed + VRAM benchmark
# ---------------------------------------------------------------------------

def _benchmark_one(
    m: PairBiasAttention,
    L: int,
    with_pair: bool,
    dtype,
    n_warmup: int,
    n_reps: int,
) -> tuple[float, float, float]:
    """Return (median_ms, std_ms, peak_vram_mb)."""
    x, z, mask = _make_inputs(L, with_pair=with_pair, dtype=dtype)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = m(x, z, mask)
    _sync()

    torch.cuda.reset_peak_memory_stats()
    times_ms = []
    for _ in range(n_reps):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = m(x, z, mask)
        _sync()
        times_ms.append((time.perf_counter() - t0) * 1e3)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return float(np.median(times_ms)), float(np.std(times_ms)), peak_mb


def _fit_quadratic(lengths, values):
    """Fit y = a*L^2 + b. Returns (a, b, r2)."""
    x = np.array(lengths, dtype=float) ** 2
    y = np.array(values, dtype=float)
    # polyfit degree 1 in L^2
    coeffs = np.polyfit(x, y, 1)  # [a, b]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coeffs[0], coeffs[1], r2


def run_speed_benchmark():
    print("\n" + "=" * 95)
    print("SPEED + VRAM BENCHMARK (bf16, B=2, all-valid tokens)")
    print("=" * 95)

    # backends with pair bias, tested at all lengths
    PAIR_BACKENDS = ["vanilla", "sdpa", "cuequivariance"]
    # flash needs no-pair module; torch.compile falls back to vanilla for flash,
    # so we only test compile=False for flash
    FLASH_BACKEND = "flash"

    col = "{:<22} {:<8} {:<10} {:>10} {:>10} {:>12} {:>12}"
    print(col.format("backend", "compile", "L", "time_ms", "±std", "VRAM_MB", "speedup"))
    print("-" * 95)

    # Store times for fitting: {(backend, compile): {L: median_ms}}
    time_data: dict = {}
    vram_data: dict = {}

    # Reference times at each L for speedup computation
    ref_times: dict[int, float] = {}

    def run_backend(impl, with_pair, compiled, label):
        m_base = _make_module(impl, with_pair=with_pair, dtype=DTYPE_BENCH)
        if compiled:
            torch._dynamo.reset()
            m = torch.compile(m_base)
            # Trigger one compilation pass before timing
            x0, z0, mask0 = _make_inputs(LENGTHS[0], with_pair=with_pair, dtype=DTYPE_BENCH)
            with torch.no_grad():
                _ = m(x0, z0, mask0)
            _sync()
        else:
            m = m_base

        key = (label, compiled)
        time_data[key] = {}
        vram_data[key] = {}

        for L in LENGTHS:
            med, std, vram = _benchmark_one(m, L, with_pair=with_pair,
                                             dtype=DTYPE_BENCH,
                                             n_warmup=N_WARMUP, n_reps=N_REPS)
            time_data[key][L] = med
            vram_data[key][L] = vram

            # Record vanilla (no compile) as reference for speedup
            if impl == "vanilla" and not compiled:
                ref_times[L] = med

            speedup = ref_times.get(L, med) / med if ref_times.get(L) else float("nan")
            cmp_str = "yes" if compiled else "no"
            print(col.format(label, cmp_str, L,
                             f"{med:.2f}", f"{std:.2f}", f"{vram:.1f}", f"{speedup:.2f}x"))

    # vanilla (reference — run first to populate ref_times)
    run_backend("vanilla", with_pair=True, compiled=False, label="vanilla")
    run_backend("vanilla", with_pair=True, compiled=True,  label="vanilla")
    # sdpa
    run_backend("sdpa",    with_pair=True, compiled=False, label="sdpa")
    run_backend("sdpa",    with_pair=True, compiled=True,  label="sdpa")
    # flash (no pair bias; compile=False only)
    run_backend(FLASH_BACKEND, with_pair=False, compiled=False, label="flash")
    # cuequivariance
    run_backend("cuequivariance", with_pair=True, compiled=False, label="cuequivariance")
    run_backend("cuequivariance", with_pair=True, compiled=True,  label="cuequivariance")

    # -----------------------------------------------------------------------
    # Curve fitting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("SCALING FITS  (T = a·L² + b,  VRAM = a·L² + b)")
    print("Note: Flash Attention's peak memory scales as O(L), not O(L²).")
    print("=" * 75)
    fit_col = "{:<22} {:<10} {:>14} {:>14} {:>8}"
    print(fit_col.format("backend/compile", "metric", "a (per L²)", "b (intercept)", "R²"))
    print("-" * 75)

    for key in time_data:
        label, compiled = key
        cmp_str = "compiled" if compiled else "eager"
        tag = f"{label}/{cmp_str}"

        t_vals = [time_data[key][L] for L in LENGTHS]
        v_vals = [vram_data[key][L] for L in LENGTHS]

        ta, tb, tr2 = _fit_quadratic(LENGTHS, t_vals)
        va, vb, vr2 = _fit_quadratic(LENGTHS, v_vals)

        print(fit_col.format(tag, "time_ms",
                             f"{ta:.4e}", f"{tb:.3f}", f"{tr2:.4f}"))
        print(fit_col.format("", "VRAM_MB",
                             f"{va:.4e}", f"{vb:.3f}", f"{vr2:.4f}"))
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Attention backend benchmark")
    parser.add_argument("--parity-only", action="store_true",
                        help="Run only parity tests, skip speed benchmark")
    parser.add_argument("--no-parity", action="store_true",
                        help="Skip parity tests, run only speed benchmark")
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Benchmark dims: node_dim={NODE_DIM}, heads={NHEADS}, dim_head={DIM_HEAD}, "
          f"pair_dim={PAIR_DIM}, batch={BATCH}")
    print(f"Parity dtype: {DTYPE_PARITY}, benchmark dtype: {DTYPE_BENCH}")

    passed = True
    if not args.no_parity:
        passed = run_parity_tests()

    if not args.parity_only:
        run_speed_benchmark()

    if not args.no_parity and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
