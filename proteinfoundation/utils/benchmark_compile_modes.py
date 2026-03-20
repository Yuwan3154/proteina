#!/usr/bin/env python
"""
Benchmark torch.compile modes, backends, and cuequivariance for ProteinTransformerAF3.

Measures compilation overhead and forward-pass speedup across configurations:
  - Compile modes: default, reduce-overhead, max-autotune
  - With/without cuequivariance kernels
  - Training (grad-enabled) vs validation (no-grad) compiled artifacts

Usage:
    python benchmark_compile_modes.py [--seq_len 128] [--batch_size 4] [--n_warmup 5] [--n_measure 10]
    python benchmark_compile_modes.py --quick   # fast run with fewer iterations
"""
import argparse
import gc
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

sys.path.insert(0, "/home/ubuntu/proteina")

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3

try:
    import cuequivariance_torch as cuet
    CUEQUIVARIANCE_AVAILABLE = True
except ImportError:
    CUEQUIVARIANCE_AVAILABLE = False


def p(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Model configs (small models with tri_mult enabled, matching dims for cueq)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "ca_80M": {
        "description": "CA-flow 80M (pair_repr_dim=512, tri_mult_c=512, cueq-compatible)",
        "config": {
            "token_dim": 768,
            "nlayers": 8,
            "nheads": 12,
            "residual_mha": True,
            "residual_transition": True,
            "parallel_mha_transition": False,
            "use_attn_pair_bias": True,
            "strict_feats": False,
            "feature_embedding_mode": "individual",
            "feats_init_seq": ["res_seq_pdb_idx", "x_sc", "chain_break_per_res"],
            "feats_cond_seq": ["time_emb"],
            "feats_pair_repr": ["rel_seq_sep", "x_sc_pair_dists", "xt_pair_dists"],
            "feats_pair_cond": ["time_emb"],
            "t_emb_dim": 256,
            "dim_cond": 512,
            "idx_emb_dim": 128,
            "fold_emb_dim": 256,
            "cath_code_dir": "/tmp",
            "multilabel_mode": "sample",
            "xt_pair_dist_dim": 64,
            "xt_pair_dist_min": 0.1,
            "xt_pair_dist_max": 3.0,
            "x_sc_pair_dist_dim": 128,
            "x_sc_pair_dist_min": 0.1,
            "x_sc_pair_dist_max": 3.0,
            "x_motif_pair_dist_dim": 128,
            "x_motif_pair_dist_min": 0.1,
            "x_motif_pair_dist_max": 3.0,
            "seq_sep_dim": 127,
            "pair_repr_dim": 512,
            "update_pair_repr": True,
            "update_pair_repr_every_n": 2,
            "use_tri_mult": True,
            "tri_mult_c": 512,  # == pair_repr_dim -> cueq compatible
            "num_registers": 10,
            "use_qkln": True,
            "num_buckets_predict_pair": 64,
            "predict_coords": True,
            "contact_map_mode": False,
            "use_torch_compile": False,  # set per test
            "use_cueq": False,  # set per test
        },
    },
    "contact_100M": {
        "description": "Contact-map 100M (pair_repr_dim=256, tri_mult_c=256, cueq-compatible)",
        "config": {
            "token_dim": 512,
            "nlayers": 12,
            "nheads": 8,
            "residual_mha": True,
            "residual_transition": True,
            "parallel_mha_transition": False,
            "use_attn_pair_bias": True,
            "strict_feats": False,
            "feats_init_seq": ["res_seq_pdb_idx", "chain_break_per_res"],
            "feats_cond_seq": ["time_emb"],
            "feats_pair_repr": ["rel_seq_sep", "contact_map_sc"],
            "feats_pair_cond": ["time_emb"],
            "t_emb_dim": 256,
            "dim_cond": 512,
            "idx_emb_dim": 128,
            "fold_emb_dim": 256,
            "cath_code_dir": "/tmp",
            "multilabel_mode": "sample",
            "xt_pair_dist_dim": 64,
            "xt_pair_dist_min": 0.1,
            "xt_pair_dist_max": 3.0,
            "x_sc_pair_dist_dim": 128,
            "x_sc_pair_dist_min": 0.1,
            "x_sc_pair_dist_max": 3.0,
            "x_motif_pair_dist_dim": 128,
            "x_motif_pair_dist_min": 0.1,
            "x_motif_pair_dist_max": 3.0,
            "seq_sep_dim": 127,
            "pair_repr_dim": 256,
            "update_pair_repr": True,
            "update_pair_repr_every_n": 2,
            "use_tri_mult": True,
            "tri_mult_c": 256,  # == pair_repr_dim -> cueq compatible
            "num_registers": 10,
            "use_qkln": True,
            "num_buckets_predict_pair": 39,
            "contact_map_mode": True,
            "contact_map_embed_dim": 64,
            "contact_map_input_dim": 1,
            "predict_coords": False,
            "non_contact_value": 0,
            "use_torch_compile": False,
            "use_cueq": False,
        },
    },
}

COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


def make_batch_ca(batch_size: int, seq_len: int, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Create a synthetic batch for CA-flow models."""
    return {
        "x_t": torch.randn(batch_size, seq_len, 3, device=device) * 0.1,
        "t": torch.rand(batch_size, device=device),
        "mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        "x_sc": torch.randn(batch_size, seq_len, 3, device=device) * 0.1,
    }


def make_batch_contact(batch_size: int, seq_len: int, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Create a synthetic batch for contact-map models."""
    contact_map_t = torch.rand(batch_size, seq_len, seq_len, device=device)
    contact_map_t = (contact_map_t + contact_map_t.transpose(-1, -2)) / 2.0
    contact_map_sc = torch.zeros(batch_size, seq_len, seq_len, device=device)
    return {
        "x_t": torch.zeros(batch_size, seq_len, 3, device=device),  # placeholder for feature factory shape inference
        "t": torch.rand(batch_size, device=device),
        "mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        "contact_map_t": contact_map_t,
        "contact_map_sc": contact_map_sc,
    }


@dataclass
class BenchmarkResult:
    model_name: str
    compile_mode: str  # "eager", "default", "reduce-overhead", "max-autotune"
    use_cueq: bool
    train_compile_time_s: float = 0.0
    train_step_ms: float = 0.0
    eval_compile_time_s: float = 0.0
    eval_step_ms: float = 0.0
    train_speedup: float = 1.0
    eval_speedup: float = 1.0


def _timed_forward(model, batch, n_warmup: int, n_measure: int, grad_enabled: bool) -> float:
    """Run forward passes and return average step time in ms."""
    for _ in range(n_warmup):
        if grad_enabled:
            model(batch)
        else:
            with torch.no_grad():
                model(batch)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_measure):
        if grad_enabled:
            model(batch)
        else:
            with torch.no_grad():
                model(batch)
    _sync()
    elapsed = time.perf_counter() - t0
    return (elapsed / n_measure) * 1000  # ms


def _first_forward_time(model, batch, grad_enabled: bool) -> float:
    """Run a single forward pass and return its time in seconds (includes compilation)."""
    _sync()
    t0 = time.perf_counter()
    if grad_enabled:
        model(batch)
    else:
        with torch.no_grad():
            model(batch)
    _sync()
    return time.perf_counter() - t0


def benchmark_model(
    model_name: str,
    config: dict,
    batch_fn,
    batch_size: int,
    seq_len: int,
    n_warmup: int,
    n_measure: int,
    compile_modes: Optional[List[str]] = None,
    device: str = "cuda",
) -> List[BenchmarkResult]:
    """Benchmark a single model config across all compile modes and cueq on/off."""
    results = []
    cueq_variants = [False]
    # Only test cueq if available and config has matching dims
    if CUEQUIVARIANCE_AVAILABLE:
        pair_dim = config.get("pair_repr_dim", 0)
        tri_c = config.get("tri_mult_c", 0)
        if pair_dim == tri_c and pair_dim % 32 == 0 and pair_dim > 0:
            cueq_variants.append(True)

    for use_cueq in cueq_variants:
        cueq_label = "cueq" if use_cueq else "no-cueq"
        p(f"\n{'='*70}")
        p(f"  Model: {model_name}  |  cueq: {use_cueq}")
        p(f"{'='*70}")

        # --- Eager baseline ---
        p(f"  [eager/{cueq_label}] Building model...")
        cfg = {**config, "use_torch_compile": False, "use_cueq": use_cueq}
        model = ProteinTransformerAF3(**cfg).to(device)
        batch = batch_fn(batch_size, seq_len, device)

        nparams = sum(p_.numel() for p_ in model.parameters()) / 1e6
        p(f"  Parameters: {nparams:.1f}M")

        train_ms_eager = _timed_forward(model, batch, n_warmup, n_measure, grad_enabled=True)
        eval_ms_eager = _timed_forward(model, batch, n_warmup, n_measure, grad_enabled=False)
        p(f"  [eager/{cueq_label}] Train step: {train_ms_eager:.1f}ms  |  Eval step: {eval_ms_eager:.1f}ms")

        results.append(BenchmarkResult(
            model_name=model_name,
            compile_mode="eager",
            use_cueq=use_cueq,
            train_step_ms=train_ms_eager,
            eval_step_ms=eval_ms_eager,
            train_speedup=1.0,
            eval_speedup=1.0,
        ))

        del model
        torch.cuda.empty_cache()
        gc.collect()

        # --- Compiled modes ---
        for mode in (compile_modes or COMPILE_MODES):
            p(f"\n  [{mode}/{cueq_label}] Building model and resetting dynamo...")
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()

            cfg_compiled = {**config, "use_torch_compile": True, "use_cueq": use_cueq}
            model = ProteinTransformerAF3(**cfg_compiled).to(device)

            # Override the compile mode by replacing the lazy compile calls
            # We do this by pre-creating the compiled artifacts with the desired mode
            model._forward_compiled_train = torch.compile(
                model._forward_impl, mode=mode
            )
            model._forward_compiled_eval = torch.compile(
                model._forward_impl, mode=mode
            )
            batch = batch_fn(batch_size, seq_len, device)

            # First train forward (includes JIT compilation)
            p(f"  [{mode}/{cueq_label}] First train forward (JIT compile)...")
            train_compile_s = _first_forward_time(model, batch, grad_enabled=True)
            p(f"  [{mode}/{cueq_label}] Train compile time: {train_compile_s:.1f}s")

            # First eval forward (includes JIT compilation for eval artifact)
            p(f"  [{mode}/{cueq_label}] First eval forward (JIT compile)...")
            eval_compile_s = _first_forward_time(model, batch, grad_enabled=False)
            p(f"  [{mode}/{cueq_label}] Eval compile time: {eval_compile_s:.1f}s")

            # Measure steady-state train step
            train_ms = _timed_forward(model, batch, n_warmup, n_measure, grad_enabled=True)

            # Measure steady-state eval step
            eval_ms = _timed_forward(model, batch, n_warmup, n_measure, grad_enabled=False)

            train_speedup = train_ms_eager / train_ms if train_ms > 0 else float("inf")
            eval_speedup = eval_ms_eager / eval_ms if eval_ms > 0 else float("inf")

            p(f"  [{mode}/{cueq_label}] Train step: {train_ms:.1f}ms (speedup: {train_speedup:.2f}x)")
            p(f"  [{mode}/{cueq_label}] Eval step:  {eval_ms:.1f}ms (speedup: {eval_speedup:.2f}x)")

            results.append(BenchmarkResult(
                model_name=model_name,
                compile_mode=mode,
                use_cueq=use_cueq,
                train_compile_time_s=train_compile_s,
                train_step_ms=train_ms,
                eval_compile_time_s=eval_compile_s,
                eval_step_ms=eval_ms,
                train_speedup=train_speedup,
                eval_speedup=eval_speedup,
            ))

            del model
            torch.cuda.empty_cache()
            gc.collect()

    return results


def print_summary_table(all_results: List[BenchmarkResult]):
    """Print a summary table of all benchmark results."""
    p("\n" + "=" * 110)
    p("BENCHMARK SUMMARY")
    p("=" * 110)
    header = (
        f"{'Model':<16} {'Mode':<18} {'CuEq':<6} "
        f"{'Train(ms)':>10} {'Speedup':>8} "
        f"{'Eval(ms)':>10} {'Speedup':>8} "
        f"{'TrainCompile(s)':>16} {'EvalCompile(s)':>15}"
    )
    p(header)
    p("-" * 110)
    for r in all_results:
        cueq_str = "yes" if r.use_cueq else "no"
        train_compile_str = f"{r.train_compile_time_s:.1f}" if r.train_compile_time_s > 0 else "-"
        eval_compile_str = f"{r.eval_compile_time_s:.1f}" if r.eval_compile_time_s > 0 else "-"
        p(
            f"{r.model_name:<16} {r.compile_mode:<18} {cueq_str:<6} "
            f"{r.train_step_ms:>10.1f} {r.train_speedup:>7.2f}x "
            f"{r.eval_step_ms:>10.1f} {r.eval_speedup:>7.2f}x "
            f"{train_compile_str:>16} {eval_compile_str:>15}"
        )
    p("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.compile modes for ProteinTransformerAF3")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--n_warmup", type=int, default=5, help="Number of warmup iterations (default: 5)")
    parser.add_argument("--n_measure", type=int, default=10, help="Number of measurement iterations (default: 10)")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer iterations")
    parser.add_argument("--models", nargs="+", default=None, choices=list(MODEL_CONFIGS.keys()),
                        help="Models to benchmark (default: all)")
    parser.add_argument("--modes", nargs="+", default=None, choices=COMPILE_MODES,
                        help="Compile modes to test (default: all)")
    args = parser.parse_args()

    if args.quick:
        args.n_warmup = 2
        args.n_measure = 3

    if args.modes:
        modes_to_test = args.modes
    else:
        modes_to_test = list(COMPILE_MODES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p(f"Device: {device}")
    if torch.cuda.is_available():
        p(f"GPU: {torch.cuda.get_device_name(0)}")
    p(f"PyTorch: {torch.__version__}")
    p(f"cuequivariance_torch available: {CUEQUIVARIANCE_AVAILABLE}")
    p(f"Config: seq_len={args.seq_len}, batch_size={args.batch_size}, "
      f"n_warmup={args.n_warmup}, n_measure={args.n_measure}")

    torch.set_float32_matmul_precision("medium")

    models_to_test = args.models or list(MODEL_CONFIGS.keys())
    all_results = []

    for model_name in models_to_test:
        model_info = MODEL_CONFIGS[model_name]
        config = model_info["config"]
        p(f"\n{'#'*70}")
        p(f"# {model_name}: {model_info['description']}")
        p(f"{'#'*70}")

        # Select batch creator based on model type
        if config.get("contact_map_mode", False):
            batch_fn = make_batch_contact
        else:
            batch_fn = make_batch_ca

        results = benchmark_model(
            model_name=model_name,
            config=config,
            batch_fn=batch_fn,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
            compile_modes=modes_to_test,
            device=device,
        )
        all_results.extend(results)

    print_summary_table(all_results)
    p("\nDone.")


if __name__ == "__main__":
    main()
