"""Isolated IPA + StructureModule torch.compile stress-test.

Constructs a StructureModule with production-realistic dims (matches the
20M Stage-1 config: c_s=384, c_z=128, no_blocks=8) and runs forward across
a swept grid of (batch_size, seq_len). Counts Dynamo recompilations and
reports — used to:

1. Reproduce whatever recompile-limit issue motivated the
   ``@torch.compiler.disable`` decorators on ``Rotation.__init__`` and
   ``Rigid.__init__`` (commit de0d93f).
2. Verify the proposed fix (remove decorators + simplify ``o_pt`` path +
   eager buffer init) doesn't regress that behavior.

Run as: CUTLASS_PATH=... python scripts/test_ipa_compile.py [--bench]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import torch

# Use absolute imports to bypass the proteinfoundation package's `__init__` side effects.
sys.path.insert(0, "/home/ubuntu/proteina")
from proteinfoundation.openfold_stub.model.structure_module import StructureModule


# Production-realistic IPA / StructureModule config (matches what
# protein_transformer.py builds for the 20M Stage-1 model).
SM_CFG = dict(
    c_s=384,
    c_z=128,
    c_ipa=16,
    c_resnet=128,
    no_heads_ipa=12,
    no_qk_points=4,
    no_v_points=8,
    dropout_rate=0.0,  # disable dropout for deterministic compile testing
    no_blocks=8,
    no_transition_layers=1,
    no_resnet_blocks=2,
    no_angles=7,
    trans_scale_factor=10,
    epsilon=1e-12,
    inf=1e5,
)


def make_module(device: torch.device, dtype: torch.dtype = torch.float32) -> StructureModule:
    sm = StructureModule(**SM_CFG).to(device=device, dtype=dtype)
    sm.eval()  # use eval to skip dropout & save one compile artifact
    return sm


def make_inputs(B: int, N: int, c_s: int, c_z: int, device: torch.device,
                dtype: torch.dtype = torch.float32,
                valid_lens: Optional[list] = None):
    """Build a single/pair representation + aatype + mask matching StructureModule's contract.

    `valid_lens`: per-sample number of valid positions (≤ N). If provided, mask has
    `valid_lens[i]` True entries followed by False (padding). The TENSOR SHAPE
    is still [B, N] regardless — only the mask values vary. This mimics how
    PaddingTransform pads short chains in a real training batch.
    """
    single = torch.randn(B, N, c_s, device=device, dtype=dtype, requires_grad=True)
    pair = torch.randn(B, N, N, c_z, device=device, dtype=dtype, requires_grad=True)
    aatype = torch.randint(0, 20, (B, N), device=device)
    if valid_lens is None:
        mask = torch.ones(B, N, device=device, dtype=torch.bool)
    else:
        assert len(valid_lens) == B, f"valid_lens must have length B={B}"
        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        for i, L in enumerate(valid_lens):
            mask[i, :int(L)] = True
    return single, pair, aatype, mask


def run_once(sm: StructureModule, single: torch.Tensor, pair: torch.Tensor,
             aatype: torch.Tensor, mask: torch.Tensor):
    """Run StructureModule.forward and return the final-iteration frames tensor."""
    out = sm(
        {"single": single, "pair": pair},
        aatype=aatype,
        mask=mask,
        inplace_safe=False,
        _offload_inference=False,
    )
    # final-iteration frames tensor [B, N, 7] — like the proteina caller uses
    f = out["frames"][-1] if out["frames"].dim() == 4 else out["frames"]
    return f


def _dynamo_counters():
    """Snapshot Dynamo's internal recompile / cache counters for diffing."""
    try:
        import torch._dynamo as dynamo
        # `recompile_reason` and `frames` counters are the most useful.
        counters = dynamo.utils.counters
        return {k: dict(v) for k, v in counters.items()}
    except Exception as e:
        return {"_err": repr(e)}


def _diff_counters(before: dict, after: dict) -> dict:
    """Return only the keys/values that changed between two snapshots."""
    out = {}
    for grp, d in after.items():
        if grp == "_err":
            out[grp] = d
            continue
        prev = before.get(grp, {})
        delta = {}
        for k, v in d.items():
            if v != prev.get(k, 0):
                delta[k] = v - prev.get(k, 0)
        if delta:
            out[grp] = delta
    return out


def sweep(sm: StructureModule,
          shapes: list[tuple[int, int]],
          device: torch.device,
          compiled: bool,
          label: str,
          valid_lens_per_step: Optional[list] = None,
          dynamic: Optional[bool] = None,
          compile_fn: Optional[callable] = None):
    """Run forward over a sequence of (B, N) shapes. Returns timing + counter info.

    Args:
        shapes: list of (B, N) tuples to run, one per call.
        compiled: if True, wrap `run_once` with torch.compile.
        valid_lens_per_step: optional list (same length as shapes); each entry is a list
            of B integers giving the valid length for each sample in that step's mask.
            Used to test variable-valid-length mask behavior with fixed tensor shapes.
        dynamic: if True, pass dynamic=True to torch.compile (force dynamic shapes).
            If False, dynamic=False (force static, recompile per shape).
            If None (default), let Dynamo decide (auto-dynamic).
        compile_fn: optional pre-built compiled function. Lets the caller cache the
            compile across multiple sweeps for fair comparison.
    """
    import torch._dynamo as dynamo

    if compile_fn is not None:
        fn = compile_fn
    elif compiled:
        compile_kwargs = {"mode": "default"}
        if dynamic is not None:
            compile_kwargs["dynamic"] = dynamic
        fn = torch.compile(run_once, **compile_kwargs)
    else:
        fn = run_once

    times = []
    print(f"\n=== {label} ===")
    header = f"{'(B, N)':<12} {'wall_ms':>9} {'recompiles_so_far':>20}"
    if valid_lens_per_step is not None:
        header += "  valid_lens"
    print(header)
    pre = _dynamo_counters()
    for i, (B, N) in enumerate(shapes):
        v_lens = valid_lens_per_step[i] if valid_lens_per_step is not None else None
        single, pair, aatype, mask = make_inputs(
            B, N, SM_CFG["c_s"], SM_CFG["c_z"], device, valid_lens=v_lens
        )
        # warmup the operation under test (first call always compiles)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = fn(sm, single, pair, aatype, mask)
            # backward to exercise the full compile graph (autograd path)
            loss = out.float().pow(2).sum()
            loss.backward()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
        except Exception as e:
            elapsed = float("nan")
            print(f"  ({B},{N}) ERRORED: {type(e).__name__}: {e}")
            continue
        times.append((B, N, elapsed))
        # Snapshot counters after this call so we can see if it caused a fresh compile.
        post = _dynamo_counters()
        delta = _diff_counters(pre, post)
        recompile_count = sum(
            v for k, v in delta.get("stats", {}).items() if "recompil" in k.lower()
        )
        # Also report "frames" counter — counts the number of compiled frames.
        frames = delta.get("frames", {})
        suffix = ""
        if v_lens is not None:
            suffix = f"  valid_lens={v_lens}"
        print(f"  ({B}, {N}):  {elapsed:>7.1f} ms     +recompiles={recompile_count} frames_delta={frames}{suffix}")
        pre = post
    return times, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run with torch.compile (default: eager only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-size-limit", type=int, default=8,
                        help="Dynamo cache_size_limit (default 8 = production default)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Show the dynamo cache limit so the test is self-documenting.
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = args.cache_size_limit
    dynamo.config.recompile_limit = args.cache_size_limit
    print(f"torch={torch.__version__} | dynamo cache_size_limit={dynamo.config.cache_size_limit}")

    # Build StructureModule once
    print("Building StructureModule (production-realistic dims)...")
    sm = make_module(device)

    # Eager warmup + correctness sanity
    print("Eager warmup (no compile)...")
    sweep(sm, [(2, 32), (2, 32)], device, compiled=False, label="EAGER warmup")

    if not args.bench:
        print("\nDone (eager only). Pass --bench to test under torch.compile.")
        return

    # =================================================================
    # SCENARIO A: Production training pattern — fixed (B, N), varying
    # valid lengths in mask. Padding pads chains to a constant max_size,
    # but each chain has its own valid length. Tensor SHAPE is constant;
    # only the mask VALUES differ. Dynamo specializes on shape, not value
    # → expected: ZERO recompiles after the first call.
    # =================================================================
    import random
    rng = random.Random(0)
    fixed_B, fixed_N = 8, 64
    n_calls_A = 12
    shapes_A = [(fixed_B, fixed_N)] * n_calls_A
    valid_lens_A = [
        [rng.randint(20, fixed_N) for _ in range(fixed_B)]
        for _ in range(n_calls_A)
    ]
    sweep(sm, shapes_A, device, compiled=True,
          label="A: Fixed (B=8, N=64), 12 calls, valid-lens vary per sample",
          valid_lens_per_step=valid_lens_A)

    # =================================================================
    # SCENARIO B (drop_last=False / overfit-mode last batch): fixed N=64,
    # varying B from 1..fixed_B. Mimics overfit mode where the last batch
    # of each epoch can be smaller.
    # =================================================================
    shapes_B = [(B, fixed_N) for B in range(1, fixed_B + 1)]
    sweep(sm, shapes_B, device, compiled=True,
          label=f"B: drop_last=False — varying B in 1..{fixed_B} at N={fixed_N}")

    # =================================================================
    # SCENARIO C: STATIC compile (dynamic=False). One specialized graph
    # per shape, no dynamic dim. Faster kernel per call but a fresh
    # compile per new shape. Use a separate StructureModule instance so
    # the Dynamo cache is fresh and the compile-counter delta is clean.
    # =================================================================
    print("\n--- Building fresh StructureModule for static-vs-dynamic comparison ---")
    torch.manual_seed(args.seed)
    sm_static = make_module(device)
    sm_dyn = make_module(device)

    # Compare wall time on a single (B=8, N=64) shape called 10 times.
    same_shapes = [(8, 64)] * 10

    print("\n--- STATIC compile (dynamic=False) on (8,64) ×10 ---")
    times_static, _ = sweep(
        sm_static, same_shapes, device, compiled=True,
        label="C-static: dynamic=False, single shape",
        dynamic=False,
    )
    print("\n--- DEFAULT compile (auto-dynamic) on (8,64) ×10 ---")
    times_def, _ = sweep(
        sm_dyn, same_shapes, device, compiled=True,
        label="C-default: dynamic=None (auto), single shape",
        dynamic=None,
    )

    # Steady-state median (skip the first 2 calls = compile warmup)
    def _median_ss(times):
        ss = sorted(t for _, _, t in times[2:])
        return ss[len(ss) // 2] if ss else float("nan")
    ms_static = _median_ss(times_static)
    ms_def = _median_ss(times_def)
    print(f"\n  STATIC median steady-state: {ms_static:7.1f} ms")
    print(f"  AUTO   median steady-state: {ms_def:7.1f} ms")
    if ms_static > 0:
        ratio = ms_def / ms_static
        print(f"  AUTO is {ratio:.2f}× the time of STATIC (>1 = static wins)")

    # Now also test STATIC behavior when shape varies — every new shape forces
    # a fresh compile.
    print("\n--- STATIC compile under SHAPE CHANGES (forces fresh compile per shape) ---")
    sm_static2 = make_module(device)
    static_varying_shapes = [(8, 64), (8, 64), (4, 64), (4, 64), (8, 64), (4, 64)]
    sweep(sm_static2, static_varying_shapes, device, compiled=True,
          label="C-static varying shapes (each new shape = fresh compile)",
          dynamic=False)

    # Final dynamo state dump
    print("\n=== Final Dynamo counters ===")
    counters = _dynamo_counters()
    for grp, d in counters.items():
        if grp == "stats" or "frame" in grp.lower():
            print(f"  [{grp}] {d}")


if __name__ == "__main__":
    main()
