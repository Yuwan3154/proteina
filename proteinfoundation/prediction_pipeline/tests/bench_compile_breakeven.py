"""Measure eager / warmup / steady-state for the compiled inference path
at two representative lengths and compute the per-length break-even.

Two configs:
  (a) compile + SDPA, cueq mul OFF (the recommended default — keeps Triton
      autotune off so warmup is dominated by Dynamo+Inductor not autotune)
  (b) eager baseline (same forward_inference path, no torch.compile)

For each L in {85, 200}:
  - 3 eager forwards (median reported)
  - 1 compile warmup (timed once)
  - 3 steady-state compiled forwards (median reported)

Then prints the break-even N = (warmup - steady) / (eager - steady).
"""
from __future__ import annotations

import os
import sys
import time
from statistics import median

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


def _make_inputs(seq_len: int, device, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    residue_type = torch.tensor(
        rng.randint(0, 20, size=(1, seq_len)), dtype=torch.long, device=device,
    )
    mask = torch.ones((1, seq_len), dtype=torch.float32, device=device)
    raw = rng.gamma(1.0, 1.0, size=(1, seq_len, seq_len, 39)).astype(np.float32)
    dgram = raw / raw.sum(axis=-1, keepdims=True)
    distogram = torch.tensor(dgram, dtype=torch.float32, device=device)
    return {"residue_type": residue_type, "mask": mask, "distogram_probs": distogram}


def _build_batch(w: OpenFoldTemplateInference, inp: dict) -> dict:
    return w.build_batch(
        distogram_probs=inp["distogram_probs"],
        residue_type=inp["residue_type"],
        mask=inp["mask"],
        template_mode="distogram_only",
        seed=0,
    )


def _time_forward(w: OpenFoldTemplateInference, batch: dict) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = w.model.forward_inference(batch)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure(L: int, eager_w: OpenFoldTemplateInference,
            compiled_w: OpenFoldTemplateInference, device) -> dict:
    inp = _make_inputs(L, device)
    eager_batch = _build_batch(eager_w, inp)
    compiled_batch = _build_batch(compiled_w, inp)

    print(f"\n[L={L}] eager x3 ...")
    eager_times = []
    for _ in range(3):
        eager_times.append(_time_forward(eager_w, eager_batch))
        print(f"    {eager_times[-1]:.3f}s")
    eager = median(eager_times)

    print(f"[L={L}] compiled warmup (1st call) ...")
    warmup = _time_forward(compiled_w, compiled_batch)
    print(f"    {warmup:.3f}s")

    print(f"[L={L}] compiled steady-state x3 ...")
    steady_times = []
    for _ in range(3):
        steady_times.append(_time_forward(compiled_w, compiled_batch))
        print(f"    {steady_times[-1]:.3f}s")
    steady = median(steady_times)

    if eager > steady:
        breakeven = (warmup - steady) / (eager - steady)
    else:
        breakeven = float("inf")

    return dict(L=L, eager=eager, warmup=warmup, steady=steady, breakeven=breakeven)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    device = torch.device("cuda")

    print("Building eager wrapper ...")
    eager_w = OpenFoldTemplateInference(
        model_name="model_1_ptm",
        max_recycling_iters=1,
        use_deepspeed_evoformer_attention=False,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
        compile_inference_path=False,
    )

    print("Building compiled wrapper (per_block, SDPA, cueq mul OFF) ...")
    compiled_w = OpenFoldTemplateInference(
        model_name="model_1_ptm",
        max_recycling_iters=1,
        use_deepspeed_evoformer_attention=False,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
        compile_inference_path=True,
        inference_attn_kernel="sdpa",
        compile_strategy="per_block",
        use_cueq_triangle_mul=False,
    )

    results = []
    for L in (85, 200):
        try:
            r = measure(L, eager_w, compiled_w, device)
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[L={L}] OOM: {e}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[L={L}] error: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    print(f"{'L':>4} {'eager':>8} {'warmup':>8} {'steady':>8} {'speedup':>8} {'breakeven N':>12}")
    for r in results:
        sp = (r["eager"] - r["steady"]) / r["eager"] * 100.0
        print(f"{r['L']:>4} {r['eager']:>8.2f} {r['warmup']:>8.2f} {r['steady']:>8.2f} "
              f"{sp:>7.1f}% {r['breakeven']:>12.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
