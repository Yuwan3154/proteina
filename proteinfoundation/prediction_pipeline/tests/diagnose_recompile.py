"""Run with TORCH_LOGS=recompiles to find the specific shape guard that fires
when L changes in the compiled inference path.

Compiles once at L=73, then runs at L=96. The torch._dynamo logger prints
recompile reasons including the specific guard expression that failed.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


def _make_inputs(seq_len: int, device, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    residue_type = torch.tensor(rng.randint(0, 20, size=(1, seq_len)), dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.float32, device=device)
    raw = rng.gamma(1.0, 1.0, size=(1, seq_len, seq_len, 39)).astype(np.float32)
    dgram = raw / raw.sum(axis=-1, keepdims=True)
    distogram = torch.tensor(dgram, dtype=torch.float32, device=device)
    return {"residue_type": residue_type, "mask": mask, "distogram_probs": distogram}


def main():
    if not torch.cuda.is_available():
        return 0
    device = torch.device("cuda")

    # NOTE: compile_strategy="whole_graph" was attempted on 2026-05-22 with
    # the v2.1 fixes in place (mark_dynamic guards, init_triton_cache,
    # single-axis-per-tensor marking). Dynamo trace exceeded 25 min wall
    # clock with no completion at L=73 and was killed manually. Verdict:
    # whole-graph compile is impractical at AlphaFold-2 scale. Use
    # per_block (default) for shape-specialisation diagnostics — the
    # TORCH_LOGS=recompiles output identifies which block's guard fires
    # when L changes.
    print("Loading wrapper ...")
    w = OpenFoldTemplateInference(
        model_name="model_1_ptm",
        max_recycling_iters=1,
        use_deepspeed_evoformer_attention=False,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
        compile_inference_path=True,
        inference_attn_kernel="sdpa",
        compile_strategy="per_block",
    )

    for L in [73, 96]:
        print(f"\n=== L={L} ===")
        inp = _make_inputs(L, device)
        batch = w.build_batch(
            distogram_probs=inp["distogram_probs"],
            residue_type=inp["residue_type"],
            mask=inp["mask"],
            template_mode="distogram_only",
            seed=0,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            out = w.model.forward_inference(batch)
        torch.cuda.synchronize()
        print(f"  dt={time.perf_counter() - t0:.2f}s pTM={float(out['ptm_score'].item()):.4f}")

    counters = torch._dynamo.utils.counters
    print("\n=== counters ===")
    for k, v in counters.items():
        if v:
            print(f"  {k}: {dict(v)}")


if __name__ == "__main__":
    raise SystemExit(main())
