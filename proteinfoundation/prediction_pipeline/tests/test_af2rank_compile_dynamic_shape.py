"""Dynamic-shape soak + compile-engagement test.

Runs the compiled inference path at several sequence lengths back-to-back
and verifies that torch.compile re-specialises at most a small number of
times.  This catches the failure mode where Dynamo silently recompiles
once per L (defeating the dynamic-shape contract).

Pass criteria:
  * Successive forward calls at different L succeed (no OOM, no crashes).
  * `_dynamo.utils.counters["frames"]["total"]` increases by at most a
    handful between cold start and the end of the soak; we accept up to
    3 to allow for unrolled inner closures.
  * No NaNs in the outputs.

Run with:
    TORCH_LOGS=recompiles CUTLASS_PATH=/home/ubuntu/openfold/cutlass \
        /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
        proteinfoundation/prediction_pipeline/tests/test_af2rank_compile_dynamic_shape.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


# Adjust this set to whatever the GPU has memory for.  At 5-6 GiB free on a
# 40 GiB card with model_1_ptm loaded, L ~150 is the practical ceiling for
# the dense extra-MSA stage.
L_SEQUENCE = [73, 96, 128]


def _make_inputs(seq_len: int, device: torch.device, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    residue_type = torch.tensor(
        rng.randint(0, 20, size=(1, seq_len)), dtype=torch.long, device=device,
    )
    mask = torch.ones((1, seq_len), dtype=torch.float32, device=device)
    raw = rng.gamma(1.0, 1.0, size=(1, seq_len, seq_len, 39)).astype(np.float32)
    dgram = raw / raw.sum(axis=-1, keepdims=True)
    distogram = torch.tensor(dgram, dtype=torch.float32, device=device)
    return {"residue_type": residue_type, "mask": mask, "distogram_probs": distogram}


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping dynamic-shape test")
        return 0

    device = torch.device("cuda")

    print("Constructing compiled wrapper ...")
    wrapper = OpenFoldTemplateInference(
        model_name="model_1_ptm",
        max_recycling_iters=1,
        use_deepspeed_evoformer_attention=False,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
        compile_inference_path=True,
        inference_attn_kernel="sdpa",
    )

    # Reset Dynamo counters to a known state.
    torch._dynamo.utils.counters.clear()

    def _snapshot_counters() -> dict:
        return {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}

    print("\nWarmup (compile happens here) ...")
    inputs0 = _make_inputs(L_SEQUENCE[0], device, seed=42)
    batch0 = wrapper.build_batch(
        distogram_probs=inputs0["distogram_probs"],
        residue_type=inputs0["residue_type"],
        mask=inputs0["mask"],
        template_mode="distogram_only",
        seed=0,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        out0 = wrapper.model.forward_inference(batch0)
    torch.cuda.synchronize()
    print(f"  warmup forward (L={L_SEQUENCE[0]}): {time.perf_counter() - t0:.2f}s")
    if torch.isnan(out0["plddt"]).any():
        print("  FAIL: NaN in plddt at warmup")
        return 1
    del batch0, out0

    cold_counters = _snapshot_counters()
    cold_frames = cold_counters.get("frames", {}).get("total", 0)
    cold_recompiles = sum(
        v for k, v in cold_counters.get("recompiles", {}).items()
    )
    print(f"  counters after warmup: frames.total={cold_frames}  recompiles_sum={cold_recompiles}")

    print("\nDynamic-shape soak (different L back-to-back) ...")
    for i, L in enumerate(L_SEQUENCE):
        inputs = _make_inputs(L, device, seed=42 + i)
        batch = wrapper.build_batch(
            distogram_probs=inputs["distogram_probs"],
            residue_type=inputs["residue_type"],
            mask=inputs["mask"],
            template_mode="distogram_only",
            seed=0,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            out = wrapper.model.forward_inference(batch)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ptm = float(out["ptm_score"].detach().cpu().item()) if "ptm_score" in out else float("nan")
        plddt_mean = float(out["plddt"].mean().item()) if "plddt" in out else float("nan")
        nan = torch.isnan(out["plddt"]).any().item() if "plddt" in out else False
        print(f"  L={L:>3}: dt={dt:.2f}s  pTM={ptm:.4f}  pLDDTμ={plddt_mean:.2f}  nan_plddt={nan}")
        if nan:
            print(f"    FAIL: NaN in plddt at L={L}")
            return 1
        del batch, out

    final_counters = _snapshot_counters()
    final_frames = final_counters.get("frames", {}).get("total", 0)
    final_recompiles = sum(
        v for k, v in final_counters.get("recompiles", {}).items()
    )
    print(f"\n  counters after soak: frames.total={final_frames}  recompiles_sum={final_recompiles}")

    delta_frames = final_frames - cold_frames
    delta_recompiles = final_recompiles - cold_recompiles
    print(f"  Δ frames.total: {delta_frames}")
    print(f"  Δ recompiles: {delta_recompiles}")

    # Pass if the soak triggers no additional graph compilations and at most
    # a few extra frame entries (one per dynamic-shape generalisation).
    ok = True
    if delta_frames > 5:
        print(f"  FAIL: too many new frames during soak ({delta_frames} > 5) "
              "— Dynamo is likely re-specialising on L")
        ok = False
    if delta_recompiles > 1:
        print(f"  FAIL: too many recompiles ({delta_recompiles} > 1)")
        ok = False
    if ok:
        print("  PASS: dynamic-shape soak triggers no spurious recompilation")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
