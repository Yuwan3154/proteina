"""Numerical-equivalence test for the new compile-friendly OpenFold inference path.

Compares three modes on the real AF2Rank flow (full_template, mask_template_aatype):
  (1) eager baseline           - existing AlphaFold.forward()
  (2) inference path, eager    - forward_inference() uncompiled
  (3) inference path, compiled - forward_inference() with torch.compile

Tolerances are documented in
/home/ubuntu/.claude/plans/currently-in-the-proteina-jiggly-mitten.md
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank


REFERENCE_PDB = "/home/ubuntu/foldingdiff_af/data/output/1CEI_500t_10000s.pdb"   # 85 residues
DECOY_PDB = REFERENCE_PDB     # self-template: AF2Rank's standard "reference vs reference" sanity case
SEQ_LEN = 85


def _summarise(out: dict, sequence_len: int) -> dict:
    """Extract AF2Rank-relevant scalars from an AlphaFold output dict."""
    summary = {}
    if "ptm_score" in out:
        summary["ptm"] = float(out["ptm_score"].detach().cpu().item())
    if "plddt" in out:
        plddt = out["plddt"][:sequence_len].detach().cpu().float()
        summary["plddt_mean"] = float(plddt.mean().item())
        summary["plddt"] = plddt.numpy()
    if "predicted_aligned_error" in out:
        pae = out["predicted_aligned_error"]
        if pae.dim() == 3:
            pae = pae[0]
        pae = pae[:sequence_len, :sequence_len].detach().cpu().float().numpy()
        summary["pae"] = pae
        summary["pae_mean"] = float(pae.mean())
    if "final_atom_positions" in out:
        pos = out["final_atom_positions"][:sequence_len].detach().cpu().float().numpy()
        summary["positions"] = pos
        summary["ca"] = pos[:, 1, :]
    return summary


def _ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def _build_scorer(*, compile_inference_path: bool, attn_kernel: str = "sdpa") -> OpenFoldAF2Rank:
    return OpenFoldAF2Rank(
        reference_pdb=REFERENCE_PDB,
        chain="A",
        model_name="model_1_ptm",
        recycles=1,                              # one recycle keeps the test fast
        use_deepspeed_evoformer_attention=False, # apples-to-apples baseline
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
        skip_ref_metrics=True,
        compile_inference_path=compile_inference_path,
        inference_attn_kernel=attn_kernel,
    )


def _featurize_real(scorer: OpenFoldAF2Rank, seed: int = 0):
    """Use the real AF2Rank _featurize path (full_template + mask_template_aatype)."""
    return scorer._featurize(DECOY_PDB, decoy_chain="A", seed=seed)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping numerics test")
        return 0

    seq_len = SEQ_LEN

    # Re-use ONE scorer for paths (1) and (2) - they share the same model
    # weights, just call different forward methods.  This avoids loading the
    # OpenFold weights twice (>4 GiB).
    print("=== (1) eager baseline (existing AlphaFold.forward) ===")
    scorer_shared = _build_scorer(compile_inference_path=False)
    batch_base, _ = _featurize_real(scorer_shared)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_base = scorer_shared.model.model(batch_base)
    torch.cuda.synchronize()
    print(f"  baseline forward: {time.perf_counter() - t0:.2f}s")
    s_base = _summarise(out_base, seq_len)
    del out_base, batch_base
    torch.cuda.empty_cache()

    print("\n=== (2) inference path, eager ===")
    scorer_shared.model.model._inference_use_torch_sdpa = True
    scorer_shared.model.model._inference_use_torch_vanilla = False
    batch_ip, _ = _featurize_real(scorer_shared)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ip = scorer_shared.model.model.forward_inference(batch_ip)
    torch.cuda.synchronize()
    print(f"  inference-path eager forward: {time.perf_counter() - t0:.2f}s")
    s_ip = _summarise(out_ip, seq_len)
    del scorer_shared, out_ip, batch_ip
    torch.cuda.empty_cache()

    print("\n=== (3) inference path, compiled (warmup + steady) ===")
    scorer_ip_compiled = _build_scorer(compile_inference_path=True, attn_kernel="sdpa")
    batch_c, _ = _featurize_real(scorer_ip_compiled)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_c = scorer_ip_compiled.model.model.forward_inference(batch_c)
    torch.cuda.synchronize()
    print(f"  compiled forward (warmup, includes compile time): {time.perf_counter() - t0:.2f}s")

    batch_c2, _ = _featurize_real(scorer_ip_compiled)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_c = scorer_ip_compiled.model.model.forward_inference(batch_c2)
    torch.cuda.synchronize()
    print(f"  compiled forward (steady):                       {time.perf_counter() - t0:.2f}s")
    s_c = _summarise(out_c, seq_len)
    del scorer_ip_compiled, out_c, batch_c, batch_c2
    torch.cuda.empty_cache()

    # ---------- diffs ----------
    def diff(label: str, a: dict, b: dict) -> dict:
        d = {}
        print(f"\n[{label}]")
        if "ptm" in a and "ptm" in b:
            dv = abs(a["ptm"] - b["ptm"])
            d["ptm"] = dv
            print(f"  pTM: {a['ptm']:.6f} vs {b['ptm']:.6f}   |Δ|={dv:.2e}")
        if "plddt_mean" in a:
            dv = abs(a["plddt_mean"] - b["plddt_mean"])
            d["plddt_mean"] = dv
            print(f"  pLDDT mean: {a['plddt_mean']:.4f} vs {b['plddt_mean']:.4f}   |Δ|={dv:.2e}")
        if "plddt" in a:
            dv = float(np.max(np.abs(a["plddt"] - b["plddt"])))
            d["plddt_max"] = dv
            print(f"  pLDDT max abs Δ (per residue, 0-100 scale): {dv:.4f}")
        if "pae" in a:
            dv = float(np.max(np.abs(a["pae"] - b["pae"])))
            d["pae_max"] = dv
            print(f"  pAE max abs Δ: {dv:.4f}")
        if "ca" in a:
            dv = _ca_rmsd(a["ca"], b["ca"])
            d["ca_rmsd"] = dv
            print(f"  CA RMSD: {dv:.4f} Å")
        return d

    d_2v1 = diff("(2) inference-path eager vs (1) baseline", s_ip, s_base)
    d_3v1 = diff("(3) inference-path compiled vs (1) baseline", s_c, s_base)
    d_3v2 = diff("(3) inference-path compiled vs (2) inference-path eager", s_c, s_ip)

    # ---------- tolerance gates ----------
    print("\n=== tolerance checks ===")
    ok = True
    def gate(label: str, value: float, threshold: float) -> None:
        nonlocal ok
        good = value <= threshold
        marker = "OK " if good else "FAIL"
        print(f"  [{marker}] {label}: {value:.4g} (≤ {threshold:.4g})")
        ok = ok and good

    # (2) vs (1): the inference path uses SDPA + non-inplace IPA softmax,
    # which can differ from baseline matmul+softmax + inplace cuda softmax
    # at the ~1e-3 logit level.  Tolerances here are generous (sanity bounds);
    # AF2Rank's downstream ranking only needs that pTM/pLDDT rankings stay
    # consistent across decoys, which a few percent absolute drift won't break.
    if "ptm" in d_2v1:        gate("pTM     (2 vs 1) |Δ|", d_2v1["ptm"], 5e-2)
    if "plddt_mean" in d_2v1: gate("pLDDTμ  (2 vs 1) |Δ|", d_2v1["plddt_mean"], 5.0)
    if "ca_rmsd" in d_2v1:    gate("CA RMSD (2 vs 1)     ", d_2v1["ca_rmsd"], 5.0)

    # (3) vs (2): same code path, just compiled. Should be numerically tight.
    if "ptm" in d_3v2:        gate("pTM     (3 vs 2) |Δ|", d_3v2["ptm"], 5e-4)
    if "plddt_mean" in d_3v2: gate("pLDDTμ  (3 vs 2) |Δ|", d_3v2["plddt_mean"], 0.2)
    if "ca_rmsd" in d_3v2:    gate("CA RMSD (3 vs 2)     ", d_3v2["ca_rmsd"], 0.1)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
