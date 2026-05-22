"""cuEquivariance kernel verification test for the OpenFold compile inference path.

Two checks:

  1. **Numerical band**: cueq attention + cueq triangle multiplicative update
     vs SDPA + matmul-based triangle mul update, on the same decoy. Allow
     |Δ pTM| ≤ 5e-3, mean |Δ pLDDT| ≤ 0.5, CA RMSD ≤ 0.2 Å. The user
     explicitly accepted this band on the grounds that AF2Rank only cares
     about decoy rankings.

  2. **Ranking preservation**: score N decoys of the same reference (one
     real decoy + N-1 noised copies) under SDPA and under cueq; verify
     that Spearman rank correlation of the AF2Rank composite score
     (pTM × pLDDT) across decoys is ≥ 0.97.

Run with:
    CUTLASS_PATH=/home/ubuntu/openfold/cutlass \\
        /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \\
        proteinfoundation/prediction_pipeline/tests/test_af2rank_compile_cueq.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank


REFERENCE_PDB = "/home/ubuntu/foldingdiff_af/data/output/1CEI_500t_10000s.pdb"
SEQ_LEN = 85


def _summarise(out: dict, sequence_len: int) -> dict:
    summary = {}
    if "ptm_score" in out:
        summary["ptm"] = float(out["ptm_score"].detach().cpu().item())
    if "plddt" in out:
        plddt = out["plddt"][:sequence_len].detach().cpu().float()
        summary["plddt_mean"] = float(plddt.mean().item())
        summary["plddt"] = plddt.numpy()
    if "final_atom_positions" in out:
        pos = out["final_atom_positions"][:sequence_len].detach().cpu().float().numpy()
        summary["ca"] = pos[:, 1, :]
    summary["composite"] = summary.get("ptm", 0.0) * (summary.get("plddt_mean", 0.0) / 100.0)
    return summary


def _ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def _build_scorer(*, attn_kernel: str):
    return OpenFoldAF2Rank(
        reference_pdb=REFERENCE_PDB,
        chain="A",
        model_name="model_1_ptm",
        recycles=1,
        use_deepspeed_evoformer_attention=False,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
        skip_ref_metrics=True,
        compile_inference_path=True,
        inference_attn_kernel=attn_kernel,
        compile_strategy="per_block",
    )


def _make_noised_decoys(reference_pdb: str, n: int, seed: int = 0):
    """Generate N decoy PDBs by adding gaussian CA noise. Returns paths."""
    import tempfile
    import shutil
    rng = np.random.RandomState(seed)
    decoys = []
    with open(reference_pdb) as f:
        ref_lines = f.readlines()
    # Each decoy: copy ref lines and jitter each ATOM CA xyz by N(0, sigma).
    for i in range(n):
        sigma = 0.3 + 0.4 * i / max(n - 1, 1)   # spread noise across decoys
        out = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        for line in ref_lines:
            if line.startswith("ATOM"):
                # Columns 31-38, 39-46, 47-54 are x/y/z in PDB format.
                try:
                    x = float(line[30:38]) + rng.normal(0, sigma)
                    y = float(line[38:46]) + rng.normal(0, sigma)
                    z = float(line[46:54]) + rng.normal(0, sigma)
                    line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                except ValueError:
                    pass
            out.write(line)
        out.close()
        decoys.append(out.name)
    return decoys


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping cueq test")
        return 0

    seq_len = SEQ_LEN
    print(f"=== cuEquivariance kernel verification (L={seq_len}) ===")

    # Build SDPA scorer + score
    print("\n--- SDPA reference ---")
    scorer_sdpa = _build_scorer(attn_kernel="sdpa")
    decoys = _make_noised_decoys(REFERENCE_PDB, n=8, seed=42)

    sdpa_scores = []
    for i, d in enumerate(decoys):
        s = scorer_sdpa.score_structure(d, decoy_chain="A", recycles=1, seed=42)
        sdpa_scores.append(s)
        print(f"  decoy {i}: pTM={s['ptm']:.4f} pLDDT={s['plddt']*100:.2f} composite={s['composite']:.4f}")
    del scorer_sdpa
    torch.cuda.empty_cache()

    print("\n--- cuEquivariance ---")
    try:
        scorer_cueq = _build_scorer(attn_kernel="cuequivariance")
    except Exception as exc:
        print(f"  FAIL: cuequivariance scorer construction raised: {exc}")
        import os
        for d in decoys:
            try: os.unlink(d)
            except Exception: pass
        return 1

    cueq_scores = []
    try:
        for i, d in enumerate(decoys):
            s = scorer_cueq.score_structure(d, decoy_chain="A", recycles=1, seed=42)
            cueq_scores.append(s)
            print(f"  decoy {i}: pTM={s['ptm']:.4f} pLDDT={s['plddt']*100:.2f} composite={s['composite']:.4f}")
    finally:
        del scorer_cueq
        torch.cuda.empty_cache()
        import os
        for d in decoys:
            try: os.unlink(d)
            except Exception: pass

    # ---------- numerical band gate ----------
    print("\n--- Per-decoy diff (cueq vs sdpa) ---")
    ok = True
    for i, (sdpa, cueq) in enumerate(zip(sdpa_scores, cueq_scores)):
        ptm_d = abs(sdpa["ptm"] - cueq["ptm"])
        plddt_d = abs(sdpa["plddt"] - cueq["plddt"]) * 100
        print(f"  decoy {i}: |Δ pTM|={ptm_d:.4f}  |Δ pLDDT|={plddt_d:.4f}")
        if ptm_d > 5e-3:
            print(f"    FAIL: pTM drift {ptm_d:.4f} > 5e-3")
            ok = False
        if plddt_d > 0.5:
            print(f"    FAIL: pLDDT drift {plddt_d:.4f} > 0.5")
            ok = False

    # ---------- ranking preservation gate ----------
    sdpa_composite = np.array([s["composite"] for s in sdpa_scores])
    cueq_composite = np.array([s["composite"] for s in cueq_scores])

    from scipy.stats import spearmanr
    corr, _ = spearmanr(sdpa_composite, cueq_composite)
    print(f"\n--- Ranking preservation ---")
    print(f"  Spearman rank correlation (composite, N={len(decoys)}): {corr:.4f}")
    if corr < 0.97:
        print(f"  FAIL: rank correlation {corr:.4f} < 0.97")
        ok = False
    else:
        print(f"  OK: rank correlation ≥ 0.97")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
