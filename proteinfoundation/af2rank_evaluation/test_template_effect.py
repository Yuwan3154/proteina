#!/usr/bin/env python3
"""
Test whether template backbone coordinates actually affect OpenFold predictions.

For each Proteina-generated PDB, scores it TWICE with the same seed:
  1. With template  (normal AF2Rank protocol — backbone coords from decoy PDB)
  2. Without template (backbone coords zeroed — sequence-only prediction)

If templates are used correctly:
  - mean tm_ref_pred (with) > mean tm_ref_pred (no)
  - tm_ref_template positively correlates with tm_ref_pred (with)

Usage (from ~/proteina):
  CUTLASS_PATH=/home/ubuntu/openfold/cutlass \
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
    proteinfoundation/af2rank_evaluation/test_template_effect.py
"""
import glob
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from af2rank_openfold_scorer import OpenFoldAF2Rank, tmscore, _USALIGN_PARALLEL_ENV

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROTEINS = [
    {
        "protein_id": "1enh_A",
        "reference_cif": os.path.expanduser("~/data/af2rank_single/pdb/en/1enh.cif"),
        "pdb_dir": os.path.expanduser(
            "~/proteina/inference/inference_test_8samples/1enh_A"
        ),
    },
    {
        "protein_id": "2a28_A",
        "reference_cif": os.path.expanduser("~/data/af2rank_single/pdb/a2/2a28.cif"),
        "pdb_dir": os.path.expanduser(
            "~/proteina/inference/inference_test_8samples/2a28_A"
        ),
    },
]

MODEL_NAME = "model_1_ptm"
RECYCLES = 1
CHAIN = "A"
SEED = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_pdb(scorer, pdb_path, no_template):
    scores = scorer.score_structure(
        pdb_path,
        decoy_chain=CHAIN,
        recycles=RECYCLES,
        seed=SEED,
        verbose=False,
        no_template=no_template,
    )
    return scores


def _extract_pred_ca(out):
    import torch
    if "final_atom_positions" in out:
        return out["final_atom_positions"][:, 1, :].detach().cpu().numpy()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_protein(cfg):
    protein_id = cfg["protein_id"]
    reference_cif = cfg["reference_cif"]
    pdb_dir = cfg["pdb_dir"]

    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, f"{protein_id}_*.pdb")))
    if not pdb_files:
        print(f"[{protein_id}] No PDB files found in {pdb_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Protein: {protein_id}  ({len(pdb_files)} structures)  seed={SEED}")
    print(f"Reference CIF: {reference_cif}")
    print(f"{'='*70}")

    scorer = OpenFoldAF2Rank(
        reference_cif,
        chain=CHAIN,
        model_name=MODEL_NAME,
        recycles=RECYCLES,
        use_deepspeed_evoformer_attention=True,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=True,
    )

    rows = []
    for pdb_path in pdb_files:
        pdb_file = os.path.basename(pdb_path)

        scores_with = _score_pdb(scorer, pdb_path, no_template=False)
        scores_no   = _score_pdb(scorer, pdb_path, no_template=True)

        rows.append({
            "pdb_file":          pdb_file,
            "tm_ref_template":   scores_with.get("tm_ref_template", float("nan")),
            "tm_ref_pred_with":  scores_with.get("tm_ref_pred",     float("nan")),
            "tm_ref_pred_no":    scores_no.get("tm_ref_pred",       float("nan")),
            "ptm_with":          scores_with.get("ptm",             float("nan")),
            "ptm_no":            scores_no.get("ptm",               float("nan")),
        })

    # Print table
    hdr = f"{'PDB file':<22} {'tm_ref_tmpl':>11} {'tm_pred_WITH':>12} {'tm_pred_NO':>10} {'ptm_WITH':>8} {'ptm_NO':>7} {'Δtm_pred':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        delta = r["tm_ref_pred_with"] - r["tm_ref_pred_no"]
        print(
            f"{r['pdb_file']:<22} "
            f"{r['tm_ref_template']:>11.4f} "
            f"{r['tm_ref_pred_with']:>12.4f} "
            f"{r['tm_ref_pred_no']:>10.4f} "
            f"{r['ptm_with']:>8.4f} "
            f"{r['ptm_no']:>7.4f} "
            f"{delta:>+8.4f}"
        )

    # Summaries
    tm_ref_tmpl  = np.array([r["tm_ref_template"]  for r in rows])
    tm_pred_with = np.array([r["tm_ref_pred_with"] for r in rows])
    tm_pred_no   = np.array([r["tm_ref_pred_no"]   for r in rows])
    ptm_with     = np.array([r["ptm_with"]         for r in rows])

    print()
    print(f"Mean tm_ref_pred  WITH template : {tm_pred_with.mean():.4f}  (std {tm_pred_with.std():.4f})")
    print(f"Mean tm_ref_pred  WITHOUT template: {tm_pred_no.mean():.4f}  (std {tm_pred_no.std():.4f})")
    print(f"Mean Δtm_pred (with - no)        : {(tm_pred_with - tm_pred_no).mean():+.4f}")
    print()

    if len(rows) >= 3:
        rho_tmpl_pred, p_tmpl_pred = spearmanr(tm_ref_tmpl, tm_pred_with)
        rho_ptm_pred,  p_ptm_pred  = spearmanr(ptm_with,    tm_pred_with)
        print(f"Spearman(tm_ref_template, tm_ref_pred_with): ρ={rho_tmpl_pred:+.3f}  p={p_tmpl_pred:.3f}")
        print(f"Spearman(ptm_with,        tm_ref_pred_with): ρ={rho_ptm_pred:+.3f}  p={p_ptm_pred:.3f}")
    else:
        print("(Too few structures for meaningful correlation — need ≥ 3)")

    print()
    if (tm_pred_with - tm_pred_no).mean() > 0.01:
        print("✅ Templates appear to HELP: mean tm_ref_pred is higher with template backbone.")
    elif (tm_pred_with - tm_pred_no).mean() < -0.01:
        print("⚠️  Templates appear to HURT: mean tm_ref_pred is lower with template backbone.")
    else:
        print("⚠️  Templates have MINIMAL EFFECT on tm_ref_pred (Δ < 0.01).")


def main():
    for cfg in PROTEINS:
        run_protein(cfg)


if __name__ == "__main__":
    main()
