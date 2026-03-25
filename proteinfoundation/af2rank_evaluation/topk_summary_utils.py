"""
Shared utilities for generating the per-protein AF2Rank top-k summary CSV.

Used by both:
  - prediction_pipeline/run_af2rank_prediction.py  (no ground truth)
  - af2rank_evaluation/run_af2rank_on_proteinebm_topk.py  (with ground truth)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CA coordinate extraction
# ---------------------------------------------------------------------------

def _extract_ca_coords_from_pdb(pdb_path: str) -> np.ndarray:
    """Return Nx3 float32 array of CA coordinates from a PDB file.

    Works for both CA-only and all-atom PDBs.
    """
    coords: List[List[float]] = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def _compute_cg2all_metrics(orig_ca_path: str, allatom_path: str) -> Dict[str, float]:
    """Compute RMSD and TM-score of cg2all all-atom structure vs original CA-only backbone.

    Uses:
      - RMSD: direct numpy computation on aligned CA coordinates (no external tool needed)
      - TM-score: via tmscore() from af2rank_openfold_scorer (USalign if available)

    Returns dict with keys 'cg2all_rmsd_to_original' and 'cg2all_tm_to_original'.
    Returns NaN values on any error.
    """
    nan = float("nan")
    result = {"cg2all_rmsd_to_original": nan, "cg2all_tm_to_original": nan}
    try:
        orig_ca = _extract_ca_coords_from_pdb(orig_ca_path)
        recon_ca = _extract_ca_coords_from_pdb(allatom_path)
    except Exception as e:
        logger.debug(f"cg2all metric: failed to extract CA coords: {e}")
        return result

    if len(orig_ca) == 0 or len(recon_ca) == 0:
        logger.debug(f"cg2all metric: empty CA coords for {orig_ca_path} or {allatom_path}")
        return result

    if len(orig_ca) != len(recon_ca):
        logger.debug(
            f"cg2all metric: length mismatch {len(orig_ca)} vs {len(recon_ca)} "
            f"for {Path(orig_ca_path).name}"
        )
        return result

    # RMSD (no superposition — cg2all preserves absolute coordinates)
    rmsd = float(np.sqrt(np.mean(np.sum((orig_ca - recon_ca) ** 2, axis=-1))))
    result["cg2all_rmsd_to_original"] = rmsd

    # TM-score via USalign (falls back to 0.0 if USalign not in PATH)
    try:
        from proteinfoundation.af2rank_evaluation.af2rank_openfold_scorer import tmscore
        tm_out = tmscore(orig_ca, recon_ca)
        result["cg2all_tm_to_original"] = float(tm_out.get("tms", nan))
    except Exception as e:
        logger.debug(f"cg2all metric: TM-score failed ({e}); leaving as NaN")

    return result


# ---------------------------------------------------------------------------
# Top-1 / top-5 aggregate helpers (mirrors run_af2rank_on_proteinebm_topk.py)
# ---------------------------------------------------------------------------

def _extract_top1_top5_metrics(af2_df: pd.DataFrame, staged_filenames: set) -> Dict[str, float]:
    """Extract top-1 (by pTM) and top-5 (best tm_ref_pred among top-5 by pTM) metrics."""
    nan = float("nan")
    out = {
        "top_1_ptm": nan, "top_1_tm_ref_pred": nan, "top_1_composite": nan, "top_1_plddt": nan,
        "top_5_ptm": nan, "top_5_tm_ref_pred": nan, "top_5_composite": nan, "top_5_plddt": nan,
    }
    needed = {"structure_file", "ptm"}
    if not needed.issubset(af2_df.columns):
        return out
    df = af2_df[af2_df["structure_file"].astype(str).isin(staged_filenames)].copy()
    df["ptm"] = pd.to_numeric(df["ptm"], errors="coerce")
    if "tm_ref_pred" in df.columns:
        df["tm_ref_pred"] = pd.to_numeric(df["tm_ref_pred"], errors="coerce")
    else:
        df["tm_ref_pred"] = nan
    if "composite" in df.columns:
        df["composite"] = pd.to_numeric(df["composite"], errors="coerce")
    else:
        df["composite"] = nan
    if "plddt" in df.columns:
        df["plddt"] = pd.to_numeric(df["plddt"], errors="coerce")
    else:
        df["plddt"] = nan
    df = df.dropna(subset=["ptm"]).sort_values("ptm", ascending=False).reset_index(drop=True)
    if len(df) == 0:
        return out
    r1 = df.iloc[0]
    out["top_1_ptm"] = float(r1["ptm"])
    out["top_1_tm_ref_pred"] = float(r1["tm_ref_pred"]) if pd.notna(r1["tm_ref_pred"]) else nan
    out["top_1_composite"] = float(r1["composite"]) if pd.notna(r1["composite"]) else nan
    out["top_1_plddt"] = float(r1["plddt"]) if pd.notna(r1["plddt"]) else nan
    top5 = df.head(5)
    # Best tm_ref_pred among top-5; if all NaN fall back to top-1
    if top5["tm_ref_pred"].notna().any():
        best_idx = int(top5["tm_ref_pred"].argmax())
    else:
        best_idx = 0
    r5 = top5.iloc[best_idx]
    out["top_5_ptm"] = float(r5["ptm"])
    out["top_5_tm_ref_pred"] = float(r5["tm_ref_pred"]) if pd.notna(r5["tm_ref_pred"]) else nan
    out["top_5_composite"] = float(r5["composite"]) if pd.notna(r5["composite"]) else nan
    out["top_5_plddt"] = float(r5["plddt"]) if pd.notna(r5["plddt"]) else nan
    return out


def _merge_min_across_models(
    m1_df: pd.DataFrame, m2_df: pd.DataFrame, staged_filenames: set
) -> pd.DataFrame:
    """Merge model_1 and model_2 results, taking element-wise min per metric.

    Each metric (ptm, tm_ref_pred, composite, plddt) is independently
    minimised across the two models for every structure.  This means that,
    for a given structure, the reported ``ptm`` and ``tm_ref_pred`` may
    originate from *different* models — the result is a conservative /
    pessimistic estimate rather than a consistent single-model selection.
    """
    needed = {"structure_file", "ptm"}
    for df in (m1_df, m2_df):
        if not needed.issubset(df.columns):
            return pd.DataFrame(columns=["structure_file", "ptm", "tm_ref_pred", "composite", "plddt"])
    m1 = m1_df[m1_df["structure_file"].astype(str).isin(staged_filenames)].copy()
    m2 = m2_df[m2_df["structure_file"].astype(str).isin(staged_filenames)].copy()
    for c in ["tm_ref_pred", "composite", "plddt"]:
        if c not in m1.columns:
            m1[c] = float("nan")
        if c not in m2.columns:
            m2[c] = float("nan")
    merged = m1.merge(m2, on="structure_file", how="inner", suffixes=("_m1", "_m2"))
    # Element-wise min across models (independent per metric — see docstring).
    merged["ptm"] = merged[["ptm_m1", "ptm_m2"]].min(axis=1)
    merged["tm_ref_pred"] = merged[["tm_ref_pred_m1", "tm_ref_pred_m2"]].min(axis=1)
    merged["composite"] = merged[["composite_m1", "composite_m2"]].min(axis=1)
    merged["plddt"] = merged[["plddt_m1", "plddt_m2"]].min(axis=1)
    return merged[["structure_file", "ptm", "tm_ref_pred", "composite", "plddt"]]


# ---------------------------------------------------------------------------
# Main summary CSV writer
# ---------------------------------------------------------------------------

def generate_topk_summary_csv(
    protein_id: str,
    topk_df: pd.DataFrame,
    m1_csv: Path,
    m2_csv: Path,
    allatom_map: Dict[str, str],
    out_dir: Path,
) -> None:
    """Write per-template CSV and protein-level summary JSON to out_dir.

    Writes two files:

    1. ``af2rank_topk_summary_{protein_id}.csv`` — one row per top-k template:
       - template metadata (file, path, cg2all path)
       - ProteinEBM energy
       - cg2all fidelity vs original CA backbone (RMSD, TM-score)
       - template vs ground-truth metrics from AF2Rank (NaN in prediction mode)
       - per-template AF2Rank metrics: m1/m2/min × ptm/plddt/composite/tm_ref_pred

    2. ``af2rank_topk_protein_summary_{protein_id}.json`` — protein-level scalar aggregates:
       - ``protein_id``, ``n_templates``
       - ``m1/m2/min_top_1/top_5 × ptm/tm_ref_pred/composite/plddt`` (24 metrics total)

    Args:
        topk_df:    Energy-ranked top-k dataframe (columns: structure_path, energy, ...).
        m1_csv:     Path to model_1_ptm AF2Rank scores CSV.
        m2_csv:     Path to model_2_ptm AF2Rank scores CSV.
        allatom_map: {orig_ca_pdb_path: cg2all_allatom_pdb_path}.  May be partial or empty.
        out_dir:    Directory where output files will be written (af2rank_on_proteinebm_top_k/).
    """
    nan = float("nan")
    out_path = out_dir / f"af2rank_topk_summary_{protein_id}.csv"

    # ── Load AF2Rank score CSVs ──────────────────────────────────────────────
    try:
        m1_df = pd.read_csv(m1_csv)
    except Exception as e:
        logger.warning(f"{protein_id}: cannot read m1 CSV {m1_csv}: {e}")
        return
    try:
        m2_df = pd.read_csv(m2_csv)
    except Exception as e:
        logger.warning(f"{protein_id}: cannot read m2 CSV {m2_csv}: {e}")
        return

    # ── Build set of top-k template filenames ────────────────────────────────
    topk_df = topk_df.copy()
    topk_df["template_file"] = topk_df["structure_path"].apply(lambda p: Path(str(p)).name)
    staged_filenames = set(topk_df["template_file"].astype(str))

    # ── Merge AF2Rank metrics per template ──────────────────────────────────
    metric_cols = [
        "structure_file",
        "predicted_structure_file",
        "predicted_structure_path",
        "ptm",
        "plddt",
        "composite",
        "tm_ref_pred",
        "tm_ref_template",
        "rmsd_ref_template",
        "tm_template_pred",
        "rmsd_template_pred",
    ]

    def _safe_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        present = [c for c in cols if c in df.columns]
        return df[present].copy()

    m1_sub = _safe_cols(m1_df, metric_cols).rename(
        columns={c: f"m1_{c}" for c in metric_cols if c != "structure_file"}
    )
    m2_sub = _safe_cols(m2_df, metric_cols).rename(
        columns={c: f"m2_{c}" for c in metric_cols if c != "structure_file"}
    )
    min_df = _merge_min_across_models(m1_df, m2_df, staged_filenames)
    min_sub = min_df.rename(columns={c: f"min_{c}" for c in ["ptm", "plddt", "composite", "tm_ref_pred"]})

    # Merge all onto topk_df (left join so every top-k template has a row)
    merged = topk_df.merge(m1_sub, left_on="template_file", right_on="structure_file", how="left")
    merged = merged.merge(m2_sub, left_on="template_file", right_on="structure_file", how="left", suffixes=("", "_m2dup"))
    merged = merged.merge(min_sub, left_on="template_file", right_on="structure_file", how="left", suffixes=("", "_mindup"))
    # Drop duplicate structure_file columns from right sides
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_m2dup") or c.endswith("_mindup") or
                                   (c == "structure_file" and c in merged.columns and merged.columns.tolist().count(c) > 1)],
                         errors="ignore")
    # Clean up any extra structure_file columns
    sf_cols = [c for c in merged.columns if c == "structure_file"]
    if sf_cols:
        merged = merged.drop(columns=sf_cols, errors="ignore")

    # ── Compute cg2all fidelity metrics ─────────────────────────────────────
    cg2all_paths: List[str] = []
    cg2all_rmsds: List[float] = []
    cg2all_tms: List[float] = []

    for _, row in topk_df.iterrows():
        orig_ca = str(row["structure_path"])
        allatom = allatom_map.get(orig_ca)
        cg2all_paths.append(allatom if allatom else "")
        if allatom and Path(allatom).exists():
            metrics = _compute_cg2all_metrics(orig_ca, allatom)
            cg2all_rmsds.append(metrics["cg2all_rmsd_to_original"])
            cg2all_tms.append(metrics["cg2all_tm_to_original"])
        else:
            cg2all_rmsds.append(nan)
            cg2all_tms.append(nan)

    merged["cg2all_path"] = cg2all_paths
    merged["cg2all_rmsd_to_original"] = cg2all_rmsds
    merged["cg2all_tm_to_original"] = cg2all_tms

    # ── Compute protein-level top-1/top-5 aggregates ────────────────────────
    m1_agg = _extract_top1_top5_metrics(m1_df, staged_filenames)
    m2_agg = _extract_top1_top5_metrics(m2_df, staged_filenames)
    min_agg = _extract_top1_top5_metrics(min_df, staged_filenames)

    # ── Assemble final column order (per-template only — no aggregates) ──────
    leading_cols = [
        "template_file",
        "structure_path",
        "cg2all_path",
        "energy",
        "cg2all_rmsd_to_original",
        "cg2all_tm_to_original",
        "m1_tm_ref_template",
        "m1_rmsd_ref_template",
        "m1_predicted_structure_path",
        "m1_tm_template_pred",
        "m1_rmsd_template_pred",
        "m1_ptm", "m1_plddt", "m1_composite", "m1_tm_ref_pred",
        "m2_predicted_structure_path",
        "m2_tm_template_pred",
        "m2_rmsd_template_pred",
        "m2_ptm", "m2_plddt", "m2_composite", "m2_tm_ref_pred",
        "min_ptm", "min_plddt", "min_composite", "min_tm_ref_pred",
    ]
    # Keep only columns that exist; append any remaining columns at the end
    present_leading = [c for c in leading_cols if c in merged.columns]
    remaining = [c for c in merged.columns if c not in set(leading_cols)]
    final_cols = present_leading + remaining
    merged = merged[final_cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.info(f"{protein_id}: wrote {len(merged)}-row topk summary to {out_path}")

    # ── Write protein-level summary JSON ────────────────────────────────────
    summary = {"protein_id": protein_id, "n_templates": len(merged)}
    summary.update({f"m1_{k}": v for k, v in m1_agg.items()})
    summary.update({f"m2_{k}": v for k, v in m2_agg.items()})
    summary.update({f"min_{k}": v for k, v in min_agg.items()})
    json_path = out_dir / f"af2rank_topk_protein_summary_{protein_id}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"{protein_id}: wrote protein-level summary to {json_path}")
