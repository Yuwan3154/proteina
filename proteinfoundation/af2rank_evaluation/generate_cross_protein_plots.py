#!/usr/bin/env python3
"""
Cross-Protein Scoring Analysis

This script generates summary plots across all protein chains from scoring analysis results.
Supported scorers:
- AF2Rank (requires `af2rank_analysis/af2rank_summary_*.json`)
- AF2Rank-on-ProteinEBM-topk (requires `af2rank_on_proteinebm_top_k/af2rank_analysis/af2rank_scores_*.csv`)
- ProteinEBM
  - TM mode (default): requires `proteinebm_analysis/proteinebm_summary_*.json`
  - energy mode: requires `proteinebm_analysis/proteinebm_scores_*.csv`

Usage:
    python generate_cross_protein_plots.py --inference_dir <path> --output_dir <path> --scorer af2rank
    python generate_cross_protein_plots.py --inference_dir <path> --output_dir <path> --scorer proteinebm
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


def find_af2rank_summaries(inference_base_dir: str) -> List[str]:
    """
    Find all AF2Rank summary JSON files in the inference directory structure.
    
    Args:
        inference_base_dir: Base inference directory
        
    Returns:
        List of paths to AF2Rank summary JSON files
    """
    pattern = os.path.join(inference_base_dir, "*", "af2rank_analysis", "af2rank_summary_*.json")
    summary_files = glob.glob(pattern)
    logger.info(f"Found {len(summary_files)} AF2Rank summary files")
    return summary_files


def find_proteinebm_score_files(inference_base_dir: str, analysis_subdir: str = "proteinebm_v2_cathmd_analysis") -> List[str]:
    """
    Find all ProteinEBM per-protein score CSVs in the inference directory structure.

    Args:
        inference_base_dir: Base inference directory
        analysis_subdir: Per-protein subdir containing ProteinEBM outputs (e.g. proteinebm_analysis or proteinebm_v2_cathmd_analysis)

    Returns:
        List of paths to ProteinEBM score CSV files
    """
    pattern = os.path.join(inference_base_dir, "*", analysis_subdir, "proteinebm_scores_*.csv")
    score_files = glob.glob(pattern)
    logger.info(f"Found {len(score_files)} ProteinEBM score files")
    return score_files


def find_proteinebm_summaries(inference_base_dir: str, analysis_subdir: str = "proteinebm_v2_cathmd_analysis") -> List[str]:
    """
    Find all ProteinEBM per-protein summary JSON files in the inference directory structure.
    """
    pattern = os.path.join(inference_base_dir, "*", analysis_subdir, "proteinebm_summary_*.json")
    summary_files = glob.glob(pattern)
    logger.info(f"Found {len(summary_files)} ProteinEBM summary files")
    return summary_files


def find_af2rank_on_proteinebm_topk_score_files(inference_base_dir: str, top_k: int) -> List[str]:
    """
    Find all AF2Rank score CSVs produced by the AF2Rank-on-ProteinEBM-topk protocol.
    Expected path pattern:
      <inference_base_dir>/<protein_id>/af2rank_on_proteinebm_top_k/af2rank_analysis/af2rank_scores_<protein_id>.csv
    """
    pattern = os.path.join(
        inference_base_dir,
        "*",
        "af2rank_on_proteinebm_top_k",
        "af2rank_analysis",
        "af2rank_scores_*.csv",
    )
    score_files = glob.glob(pattern)
    logger.info(f"Found {len(score_files)} AF2Rank-on-ProteinEBM-topk score files (using folder af2rank_on_proteinebm_top_k)")
    return score_files


def load_af2rank_on_proteinebm_topk_data(
    score_files: List[str],
    dataset_file: str,
    id_col: str,
    tms_col: str,
    top_k: int,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
) -> pd.DataFrame:
    """
    Load per-protein summary metrics from AF2Rank-on-ProteinEBM-topk score CSVs.

    We rank templates by AF2Rank pTM (descending) and then take only the top-K entries
    (K is provided by the caller). For each protein:
      - top_1_tm_ref_pred: tm_ref_pred at rank 1
      - top_5_tm_ref_pred: best tm_ref_pred among top min(5, n) by pTM
      - top_1_tm_ref_template: tm_ref_template at rank 1 (same entry as top_1_tm_ref_pred)
      - top_5_tm_ref_template: tm_ref_template for the entry that achieves top_5_tm_ref_pred
      - n_scored: number of AF2Rank-scored templates available in that folder
      - min_energy_topk: minimum ProteinEBM energy among top-k templates
      - max_ptm_topk / m2_max_ptm_topk / min_max_ptm_topk: best pTM per model variant

    If fewer than 5 entries exist, we use whatever is present.
    """
    data = []

    dataset_df = pd.read_csv(dataset_file)
    dataset_proteins = set(dataset_df[id_col].tolist())
    logger.info(f"Dataset contains {len(dataset_proteins)} proteins")

    def _extract_metrics(df: pd.DataFrame, staged_names: set, top_k: int) -> dict:
        needed = {"ptm", "tm_ref_pred", "tm_ref_template", "structure_file"}
        if not needed.issubset(df.columns):
            return {}
        d = df[df["structure_file"].astype(str).isin(staged_names)].copy()
        d = d.dropna(subset=["ptm", "tm_ref_pred", "tm_ref_template"])
        if len(d) == 0:
            return {}
        for c in ["ptm", "tm_ref_pred", "tm_ref_template", "composite", "plddt"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["ptm", "tm_ref_pred"]).sort_values("ptm", ascending=False).reset_index(drop=True)
        if len(d) == 0:
            return {}
        k_eff = min(top_k, len(d))
        d = d.iloc[:k_eff]
        r1 = d.iloc[0]
        k5 = min(5, len(d))
        d5 = d.iloc[:k5]
        best_idx = int(np.argmax(d5["tm_ref_pred"].to_numpy()))
        r5 = d5.iloc[best_idx]
        return {
            "top_1_ptm": float(r1["ptm"]),
            "top_1_tm_ref_pred": float(r1["tm_ref_pred"]),
            "top_1_tm_ref_template": float(r1["tm_ref_template"]),
            "top_1_composite": float(r1["composite"]) if "composite" in r1 and pd.notna(r1.get("composite")) else float("nan"),
            "top_1_plddt": float(r1["plddt"]) if "plddt" in r1 and pd.notna(r1.get("plddt")) else float("nan"),
            "top_5_ptm": float(r5["ptm"]),
            "top_5_tm_ref_pred": float(r5["tm_ref_pred"]),
            "top_5_tm_ref_template": float(r5["tm_ref_template"]),
            "top_5_composite": float(r5["composite"]) if "composite" in r5 and pd.notna(r5.get("composite")) else float("nan"),
            "top_5_plddt": float(r5["plddt"]) if "plddt" in r5 and pd.notna(r5.get("plddt")) else float("nan"),
        }

    def _merge_min(m1_df: pd.DataFrame, m2_df: pd.DataFrame, staged_names: set) -> pd.DataFrame:
        """Merge model_1 and model_2 AF2Rank results, taking element-wise min per metric.

        Each metric (ptm, tm_ref_pred, composite, plddt, tm_ref_template) is
        independently minimised across the two models for every structure.  This
        means that, for a given structure, the reported ``ptm`` and
        ``tm_ref_pred`` may originate from *different* models — the result is a
        conservative / pessimistic estimate rather than a consistent single-model
        selection.
        """
        needed = {"structure_file", "ptm", "tm_ref_pred"}
        if not needed.issubset(m1_df.columns) or not needed.issubset(m2_df.columns):
            return pd.DataFrame()
        for c in ["composite", "plddt"]:
            if c not in m1_df.columns:
                m1_df = m1_df.copy()
                m1_df[c] = float("nan")
            if c not in m2_df.columns:
                m2_df = m2_df.copy()
                m2_df[c] = float("nan")
        m1 = m1_df[m1_df["structure_file"].astype(str).isin(staged_names)].copy()
        m2 = m2_df[m2_df["structure_file"].astype(str).isin(staged_names)].copy()
        merged = m1.merge(m2, on="structure_file", how="inner", suffixes=("_m1", "_m2"))
        # Element-wise min across models (independent per metric — see docstring).
        merged["ptm"] = merged[["ptm_m1", "ptm_m2"]].min(axis=1)
        merged["tm_ref_pred"] = merged[["tm_ref_pred_m1", "tm_ref_pred_m2"]].min(axis=1)
        merged["composite"] = merged[["composite_m1", "composite_m2"]].min(axis=1)
        merged["plddt"] = merged[["plddt_m1", "plddt_m2"]].min(axis=1)
        merged["tm_ref_template"] = merged[["tm_ref_template_m1", "tm_ref_template_m2"]].min(axis=1)
        return merged[["structure_file", "ptm", "tm_ref_pred", "tm_ref_template", "composite", "plddt"]]

    for score_file in score_files:
        protein_id = Path(score_file).parent.parent.parent.name
        if protein_id not in dataset_proteins:
            continue

        m1_df = pd.read_csv(score_file)
        needed = {"ptm", "tm_ref_pred", "tm_ref_template", "structure_file"}
        missing = sorted([c for c in needed if c not in m1_df.columns])
        if missing:
            logger.warning(
                f"Skipping {protein_id}: missing columns {missing} in {score_file} "
                f"(stale pre-fix CSV — re-run AF2Rank scoring to populate TM metrics). "
                f"Columns present: {m1_df.columns.tolist()}"
            )
            continue

        staged_dir = Path(score_file).parent.parent / "staged_topk_templates"
        staged_names = set(f.name for f in staged_dir.iterdir() if f.is_file() or f.is_symlink()) if staged_dir.exists() else set()
        if not staged_names:
            staged_names = set(m1_df["structure_file"].astype(str).tolist())
        if len(staged_names) == 0:
            continue

        m1_metrics = _extract_metrics(m1_df, staged_names, top_k)
        if not m1_metrics:
            continue

        m2_path = Path(score_file).parent.parent / "af2rank_analysis_model_2_ptm" / Path(score_file).name

        # Load ProteinEBM energy for this protein (min energy among top-k)
        protein_dir = Path(score_file).parent.parent.parent
        ebm_csv_path = protein_dir / proteinebm_analysis_subdir / f"proteinebm_scores_{protein_id}.csv"
        min_energy = float("nan")
        if ebm_csv_path.exists():
            try:
                ebm_df = pd.read_csv(ebm_csv_path)
                if "energy" in ebm_df.columns and "structure_file" in ebm_df.columns:
                    ebm_staged = ebm_df[ebm_df["structure_file"].astype(str).isin(staged_names)]
                    if len(ebm_staged) > 0:
                        min_energy = float(ebm_staged["energy"].astype(float).min())
            except Exception as e:
                logger.debug(f"Could not read ProteinEBM scores for {protein_id}: {e}")

        row = {
            "protein_id": protein_id,
            "n_scored": len(m1_df[m1_df["structure_file"].astype(str).isin(staged_names)]),
            "top_1_tm_ref_pred": m1_metrics["top_1_tm_ref_pred"],
            "top_5_tm_ref_pred": m1_metrics["top_5_tm_ref_pred"],
            "top_1_tm_ref_template": m1_metrics["top_1_tm_ref_template"],
            "top_5_tm_ref_template": m1_metrics["top_5_tm_ref_template"],
            "top_1_ptm": m1_metrics["top_1_ptm"],
            "top_5_ptm": m1_metrics["top_5_ptm"],
            "top_1_composite": m1_metrics["top_1_composite"],
            "top_5_composite": m1_metrics["top_5_composite"],
            "top_1_plddt": m1_metrics["top_1_plddt"],
            "top_5_plddt": m1_metrics["top_5_plddt"],
            "scores_csv": str(score_file),
            "min_energy_topk": min_energy,
            # max_ptm_topk = top_1_ptm (top-1 by pTM is the max pTM)
            "max_ptm_topk": m1_metrics["top_1_ptm"],
        }
        if m2_path.exists():
            m2_df = pd.read_csv(m2_path)
            if needed.issubset(m2_df.columns):
                m2_metrics = _extract_metrics(m2_df, staged_names, top_k)
                min_df = _merge_min(m1_df, m2_df, staged_names)
                min_metrics = _extract_metrics(min_df, staged_names, top_k) if len(min_df) > 0 else {}
                for prefix, metrics in [("m2", m2_metrics), ("min", min_metrics)]:
                    if metrics:
                        for k, v in metrics.items():
                            row[f"{prefix}_{k}"] = v
                # max pTM aliases for m2 and min variants
                if m2_metrics:
                    row["m2_max_ptm_topk"] = m2_metrics["top_1_ptm"]
                if min_metrics:
                    row["min_max_ptm_topk"] = min_metrics["top_1_ptm"]
        data.append(row)

    if len(data) == 0:
        base_cols = [
            "protein_id", "n_scored",
            "top_1_tm_ref_pred", "top_5_tm_ref_pred", "top_1_tm_ref_template", "top_5_tm_ref_template",
            "top_1_ptm", "top_5_ptm", "top_1_composite", "top_5_composite", "top_1_plddt", "top_5_plddt",
            "scores_csv", tms_col, "in_train", "length",
        ]
        for prefix in ["m2", "min"]:
            for k in ["top_1_ptm", "top_5_ptm", "top_1_tm_ref_pred", "top_5_tm_ref_pred", "top_1_composite", "top_5_composite", "top_1_plddt", "top_5_plddt"]:
                base_cols.append(f"{prefix}_{k}")
        return pd.DataFrame(columns=base_cols)

    out = pd.DataFrame(data)
    logger.info(f"Loaded {len(out)} proteins from AF2Rank-on-ProteinEBM-topk results (filtered by dataset)")

    out = out.merge(
        dataset_df[[id_col, tms_col, "in_train", "length"]],
        left_on="protein_id",
        right_on=id_col,
        how="left",
    )
    out.drop(columns=[id_col], inplace=True)
    return out


def create_af2rank_on_proteinebm_topk_plots(df: pd.DataFrame, output_dir: str, tms_col: str, top_k: int) -> None:
    """
    Create cross-protein plots for AF2Rank-on-ProteinEBM-topk:
      - Reference TM vs ProteinEBM energy (min energy among top-k)
      - Reference TM vs AF2Rank pTM (max pTM among top-k, per model variant)
      - Reference TM vs Top-1/Top-5 tm_ref_pred (rank by pTM)
      - Template TM vs Prediction TM (top-1/top-5, per model variant)
      - TM_ref_pred vs pTM / composite / pLDDT (per model variant)
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    # ── Helper: single-panel scatter ──────────────────────────────────────────
    def _scatter_single(x_data, y_data, df_valid, title, xlabel, ylabel, out_path, diagonal=False):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            x_data, y_data,
            alpha=0.3,
            c=df_valid["in_train"].map({True: "blue", False: "red"}),
            s=df_valid["length"] / 1.5,
        )
        if diagonal:
            ax.plot([0, 1], [0, 1], "r--", alpha=0.75, linewidth=2)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=900, bbox_inches="tight")
        logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out_path}")
        plt.close()

    # ── Figure 0a: Reference TM vs ProteinEBM energy ─────────────────────────
    if "min_energy_topk" in df.columns:
        valid_e = df.dropna(subset=[tms_col, "min_energy_topk", "in_train", "length"])
        if len(valid_e) > 0:
            _scatter_single(
                valid_e[tms_col], valid_e["min_energy_topk"], valid_e,
                f"Reference TM vs ProteinEBM energy (top-{int(top_k)} templates by min energy)",
                "Reference TM score", "ProteinEBM energy (lower is better)",
                os.path.join(output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_ref_tm_vs_energy_top{int(top_k)}.png"),
            )

    # ── Figure 0b: Reference TM vs AF2Rank pTM (per variant) ─────────────────
    for ptm_col, label, suffix in [
        ("max_ptm_topk", "model_1", ""),
        ("m2_max_ptm_topk", "model_2_ptm", "_model_2_ptm"),
        ("min_max_ptm_topk", "min(pTM_1, pTM_2)", "_min"),
    ]:
        if ptm_col in df.columns:
            valid_p = df.dropna(subset=[tms_col, ptm_col, "in_train", "length"])
            if len(valid_p) > 0:
                _scatter_single(
                    valid_p[tms_col], valid_p[ptm_col], valid_p,
                    f"Reference TM vs AF2Rank pTM ({label}, top-{int(top_k)})",
                    "Reference TM score", f"AF2Rank pTM ({label})",
                    os.path.join(output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_ref_tm_vs_ptm_top{int(top_k)}{suffix}.png"),
                )

    # ── Figure 1: reference TM vs prediction TM (top-1/top-5) ────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    valid_1 = df.dropna(subset=[tms_col, "top_1_tm_ref_pred", "in_train", "length"])
    if len(valid_1) > 0:
        ax1.scatter(
            valid_1[tms_col],
            valid_1["top_1_tm_ref_pred"],
            alpha=0.3,
            c=valid_1["in_train"].map({True: "blue", False: "red"}),
            s=valid_1["length"] / 1.5,
        )
        ax1.plot([0, 1], [0, 1], "r--", alpha=0.75, linewidth=2)
        ax1.set_xlabel("Reference TM Score", fontsize=12)
        ax1.set_ylabel("Top-1 Prediction TM (tm_ref_pred) by pTM", fontsize=12)
        ax1.set_title(f"Reference TM vs Top-1 prediction TM\n(AF2Rank on ProteinEBM top-{int(top_k)} templates; rank by pTM)", fontsize=13)
        ax1.grid(True, alpha=0.3)

    valid_2 = df.dropna(subset=[tms_col, "top_5_tm_ref_pred", "in_train", "length"])
    if len(valid_2) > 0:
        ax2.scatter(
            valid_2[tms_col],
            valid_2["top_5_tm_ref_pred"],
            alpha=0.3,
            c=valid_2["in_train"].map({True: "blue", False: "red"}),
            s=valid_2["length"] / 1.5,
        )
        ax2.plot([0, 1], [0, 1], "r--", alpha=0.75, linewidth=2)
        ax2.set_xlabel("Reference TM Score", fontsize=12)
        ax2.set_ylabel("Top-5 Prediction TM (best tm_ref_pred in top-5 by pTM)", fontsize=12)
        ax2.set_title(f"Reference TM vs Top-5 prediction TM\n(AF2Rank on ProteinEBM top-{int(top_k)} templates; rank by pTM)", fontsize=13)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_top_analysis_top{int(top_k)}.png")
    plt.savefig(out_path, dpi=900, bbox_inches="tight")
    logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out_path}")
    plt.close()

    # ── Helper: template TM vs prediction TM (2-panel) ───────────────────────
    def _plot_template_vs_prediction(df_plot, prefix, label, file_suffix):
        t1_tmpl = f"{prefix}_top_1_tm_ref_template" if prefix else "top_1_tm_ref_template"
        t1_pred = f"{prefix}_top_1_tm_ref_pred" if prefix else "top_1_tm_ref_pred"
        t5_tmpl = f"{prefix}_top_5_tm_ref_template" if prefix else "top_5_tm_ref_template"
        t5_pred = f"{prefix}_top_5_tm_ref_pred" if prefix else "top_5_tm_ref_pred"
        if t1_tmpl not in df_plot.columns or t1_pred not in df_plot.columns:
            return
        fig_tv, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 8))
        v1 = df_plot.dropna(subset=[t1_tmpl, t1_pred, "in_train", "length"])
        if len(v1) > 0:
            ax_l.scatter(v1[t1_tmpl], v1[t1_pred], alpha=0.3,
                         c=v1["in_train"].map({True: "blue", False: "red"}), s=v1["length"] / 1.5)
            ax_l.plot([0, 1], [0, 1], "r--", alpha=0.75, linewidth=2)
            ax_l.set_xlabel(f"Top-1 Template TM ({label})", fontsize=12)
            ax_l.set_ylabel(f"Top-1 Prediction TM ({label})", fontsize=12)
            ax_l.set_title(f"Template TM vs Prediction TM (Top-1, {label})\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)", fontsize=13)
            ax_l.grid(True, alpha=0.3)
        v5 = df_plot.dropna(subset=[t5_tmpl, t5_pred, "in_train", "length"])
        if len(v5) > 0:
            ax_r.scatter(v5[t5_tmpl], v5[t5_pred], alpha=0.3,
                         c=v5["in_train"].map({True: "blue", False: "red"}), s=v5["length"] / 1.5)
            ax_r.plot([0, 1], [0, 1], "r--", alpha=0.75, linewidth=2)
            ax_r.set_xlabel(f"Top-5 Template TM ({label})", fontsize=12)
            ax_r.set_ylabel(f"Top-5 Prediction TM ({label})", fontsize=12)
            ax_r.set_title(f"Template TM vs Prediction TM (Top-5, {label})\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)", fontsize=13)
            ax_r.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"cross_protein_af2rank_on_proteinebm_top_k_template_vs_prediction_top{int(top_k)}{file_suffix}.png"
        out = os.path.join(output_dir, fname)
        plt.savefig(out, dpi=900, bbox_inches="tight")
        logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out}")
        plt.close()

    # ── Figure 2: template TM vs prediction TM (all model variants) ──────────
    _plot_template_vs_prediction(df, "", "model_1", "")
    if "m2_top_1_tm_ref_template" in df.columns:
        _plot_template_vs_prediction(df, "m2", "model_2_ptm", "_model_2_ptm")
    if "min_top_1_tm_ref_template" in df.columns:
        _plot_template_vs_prediction(df, "min", "min", "_min")

    # ── Figure 3: TM_ref_pred vs pTM (does AF2 confidence correlate with prediction quality?)
    if "top_1_ptm" in df.columns and "top_5_ptm" in df.columns:
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(18, 8))

        valid_5 = df.dropna(subset=["top_1_ptm", "top_1_tm_ref_pred", "in_train", "length"])
        if len(valid_5) > 0:
            ax5.scatter(
                valid_5["top_1_ptm"],
                valid_5["top_1_tm_ref_pred"],
                alpha=0.3,
                c=valid_5["in_train"].map({True: "blue", False: "red"}),
                s=valid_5["length"] / 1.5,
            )
            ax5.set_xlabel("AF2 pTM (confidence)", fontsize=12)
            ax5.set_ylabel("TM(ref vs pred)", fontsize=12)
            ax5.set_title(f"Top-1 by pTM: Prediction TM vs AF2 pTM\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)", fontsize=13)
            ax5.grid(True, alpha=0.3)

        valid_6 = df.dropna(subset=["top_5_ptm", "top_5_tm_ref_pred", "in_train", "length"])
        if len(valid_6) > 0:
            ax6.scatter(
                valid_6["top_5_ptm"],
                valid_6["top_5_tm_ref_pred"],
                alpha=0.3,
                c=valid_6["in_train"].map({True: "blue", False: "red"}),
                s=valid_6["length"] / 1.5,
            )
            ax6.set_xlabel("AF2 pTM (confidence)", fontsize=12)
            ax6.set_ylabel("TM(ref vs pred)", fontsize=12)
            ax6.set_title(
                f"Top-5 by pTM (best tm_ref_pred): Prediction TM vs AF2 pTM\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)",
                fontsize=13,
            )
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path3 = os.path.join(
            output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_tm_ref_pred_vs_ptm_top{int(top_k)}.png"
        )
        plt.savefig(out_path3, dpi=900, bbox_inches="tight")
        logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out_path3}")
        plt.close()

    # ── Figure 4: TM_ref_pred vs composite score
    if "top_1_composite" in df.columns and "top_5_composite" in df.columns:
        fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(18, 8))

        valid_7 = df.dropna(subset=["top_1_composite", "top_1_tm_ref_pred", "in_train", "length"])
        if len(valid_7) > 0:
            ax7.scatter(
                valid_7["top_1_composite"],
                valid_7["top_1_tm_ref_pred"],
                alpha=0.3,
                c=valid_7["in_train"].map({True: "blue", False: "red"}),
                s=valid_7["length"] / 1.5,
            )
            ax7.set_xlabel("Composite score", fontsize=12)
            ax7.set_ylabel("TM(ref vs pred)", fontsize=12)
            ax7.set_title(f"Top-1 by pTM: Prediction TM vs composite score\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)", fontsize=13)
            ax7.grid(True, alpha=0.3)

        valid_8 = df.dropna(subset=["top_5_composite", "top_5_tm_ref_pred", "in_train", "length"])
        if len(valid_8) > 0:
            ax8.scatter(
                valid_8["top_5_composite"],
                valid_8["top_5_tm_ref_pred"],
                alpha=0.3,
                c=valid_8["in_train"].map({True: "blue", False: "red"}),
                s=valid_8["length"] / 1.5,
            )
            ax8.set_xlabel("Composite score", fontsize=12)
            ax8.set_ylabel("TM(ref vs pred)", fontsize=12)
            ax8.set_title(
                f"Top-5 by pTM (best tm_ref_pred): Prediction TM vs composite score\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)",
                fontsize=13,
            )
            ax8.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path4 = os.path.join(
            output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_tm_ref_pred_vs_composite_top{int(top_k)}.png"
        )
        plt.savefig(out_path4, dpi=900, bbox_inches="tight")
        logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out_path4}")
        plt.close()

    # ── Figure 5: TM_ref_pred vs pLDDT
    if "top_1_plddt" in df.columns and "top_5_plddt" in df.columns:
        fig5, (ax9, ax10) = plt.subplots(1, 2, figsize=(18, 8))

        valid_9 = df.dropna(subset=["top_1_plddt", "top_1_tm_ref_pred", "in_train", "length"])
        if len(valid_9) > 0:
            ax9.scatter(
                valid_9["top_1_plddt"],
                valid_9["top_1_tm_ref_pred"],
                alpha=0.3,
                c=valid_9["in_train"].map({True: "blue", False: "red"}),
                s=valid_9["length"] / 1.5,
            )
            ax9.set_xlabel("pLDDT", fontsize=12)
            ax9.set_ylabel("TM(ref vs pred)", fontsize=12)
            ax9.set_title(f"Top-1 by pTM: Prediction TM vs pLDDT\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)", fontsize=13)
            ax9.grid(True, alpha=0.3)

        valid_10 = df.dropna(subset=["top_5_plddt", "top_5_tm_ref_pred", "in_train", "length"])
        if len(valid_10) > 0:
            ax10.scatter(
                valid_10["top_5_plddt"],
                valid_10["top_5_tm_ref_pred"],
                alpha=0.3,
                c=valid_10["in_train"].map({True: "blue", False: "red"}),
                s=valid_10["length"] / 1.5,
            )
            ax10.set_xlabel("pLDDT", fontsize=12)
            ax10.set_ylabel("TM(ref vs pred)", fontsize=12)
            ax10.set_title(
                f"Top-5 by pTM (best tm_ref_pred): Prediction TM vs pLDDT\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)",
                fontsize=13,
            )
            ax10.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path5 = os.path.join(
            output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_tm_ref_pred_vs_plddt_top{int(top_k)}.png"
        )
        plt.savefig(out_path5, dpi=900, bbox_inches="tight")
        logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out_path5}")
        plt.close()

    # ── Per-variant plots (m2 / min) for confidence vs prediction quality ─────
    def _plot_variant(df_plot: pd.DataFrame, prefix: str, label: str, xlabel_override: dict = None) -> None:
        xlabel_override = xlabel_override or {}
        for suffix, x, y, top_num in [
            ("ptm", f"{prefix}_top_1_ptm", f"{prefix}_top_1_tm_ref_pred", "1"),
            ("ptm", f"{prefix}_top_5_ptm", f"{prefix}_top_5_tm_ref_pred", "5"),
            ("composite", f"{prefix}_top_1_composite", f"{prefix}_top_1_tm_ref_pred", "1"),
            ("composite", f"{prefix}_top_5_composite", f"{prefix}_top_5_tm_ref_pred", "5"),
            ("plddt", f"{prefix}_top_1_plddt", f"{prefix}_top_1_tm_ref_pred", "1"),
            ("plddt", f"{prefix}_top_5_plddt", f"{prefix}_top_5_tm_ref_pred", "5"),
        ]:
            if x not in df_plot.columns or y not in df_plot.columns:
                continue
            valid = df_plot.dropna(subset=[x, y, "in_train", "length"])
            if len(valid) == 0:
                continue
            xlab = xlabel_override.get(suffix) or ("AF2 pTM" if suffix == "ptm" else "Composite score" if suffix == "composite" else "pLDDT")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(
                valid[x],
                valid[y],
                alpha=0.3,
                c=valid["in_train"].map({True: "blue", False: "red"}),
                s=valid["length"] / 1.5,
            )
            ax.set_xlabel(xlab, fontsize=12)
            ax.set_ylabel("TM(ref vs pred)", fontsize=12)
            ax.set_title(f"Top-{top_num} by pTM ({label}): TM vs {suffix}\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)", fontsize=13)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(
                output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_tm_ref_pred_vs_{suffix}_top{top_num}_top{int(top_k)}_{label.replace(' ', '_')}.png"
            )
            plt.savefig(out_path, dpi=900, bbox_inches="tight")
            logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out_path}")
            plt.close()

    if "m2_top_1_ptm" in df.columns:
        _plot_variant(df, "m2", "model_2_ptm")
    if "min_top_1_ptm" in df.columns:
        _plot_variant(df, "min", "min", {"ptm": "min(pTM)", "composite": "min(composite)", "plddt": "min(pLDDT)"})


def load_summary_data(summary_files: List[str], dataset_file: str, id_col: str, tms_col: str) -> pd.DataFrame:
    """
    Load and compile summary data from all AF2Rank JSON files.
    
    Args:
        summary_files: List of paths to summary JSON files
        dataset_file: Name of the dataset file to analyze
        id_col: Column to use as the protein ID
        tms_col: Column to use as the TM score
    Returns:
        DataFrame with compiled summary data
    """
    data = []
    
    # Load dataset file
    dataset_df = pd.read_csv(dataset_file)
    dataset_proteins = set(dataset_df[id_col].tolist())
    
    logger.info(f"Dataset contains {len(dataset_proteins)} proteins")
    
    for summary_file in summary_files:
        # Extract protein ID from path: .../protein_id/af2rank_analysis/af2rank_summary_*.json
        protein_id = Path(summary_file).parent.parent.name
        
        # Skip if not in dataset
        if protein_id not in dataset_proteins:
            continue
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # Extract required metrics (protein_id already extracted from path)
        spearman_rho_composite = summary.get('spearman_correlation_rho_composite') or summary.get('spearman_correlation_rho')  # Backward compat
        spearman_rho_ptm = summary.get('spearman_correlation_rho_ptm')
        max_tm_ref_template = summary.get('max_tm_ref_template')
        max_tm_ref_pred = summary.get('max_tm_ref_pred')
        tm_ref_template_at_max_composite = summary.get('tm_ref_template_at_max_composite')
        tm_ref_pred_at_max_composite = summary.get('tm_ref_pred_at_max_composite')
        tm_ref_pred_at_max_ptm = summary.get('tm_ref_pred_at_max_ptm')
        top_1_tm_ref_template = summary.get('top_1_tm_ref_template')
        top_5_tm_ref_template = summary.get('top_5_tm_ref_template')
        top_1_tm_ref_pred = summary.get('top_1_tm_ref_pred')
        top_5_tm_ref_pred = summary.get('top_5_tm_ref_pred')

        # Only include if we have the required metrics
        if (
            spearman_rho_composite is not None
            and max_tm_ref_template is not None
            and tm_ref_template_at_max_composite is not None
        ):
            data.append({
                'protein_id': protein_id,
                'spearman_rho_composite': spearman_rho_composite,
                'spearman_rho_ptm': spearman_rho_ptm,
                'max_tm_ref_template': max_tm_ref_template,
                'max_tm_ref_pred': max_tm_ref_pred,
                'tm_ref_template_at_max_composite': tm_ref_template_at_max_composite,
                'tm_ref_pred_at_max_composite': tm_ref_pred_at_max_composite,
                'tm_ref_pred_at_max_ptm': tm_ref_pred_at_max_ptm,
                'top_1_tm_ref_template': top_1_tm_ref_template,
                'top_5_tm_ref_template': top_5_tm_ref_template,
                'top_1_tm_ref_pred': top_1_tm_ref_pred,
                'top_5_tm_ref_pred': top_5_tm_ref_pred,
            })
    
    df = pd.DataFrame(data)
    
    logger.info(f"Loaded {len(df)} proteins from AF2Rank results (filtered by dataset)")
    
    # Merge with dataset
    df = df.merge(dataset_df[[id_col, tms_col, 'in_train', 'length']], left_on='protein_id', right_on=id_col, how='left')
    df.drop(columns=[id_col], inplace=True)
    
    logger.info(f"Final dataset: {len(df)} proteins with complete metrics")
    return df


def load_proteinebm_data(score_files: List[str], dataset_file: str, id_col: str, tms_col: str) -> pd.DataFrame:
    """
    Load and compile per-protein energy statistics from ProteinEBM score CSVs.

    Each protein contributes a single row with summary energy statistics across its decoys.
    """
    data = []

    dataset_df = pd.read_csv(dataset_file)
    dataset_proteins = set(dataset_df[id_col].tolist())
    logger.info(f"Dataset contains {len(dataset_proteins)} proteins")

    for score_file in score_files:
        # Extract protein ID from path: .../protein_id/proteinebm_analysis/proteinebm_scores_*.csv
        protein_id = Path(score_file).parent.parent.name

        if protein_id not in dataset_proteins:
            continue

        scores_df = pd.read_csv(score_file)
        if "energy" not in scores_df.columns:
            raise KeyError(f"Missing required column 'energy' in {score_file}. Columns: {scores_df.columns.tolist()}")

        energies = scores_df["energy"].astype(float)
        n = int(len(energies))
        if n == 0:
            continue

        t_vals = scores_df["t"].unique().tolist() if "t" in scores_df.columns else []

        data.append({
            "protein_id": protein_id,
            "n_structures": n,
            "t_values": t_vals,
            "min_energy": float(energies.min()),
            "mean_energy": float(energies.mean()),
            "median_energy": float(energies.median()),
            "max_energy": float(energies.max()),
            "std_energy": float(energies.std()),
            "scores_csv": str(score_file),
        })

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} proteins from ProteinEBM results (filtered by dataset)")

    df = df.merge(dataset_df[[id_col, tms_col, "in_train", "length"]], left_on="protein_id", right_on=id_col, how="left")
    df.drop(columns=[id_col], inplace=True)

    logger.info(f"Final dataset: {len(df)} proteins with complete metrics")
    return df


def load_proteinebm_summary_data(summary_files: List[str], dataset_file: str, id_col: str, tms_col: str) -> pd.DataFrame:
    """
    Load and compile AF2Rank-style summary metrics from ProteinEBM summaries.
    This expects `proteinebm_scorer.py` to have populated the same keys used in AF2Rank plots:
      - spearman_correlation_rho_composite  (here: (-energy) vs tm_ref_template)
      - max_tm_ref_template
      - tm_ref_template_at_max_composite   (here: tm at min energy)
      - top_1_tm_ref_template / top_5_tm_ref_template
    """
    data = []

    dataset_df = pd.read_csv(dataset_file)
    dataset_proteins = set(dataset_df[id_col].tolist())
    logger.info(f"Dataset contains {len(dataset_proteins)} proteins")

    for summary_file in summary_files:
        protein_id = Path(summary_file).parent.parent.name
        if protein_id not in dataset_proteins:
            continue

        with open(summary_file, "r") as f:
            summary = json.load(f)

        spearman_rho = summary.get("spearman_correlation_rho_composite")
        max_tm_ref_template = summary.get("max_tm_ref_template")
        tm_ref_template_at_best = summary.get("tm_ref_template_at_max_composite")
        top_1_tm_ref_template = summary.get("top_1_tm_ref_template")
        top_5_tm_ref_template = summary.get("top_5_tm_ref_template")

        if spearman_rho is None or max_tm_ref_template is None or tm_ref_template_at_best is None:
            continue

        data.append(
            {
                "protein_id": protein_id,
                "spearman_rho_composite": spearman_rho,
                "max_tm_ref_template": max_tm_ref_template,
                "tm_ref_template_at_max_composite": tm_ref_template_at_best,
                "top_1_tm_ref_template": top_1_tm_ref_template,
                "top_5_tm_ref_template": top_5_tm_ref_template,
            }
        )

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} proteins from ProteinEBM summary files (filtered by dataset)")

    df = df.merge(dataset_df[[id_col, tms_col, "in_train", "length"]], left_on="protein_id", right_on=id_col, how="left")
    df.drop(columns=[id_col], inplace=True)
    return df


def create_proteinebm_plots(df: pd.DataFrame, output_dir: str, tms_col: str) -> None:
    """
    Create cross-protein ProteinEBM plots (energy statistics vs dataset metadata).
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 18))

    # Plot 1: Reference TM score vs min energy
    valid_1 = df.dropna(subset=[tms_col, "min_energy", "in_train", "length"])
    if len(valid_1) > 0:
        ax1.scatter(
            valid_1[tms_col],
            valid_1["min_energy"],
            alpha=0.3,
            c=valid_1["in_train"].map({True: "blue", False: "red"}),
            s=valid_1["length"] / 1.5,
        )
        ax1.set_xlabel("Reference TM Score", fontsize=12)
        ax1.set_ylabel("Min ProteinEBM Energy (lower is better)", fontsize=12)
        ax1.set_title("Reference TM Score vs. Best (Min) ProteinEBM Energy", fontsize=15)
        ax1.grid(True, alpha=0.3)

        if len(valid_1) > 2:
            corr = np.corrcoef(valid_1[tms_col], valid_1["min_energy"])[0, 1]
            ax1.text(
                0.05,
                0.95,
                f"R = {corr:.3f}",
                transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # Plot 2: Reference TM score vs mean energy
    valid_2 = df.dropna(subset=[tms_col, "mean_energy", "in_train", "length"])
    if len(valid_2) > 0:
        ax2.scatter(
            valid_2[tms_col],
            valid_2["mean_energy"],
            alpha=0.3,
            c=valid_2["in_train"].map({True: "blue", False: "red"}),
            s=valid_2["length"] / 1.5,
        )
        ax2.set_xlabel("Reference TM Score", fontsize=12)
        ax2.set_ylabel("Mean ProteinEBM Energy", fontsize=12)
        ax2.set_title("Reference TM Score vs. Mean ProteinEBM Energy", fontsize=15)
        ax2.grid(True, alpha=0.3)

        if len(valid_2) > 2:
            corr = np.corrcoef(valid_2[tms_col], valid_2["mean_energy"])[0, 1]
            ax2.text(
                0.05,
                0.95,
                f"R = {corr:.3f}",
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # Plot 3: Protein length vs min energy
    valid_3 = df.dropna(subset=["length", "min_energy", "in_train"])
    if len(valid_3) > 0:
        ax3.scatter(
            valid_3["length"],
            valid_3["min_energy"],
            alpha=0.3,
            c=valid_3["in_train"].map({True: "blue", False: "red"}),
            s=60,
        )
        ax3.set_xlabel("Protein length", fontsize=12)
        ax3.set_ylabel("Min ProteinEBM Energy", fontsize=12)
        ax3.set_title("Protein length vs. Best (Min) ProteinEBM Energy", fontsize=15)
        ax3.grid(True, alpha=0.3)

    # Plot 4: Min energy vs std energy
    valid_4 = df.dropna(subset=["min_energy", "std_energy", "in_train"])
    if len(valid_4) > 0:
        ax4.scatter(
            valid_4["min_energy"],
            valid_4["std_energy"],
            alpha=0.3,
            c=valid_4["in_train"].map({True: "blue", False: "red"}),
            s=60,
        )
        ax4.set_xlabel("Min ProteinEBM Energy", fontsize=12)
        ax4.set_ylabel("Std ProteinEBM Energy", fontsize=12)
        ax4.set_title("Energy dispersion vs. best energy (per protein)", fontsize=15)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cross_protein_proteinebm_energy_analysis.png")
    plt.savefig(plot_path, dpi=900, bbox_inches="tight")
    logger.info(f"Saved cross-protein ProteinEBM plot: {plot_path}")
    plt.close()


def save_proteinebm_statistics(df: pd.DataFrame, output_dir: str, tms_col: str) -> None:
    """
    Save ProteinEBM cross-protein summary dataset and aggregate statistics.
    """
    csv_path = os.path.join(output_dir, "cross_protein_summary_data.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary data: {csv_path}")

    stats = {
        "scorer": "proteinebm",
        "total_proteins": int(len(df)),
        "statistics": {
            "n_structures": {
                "mean": float(df["n_structures"].mean()),
                "median": float(df["n_structures"].median()),
                "min": float(df["n_structures"].min()),
                "max": float(df["n_structures"].max()),
            },
            "min_energy": {
                "mean": float(df["min_energy"].mean()),
                "median": float(df["min_energy"].median()),
                "std": float(df["min_energy"].std()),
                "min": float(df["min_energy"].min()),
                "max": float(df["min_energy"].max()),
            },
            "mean_energy": {
                "mean": float(df["mean_energy"].mean()),
                "median": float(df["mean_energy"].median()),
                "std": float(df["mean_energy"].std()),
                "min": float(df["mean_energy"].min()),
                "max": float(df["mean_energy"].max()),
            },
        },
    }

    if tms_col in df.columns:
        valid = df.dropna(subset=[tms_col, "min_energy", "mean_energy"])
        if len(valid) > 1:
            stats["correlations"] = {
                "reference_tms_vs_min_energy": float(np.corrcoef(valid[tms_col], valid["min_energy"])[0, 1]),
                "reference_tms_vs_mean_energy": float(np.corrcoef(valid[tms_col], valid["mean_energy"])[0, 1]),
            }

    stats_path = os.path.join(output_dir, "cross_protein_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved summary statistics: {stats_path}")


def create_summary_plots(df: pd.DataFrame, output_dir: str, tms_col: str, scorer: str) -> None:
    """
    Create cross-protein summary plots.
    
    Args:
        df: DataFrame with compiled summary data
        output_dir: Directory to save plots
        tms_col: Column to use as the TM score
    """
    os.makedirs(output_dir, exist_ok=True)

    if scorer == "af2rank":
        score_name = "AF2Rank score (pTM × pLDDT)"
        best_score_name = "Highest AF2Rank score"
        rho_label = "Spearman ρ (AF2Rank score vs template quality)"
        tm_at_best_label = "Template TM at highest AF2Rank score"
        max_tm_label = "Max template TM"
        top1_label = "Top-1 template TM by AF2Rank score"
        top5_label = "Top-5 template TM by AF2Rank score"
    else:
        score_name = "ProteinEBM score (-energy)"
        best_score_name = "Best ProteinEBM score (min energy)"
        rho_label = "Spearman ρ (ProteinEBM score vs template quality)"
        tm_at_best_label = "Template TM at best ProteinEBM score (min energy)"
        max_tm_label = "Max template TM"
        top1_label = "Top-1 template TM by ProteinEBM score"
        top5_label = "Top-5 template TM by ProteinEBM score"
    
    # Set up the plotting style
    plt.style.use('default')

    # ----------------------------
    # ProteinEBM: compact plot grids (no blank subplots)
    # ----------------------------
    if scorer != "af2rank":
        # Correlation figure: 2 panels
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        valid_data_1 = df.dropna(subset=['max_tm_ref_template', 'spearman_rho_composite'])
        if len(valid_data_1) > 0:
            ax1.scatter(valid_data_1['max_tm_ref_template'], valid_data_1['spearman_rho_composite'], alpha=0.7, s=60)
            ax1.set_xlabel(max_tm_label, fontsize=12)
            ax1.set_ylabel(rho_label, fontsize=12)
            ax1.set_title(f'{max_tm_label} vs rank correlation\n(How well does {score_name} rank template quality?)', fontsize=13)
            ax1.grid(True, alpha=0.3)

        valid_data_2 = df.dropna(subset=['tm_ref_template_at_max_composite', 'spearman_rho_composite'])
        if len(valid_data_2) > 0:
            ax2.scatter(valid_data_2['tm_ref_template_at_max_composite'], valid_data_2['spearman_rho_composite'], alpha=0.7, s=60)
            ax2.set_xlabel(tm_at_best_label, fontsize=12)
            ax2.set_ylabel(rho_label, fontsize=12)
            ax2.set_title(f'TM at {best_score_name} vs rank correlation\n(Does the top score correspond to a good template?)', fontsize=13)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path_1 = os.path.join(output_dir, f"cross_protein_{scorer}_correlation_analysis.png")
        plt.savefig(plot_path_1, dpi=900, bbox_inches='tight')
        logger.info(f"Saved cross-protein correlation plot: {plot_path_1}")
        plt.close()

        # Top figure: 4 panels
        fig2, ((ax5, ax6), (ax9, ax10)) = plt.subplots(2, 2, figsize=(18, 18))

        valid_data_5 = df.dropna(subset=['max_tm_ref_template', 'top_1_tm_ref_template'])
        if len(valid_data_5) > 0:
            in_train = valid_data_5[valid_data_5['in_train'] == True]
            not_in_train = valid_data_5[valid_data_5['in_train'] == False]
            if len(in_train) > 0:
                ax5.scatter(in_train['max_tm_ref_template'], in_train['top_1_tm_ref_template'], alpha=0.3, c='blue', s=in_train['length'] / 1.5, label='In Train')
            if len(not_in_train) > 0:
                ax5.scatter(not_in_train['max_tm_ref_template'], not_in_train['top_1_tm_ref_template'], alpha=0.3, c='red', s=not_in_train['length'] / 1.5, label='Not In Train')
            ax5.legend(fontsize=12, scatterpoints=1, loc='upper left')
            ax5.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=2)
            ax5.set_xlabel(max_tm_label, fontsize=12)
            ax5.set_ylabel(top1_label, fontsize=12)
            ax5.set_title(f'{max_tm_label} vs {top1_label}', fontsize=15)
            ax5.grid(True, alpha=0.3)

        valid_data_6 = df.dropna(subset=['max_tm_ref_template', 'top_5_tm_ref_template'])
        if len(valid_data_6) > 0:
            ax6.scatter(valid_data_6['max_tm_ref_template'], valid_data_6['top_5_tm_ref_template'], alpha=0.3, c=valid_data_6['in_train'].map({True: 'blue', False: 'red'}), s=valid_data_6['length'] / 1.5)
            ax6.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=2)
            ax6.set_xlabel(max_tm_label, fontsize=12)
            ax6.set_ylabel(top5_label, fontsize=12)
            ax6.set_title(f'{max_tm_label} vs {top5_label}', fontsize=15)
            ax6.grid(True, alpha=0.3)

        valid_data_9 = df.dropna(subset=[tms_col, 'top_1_tm_ref_template'])
        if len(valid_data_9) > 0:
            ax9.scatter(valid_data_9[tms_col], valid_data_9['top_1_tm_ref_template'], alpha=0.3, c=valid_data_9['in_train'].map({True: 'blue', False: 'red'}), s=valid_data_9['length'] / 1.5)
            ax9.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=2)
            ax9.set_xlabel('Reference TM Score', fontsize=12)
            ax9.set_ylabel(top1_label, fontsize=12)
            ax9.set_title(f'Reference TM vs {top1_label}', fontsize=15)
            ax9.grid(True, alpha=0.3)

        valid_data_10 = df.dropna(subset=[tms_col, 'top_5_tm_ref_template'])
        if len(valid_data_10) > 0:
            ax10.scatter(valid_data_10[tms_col], valid_data_10['top_5_tm_ref_template'], alpha=0.3, c=valid_data_10['in_train'].map({True: 'blue', False: 'red'}), s=valid_data_10['length'] / 1.5)
            ax10.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=2)
            ax10.set_xlabel('Reference TM Score', fontsize=12)
            ax10.set_ylabel(top5_label, fontsize=12)
            ax10.set_title(f'Reference TM vs {top5_label}', fontsize=15)
            ax10.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path_2 = os.path.join(output_dir, f"cross_protein_{scorer}_top_analysis.png")
        plt.savefig(plot_path_2, dpi=900, bbox_inches='tight')
        logger.info(f"Saved cross-protein top analysis plot: {plot_path_2}")
        plt.close()
        return

    # ----------------------------
    # AF2Rank: keep full grids (all panels used)
    # ----------------------------
    # Create first figure with four subplots (2x2 grid) for correlation analysis
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 18))

    # Plot 1: Max TM-score vs Spearman correlation (score vs template quality)
    valid_data_1 = df.dropna(subset=['max_tm_ref_template', 'spearman_rho_composite'])
    if len(valid_data_1) > 0:
        ax1.scatter(valid_data_1['max_tm_ref_template'], valid_data_1['spearman_rho_composite'], alpha=0.7, s=60)
        ax1.set_xlabel(max_tm_label, fontsize=12)
        ax1.set_ylabel(rho_label, fontsize=12)
        ax1.set_title(f'{max_tm_label} vs rank correlation\n(How well does {score_name} rank template quality?)', fontsize=13)
        ax1.grid(True, alpha=0.3)

    # Plot 2: Template quality at best score vs Spearman correlation (score)
    valid_data_2 = df.dropna(subset=['tm_ref_template_at_max_composite', 'spearman_rho_composite'])
    if len(valid_data_2) > 0:
        ax2.scatter(valid_data_2['tm_ref_template_at_max_composite'], valid_data_2['spearman_rho_composite'], alpha=0.7, s=60)
        ax2.set_xlabel(tm_at_best_label, fontsize=12)
        ax2.set_ylabel(rho_label, fontsize=12)
        ax2.set_title(f'TM at {best_score_name} vs rank correlation\n(Does the top score correspond to a good template?)', fontsize=13)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction quality at max pTM vs Spearman correlation (pTM)
    if "tm_ref_pred_at_max_ptm" in df.columns and "spearman_rho_ptm" in df.columns:
        valid_data_3 = df.dropna(subset=['tm_ref_pred_at_max_ptm', 'spearman_rho_ptm'])
        if len(valid_data_3) > 0:
            ax3.scatter(valid_data_3['tm_ref_pred_at_max_ptm'], valid_data_3['spearman_rho_ptm'], alpha=0.7, s=60)
            ax3.set_xlabel('Prediction TM Score at Highest pTM', fontsize=12)
            ax3.set_ylabel('Spearman ρ (pTM vs Prediction Quality)', fontsize=12)
            ax3.set_title('Prediction Quality of Best pTM vs. Correlation\n(Does pTM correlate with prediction quality?)', fontsize=13)
            ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison of two correlation types
    if "spearman_rho_ptm" in df.columns:
        valid_data_4 = df.dropna(subset=['spearman_rho_composite', 'spearman_rho_ptm'])
        if len(valid_data_4) > 0:
            ax4.scatter(valid_data_4['spearman_rho_composite'], valid_data_4['spearman_rho_ptm'], alpha=0.7, s=60)
            ax4.set_xlabel('Spearman ρ (AF2Rank score vs Template Quality)', fontsize=12)
            ax4.set_ylabel('Spearman ρ (pTM vs Prediction Quality)', fontsize=12)
            ax4.set_title('Comparison of AF2Rank Correlations\n(Template ranking vs Prediction quality)', fontsize=13)
            ax4.grid(True, alpha=0.3)
            ax4.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plot_path_1 = os.path.join(output_dir, f"cross_protein_{scorer}_correlation_analysis.png")
    plt.savefig(plot_path_1, dpi=900, bbox_inches='tight')
    logger.info(f"Saved cross-protein correlation plot: {plot_path_1}")
    plt.close()

    # Create second figure with eight subplots (2x4 grid) for top analysis
    fig2, ((ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(2, 4, figsize=(32, 18))

    # Plot 5: Max TM-score vs Top 1 Template TM-score By Composite
    valid_data_5 = df.dropna(subset=['max_tm_ref_template','top_1_tm_ref_template'])
    
    if len(valid_data_5) > 0:
        # Separate data by training status for proper legend
        in_train = valid_data_5[valid_data_5['in_train'] == True]
        not_in_train = valid_data_5[valid_data_5['in_train'] == False]
        
        if len(in_train) > 0:
            ax5.scatter(
                in_train['max_tm_ref_template'], 
                in_train['top_1_tm_ref_template'],
                alpha=0.3,
                c='blue',
                s=in_train['length'] / 1.5,
                label='In Train'
            )
        
        if len(not_in_train) > 0:
            ax5.scatter(
                not_in_train['max_tm_ref_template'], 
                not_in_train['top_1_tm_ref_template'],
                alpha=0.3,
                c='red',
                s=not_in_train['length'] / 1.5,
                label='Not In Train'
            )
        
        lgnd = ax5.legend(fontsize=14, scatterpoints=1, loc='upper left')
        if len(lgnd.legend_handles) >= 1:
            lgnd.legend_handles[0]._sizes = [40]  # In newer matplotlib, do lgnd.legend_handles[0]._legmarker.set_markersize(40)
        if len(lgnd.legend_handles) >= 2:
            lgnd.legend_handles[1]._sizes = [40]
        line5 = ax5.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax5.set_xlabel(max_tm_label, fontsize=12)
        ax5.set_ylabel(top1_label, fontsize=12)
        ax5.set_title(f'{max_tm_label} vs {top1_label}', fontsize=15)
        ax5.grid(True, alpha=0.3)

    # Plot 6: Max TM-score vs Top 5 Template TM-score By Composite
    valid_data_6 = df.dropna(subset=['max_tm_ref_template','top_5_tm_ref_template'])
    
    if len(valid_data_6) > 0:
        scatter6 = ax6.scatter(
            valid_data_6['max_tm_ref_template'], 
            valid_data_6['top_5_tm_ref_template'],
            alpha=0.3,
            c=valid_data_6['in_train'].map({True: 'blue', False: 'red'}),
            s=valid_data_6['length'] / 1.5
        )
        line6 = ax6.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax6.set_xlabel(max_tm_label, fontsize=12)
        ax6.set_ylabel(top5_label, fontsize=12)
        ax6.set_title(f'{max_tm_label} vs {top5_label}', fontsize=15)
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Max TM-score vs Top 1 Prediction TM-score By pTM (AF2Rank only)
    if scorer == "af2rank" and "max_tm_ref_pred" in df.columns and "top_1_tm_ref_pred" in df.columns:
        valid_data_7 = df.dropna(subset=['max_tm_ref_pred','top_1_tm_ref_pred'])
        if len(valid_data_7) > 0:
            ax7.scatter(
                valid_data_7['max_tm_ref_pred'],
                valid_data_7['top_1_tm_ref_pred'],
                alpha=0.3,
                c=valid_data_7['in_train'].map({True: 'blue', False: 'red'}),
                s=valid_data_7['length'] / 1.5
            )
            ax7.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
            ax7.set_xlabel('Max Prediction TM Score', fontsize=12)
            ax7.set_ylabel('Top 1 Prediction TM Score By pTM', fontsize=12)
            ax7.set_title('Max Prediction TM Score vs. Top 1 Prediction TM Score By pTM', fontsize=15)
            ax7.grid(True, alpha=0.3)
    
    # Plot 8: Max TM-score vs Top 5 Prediction TM-score By pTM (AF2Rank only)
    if scorer == "af2rank" and "max_tm_ref_pred" in df.columns and "top_5_tm_ref_pred" in df.columns:
        valid_data_8 = df.dropna(subset=['max_tm_ref_pred','top_5_tm_ref_pred'])
        if len(valid_data_8) > 0:
            ax8.scatter(
                valid_data_8['max_tm_ref_pred'],
                valid_data_8['top_5_tm_ref_pred'],
                alpha=0.3,
                c=valid_data_8['in_train'].map({True: 'blue', False: 'red'}),
                s=valid_data_8['length'] / 1.5
            )
            ax8.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
            ax8.set_xlabel('Max Prediction TM Score', fontsize=12)
            ax8.set_ylabel('Top 5 Prediction TM Score By pTM', fontsize=12)
            ax8.set_title('Max Prediction TM Score vs. Top 5 Prediction TM Score By pTM', fontsize=15)
            ax8.grid(True, alpha=0.3)
    
    # Plot 9: Reference TM score vs. Top 1 Template TM-score By Composite
    valid_data_9 = df.dropna(subset=[tms_col, 'top_1_tm_ref_template'])
    
    if len(valid_data_9) > 0:
        scatter9 = ax9.scatter(
            valid_data_9[tms_col], 
            valid_data_9['top_1_tm_ref_template'],
            alpha=0.3,
            c=valid_data_9['in_train'].map({True: 'blue', False: 'red'}),
            s=valid_data_9['length'] / 1.5
        )
        line9 = ax9.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax9.set_xlabel('Reference TM Score', fontsize=12)
        ax9.set_ylabel(top1_label, fontsize=12)
        ax9.set_title(f'Reference TM vs {top1_label}', fontsize=15)
        ax9.grid(True, alpha=0.3)
    
    # Plot 10: Reference TM score vs. Top 5 Template TM-score By Composite
    valid_data_10 = df.dropna(subset=[tms_col, 'top_5_tm_ref_template'])
    
    if len(valid_data_10) > 0:
        scatter10 = ax10.scatter(
            valid_data_10[tms_col], 
            valid_data_10['top_5_tm_ref_template'],
            alpha=0.3,
            c=valid_data_10['in_train'].map({True: 'blue', False: 'red'}),
            s=valid_data_10['length'] / 1.5
        )
        line10 = ax10.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax10.set_xlabel('Reference TM Score', fontsize=12)
        ax10.set_ylabel(top5_label, fontsize=12)
        ax10.set_title(f'Reference TM vs {top5_label}', fontsize=15)
        ax10.grid(True, alpha=0.3)
    
    # Plot 11: Reference TM score vs. Top 1 Prediction TM-score By pTM (AF2Rank only)
    if scorer == "af2rank" and "top_1_tm_ref_pred" in df.columns:
        valid_data_11 = df.dropna(subset=[tms_col, 'top_1_tm_ref_pred'])
        if len(valid_data_11) > 0:
            ax11.scatter(
                valid_data_11[tms_col],
                valid_data_11['top_1_tm_ref_pred'],
                alpha=0.3,
                c=valid_data_11['in_train'].map({True: 'blue', False: 'red'}),
                s=valid_data_11['length'] / 1.5
            )
            ax11.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
            ax11.set_xlabel('Reference TM Score', fontsize=12)
            ax11.set_ylabel('Top 1 Prediction TM Score By pTM', fontsize=12)
            ax11.set_title('Reference TM Score vs. Top 1 Prediction TM Score By pTM', fontsize=15)
            ax11.grid(True, alpha=0.3)
    
    # Plot 12: Reference TM score vs. Top 5 Prediction TM-score By pTM (AF2Rank only)
    if scorer == "af2rank" and "top_5_tm_ref_pred" in df.columns:
        valid_data_12 = df.dropna(subset=[tms_col, 'top_5_tm_ref_pred'])
        if len(valid_data_12) > 0:
            ax12.scatter(
                valid_data_12[tms_col],
                valid_data_12['top_5_tm_ref_pred'],
                alpha=0.3,
                c=valid_data_12['in_train'].map({True: 'blue', False: 'red'}),
                s=valid_data_12['length'] / 1.5
            )
            ax12.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
            ax12.set_xlabel('Reference TM Score', fontsize=12)
            ax12.set_ylabel('Top 5 Prediction TM Score By pTM', fontsize=12)
            ax12.set_title('Reference TM Score vs. Top 5 Prediction TM Score By pTM', fontsize=15)
            ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the second plot
    plot_path_2 = os.path.join(output_dir, f"cross_protein_{scorer}_top_analysis.png")
    plt.savefig(plot_path_2, dpi=900, bbox_inches='tight')
    logger.info(f"Saved cross-protein top analysis plot: {plot_path_2}")
    
    plt.close()


def save_summary_statistics(df: pd.DataFrame, output_dir: str, scorer: str) -> None:
    """
    Save summary statistics to CSV and JSON files.
    
    Args:
        df: DataFrame with compiled summary data
        output_dir: Directory to save files
    """
    # Save the full dataset
    csv_path = os.path.join(output_dir, "cross_protein_summary_data.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary data: {csv_path}")
    
    # Calculate and save summary statistics
    stats = {
        "scorer": scorer,
        "total_proteins": len(df),
        "statistics": {
            "spearman_correlation_rho_composite": {
                "mean": float(df['spearman_rho_composite'].mean()),
                "median": float(df['spearman_rho_composite'].median()),
                "std": float(df['spearman_rho_composite'].std()),
                "min": float(df['spearman_rho_composite'].min()),
                "max": float(df['spearman_rho_composite'].max())
            },
            **(
                {
                    "spearman_correlation_rho_ptm": {
                        "mean": float(df['spearman_rho_ptm'].mean()),
                        "median": float(df['spearman_rho_ptm'].median()),
                        "std": float(df['spearman_rho_ptm'].std()),
                        "min": float(df['spearman_rho_ptm'].min()),
                        "max": float(df['spearman_rho_ptm'].max())
                    }
                }
                if "spearman_rho_ptm" in df.columns
                else {}
            ),
            "max_tm_ref_template": {
                "mean": float(df['max_tm_ref_template'].mean()),
                "median": float(df['max_tm_ref_template'].median()),
                "std": float(df['max_tm_ref_template'].std()),
                "min": float(df['max_tm_ref_template'].min()),
                "max": float(df['max_tm_ref_template'].max())
            },
            "tm_ref_template_at_max_composite": {
                "mean": float(df['tm_ref_template_at_max_composite'].mean()),
                "median": float(df['tm_ref_template_at_max_composite'].median()),
                "std": float(df['tm_ref_template_at_max_composite'].std()),
                "min": float(df['tm_ref_template_at_max_composite'].min()),
                "max": float(df['tm_ref_template_at_max_composite'].max())
            },
            **(
                {
                    "tm_ref_pred_at_max_ptm": {
                        "mean": float(df['tm_ref_pred_at_max_ptm'].mean()),
                        "median": float(df['tm_ref_pred_at_max_ptm'].median()),
                        "std": float(df['tm_ref_pred_at_max_ptm'].std()),
                        "min": float(df['tm_ref_pred_at_max_ptm'].min()),
                        "max": float(df['tm_ref_pred_at_max_ptm'].max())
                    }
                }
                if "tm_ref_pred_at_max_ptm" in df.columns
                else {}
            )
        },
        "correlations": {
            "max_tm_vs_spearman_rho_composite": float(np.corrcoef(df['max_tm_ref_template'], df['spearman_rho_composite'])[0,1]),
            "tm_at_max_composite_vs_spearman_rho_composite": float(np.corrcoef(df['tm_ref_template_at_max_composite'], df['spearman_rho_composite'])[0,1]),
            **(
                {
                    "tm_at_max_ptm_vs_spearman_rho_ptm": float(np.corrcoef(df['tm_ref_pred_at_max_ptm'].dropna(), df['spearman_rho_ptm'].dropna())[0,1]),
                    "spearman_rho_composite_vs_spearman_rho_ptm": float(np.corrcoef(df['spearman_rho_composite'].dropna(), df['spearman_rho_ptm'].dropna())[0,1]),
                }
                if "tm_ref_pred_at_max_ptm" in df.columns and "spearman_rho_ptm" in df.columns
                else {}
            )
        }
    }
    
    stats_path = os.path.join(output_dir, "cross_protein_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved summary statistics: {stats_path}")


def load_diversity_summary_data(inference_dir: str, protein_ids: List[str] | None = None, subdir: str = "proteina_diversity") -> pd.DataFrame:
    """
    Load diversity summaries from per-protein directories and return a DataFrame.

    Columns: protein_id, mean_tem_to_tem_tm, std_tem_to_tem_tm, median_tem_to_tem_tm,
             min_tem_to_tem_tm, max_tem_to_tem_tm, n_samples, n_pairs
    """
    from proteinfoundation.af2rank_evaluation.proteina_analysis import find_analysis_summaries
    from proteinfoundation.af2rank_evaluation.proteina_diversity import find_diversity_summaries

    summary_files = find_diversity_summaries(inference_dir, subdir)
    summary_files += find_analysis_summaries(inference_dir, "proteina_analysis")
    data_by_protein = {}
    for path in summary_files:
        try:
            with open(path, "r") as f:
                s = json.load(f)
            pid = s.get("protein_id") or Path(path).parent.parent.name
            if protein_ids is not None and pid not in protein_ids:
                continue
            data_by_protein[pid] = {
                "protein_id": pid,
                "mean_tem_to_tem_tm": s.get("mean_tem_to_tem_tm"),
                "std_tem_to_tem_tm": s.get("std_tem_to_tem_tm"),
                "median_tem_to_tem_tm": s.get("median_tem_to_tem_tm"),
                "min_tem_to_tem_tm": s.get("min_tem_to_tem_tm"),
                "max_tem_to_tem_tm": s.get("max_tem_to_tem_tm"),
                "n_samples": s.get("n_samples"),
                "n_pairs": s.get("n_pairs"),
            }
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load diversity summary {path}: {e}")

    df = pd.DataFrame(list(data_by_protein.values()))
    logger.info(f"Loaded diversity data for {len(df)} proteins")
    return df


def create_diversity_vs_quality_plot(
    df: pd.DataFrame,
    output_dir: str,
    top1_tm_col: str,
    top1_tm_label: str = "Top-1 Template TM (by scorer)",
) -> None:
    """
    Scatter plot: median_tem_to_tem_tm (X) vs top-1 template TM by scorer (Y).
    """
    required = ["median_tem_to_tem_tm", top1_tm_col]
    opt_cols = ["in_train", "length"]
    drop_cols = [c for c in required + opt_cols if c in df.columns]
    valid = df.dropna(subset=[c for c in required if c in df.columns])
    if len(valid) == 0:
        logger.warning("No valid data for diversity-vs-quality plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    if "in_train" in valid.columns:
        colors = valid["in_train"].map({True: "blue", False: "red"})
    else:
        colors = "#4C72B0"
    sizes = valid["length"] / 1.5 if "length" in valid.columns else 20

    ax.scatter(
        valid["median_tem_to_tem_tm"],
        valid[top1_tm_col],
        alpha=0.3,
        c=colors,
        s=sizes,
    )
    ax.set_xlabel("Median Sample-to-Sample TM (diversity)", fontsize=12)
    ax.set_ylabel(top1_tm_label, fontsize=12)
    ax.set_title("Sample Diversity vs. Top-1 Quality", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "cross_protein_diversity_vs_top1_tm.png")
    plt.savefig(out_path, dpi=900, bbox_inches="tight")
    logger.info(f"Saved diversity-vs-quality plot: {out_path}")
    plt.close()


def add_diversity_to_summary_statistics(stats: dict, df: pd.DataFrame) -> dict:
    """Add diversity statistics to an existing summary statistics dict."""
    div_cols = {
        "mean_tem_to_tem_tm": "mean_tem_to_tem_tm",
        "std_tem_to_tem_tm": "std_tem_to_tem_tm",
        "median_tem_to_tem_tm": "median_tem_to_tem_tm",
    }
    div_stats = {}
    for label, col in div_cols.items():
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                div_stats[label] = {
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
    if div_stats:
        stats["diversity_statistics"] = div_stats
        stats["diversity_statistics"]["num_proteins_with_diversity"] = int(df["median_tem_to_tem_tm"].notna().sum())
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate cross-protein scoring analysis plots')
    parser.add_argument('--inference_dir', required=True, 
                       help='Base inference directory containing protein chain results')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for plots and summary files')
    parser.add_argument(
        '--scorer',
        choices=['af2rank', 'proteinebm', 'af2rank_on_proteinebm_topk'],
        default='af2rank',
        help='Which scorer outputs to aggregate',
    )
    parser.add_argument(
        '--af2rank_top_k',
        type=int,
        default=0,
        help='When --scorer=af2rank_on_proteinebm_topk, use this K cutoff when ranking by pTM (folder is always af2rank_on_proteinebm_top_k).',
    )
    parser.add_argument(
        '--proteinebm_plot_mode',
        choices=['tm', 'energy'],
        default='tm',
        help='When --scorer=proteinebm, choose whether to plot TM-based (summary) metrics or energy-based diagnostics',
    )
    parser.add_argument(
        '--proteinebm_analysis_subdir',
        default='proteinebm_v2_cathmd_analysis',
        help='Per-protein subdir containing ProteinEBM outputs (default: proteinebm_v2_cathmd_analysis)',
    )
    parser.add_argument('--dataset_file', required=True,
                       help='Name of the dataset file to analyze')
    parser.add_argument('--id_col', default='natives_rcsb',
                       help='Column name to use as protein ID (default: natives_rcsb)')
    parser.add_argument('--tms_col', default='tms_single',
                       help='Column name to use as TM score (default: tms_single)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    logger.info(f"🔬 Starting cross-protein analysis (scorer={args.scorer})")
    logger.info(f"📂 Inference directory: {args.inference_dir}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"📄 Dataset file: {args.dataset_file}")
    logger.info(f"🔑 ID column: {args.id_col}")
    logger.info(f"🔑 TM score column: {args.tms_col}")

    # Validate input directory
    if not os.path.exists(args.inference_dir):
        logger.error(f"Inference directory not found: {args.inference_dir}")
        sys.exit(1)
    
    # Validate dataset file
    if not os.path.exists(args.dataset_file):
        logger.error(f"Dataset file not found: {args.dataset_file}")
        sys.exit(1)
    
    if args.scorer == "af2rank":
        # Find all AF2Rank summary files
        summary_files = find_af2rank_summaries(args.inference_dir)
        if not summary_files:
            logger.error("No AF2Rank summary files found")
            sys.exit(1)

        # Load and compile data
        df = load_summary_data(summary_files, args.dataset_file, args.id_col, args.tms_col)
    elif args.scorer == "af2rank_on_proteinebm_topk":
        if int(args.af2rank_top_k) <= 0:
            logger.error("--af2rank_top_k must be > 0 when --scorer=af2rank_on_proteinebm_topk")
            sys.exit(1)
        score_files = find_af2rank_on_proteinebm_topk_score_files(args.inference_dir, int(args.af2rank_top_k))
        if not score_files:
            logger.error("No AF2Rank-on-ProteinEBM-topk score files found")
            sys.exit(1)
        df = load_af2rank_on_proteinebm_topk_data(
            score_files,
            args.dataset_file,
            args.id_col,
            args.tms_col,
            top_k=int(args.af2rank_top_k),
            proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
        )
    else:
        if args.proteinebm_plot_mode == "tm":
            summary_files = find_proteinebm_summaries(args.inference_dir, args.proteinebm_analysis_subdir)
            if not summary_files:
                logger.error("No ProteinEBM summary files found")
                sys.exit(1)
            df = load_proteinebm_summary_data(summary_files, args.dataset_file, args.id_col, args.tms_col)
        else:
            score_files = find_proteinebm_score_files(args.inference_dir, args.proteinebm_analysis_subdir)
            if not score_files:
                logger.error("No ProteinEBM score files found")
                sys.exit(1)
            df = load_proteinebm_data(score_files, args.dataset_file, args.id_col, args.tms_col)
    
    if len(df) == 0:
        logger.error("No valid data found in scoring outputs")
        sys.exit(1)
    
    logger.info(f"📊 Analyzing {len(df)} proteins")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.scorer == "af2rank":
        # Generate plots
        create_summary_plots(df, args.output_dir, args.tms_col, scorer=args.scorer)
        # Save summary statistics
        save_summary_statistics(df, args.output_dir, scorer=args.scorer)
    elif args.scorer == "af2rank_on_proteinebm_topk":
        create_af2rank_on_proteinebm_topk_plots(df, args.output_dir, args.tms_col, int(args.af2rank_top_k))
        # Save the full dataset for debugging/inspection.
        csv_path = os.path.join(args.output_dir, "cross_protein_summary_data.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary data: {csv_path}")
    else:
        if args.proteinebm_plot_mode == "tm":
            # TM-based (AF2Rank-style) summaries.
            create_summary_plots(df, args.output_dir, args.tms_col, scorer=args.scorer)
            save_summary_statistics(df, args.output_dir, scorer=args.scorer)
        else:
            # Energy-based diagnostic plots.
            create_proteinebm_plots(df, args.output_dir, args.tms_col)
            save_proteinebm_statistics(df, args.output_dir, args.tms_col)

    # ── Diversity-vs-quality cross-protein plot ───────────────────────────────
    protein_ids_in_df = set(df["protein_id"].tolist()) if "protein_id" in df.columns else None
    div_df = load_diversity_summary_data(args.inference_dir, protein_ids=protein_ids_in_df)
    if len(div_df) > 0:
        df = df.merge(div_df[["protein_id", "mean_tem_to_tem_tm", "std_tem_to_tem_tm", "median_tem_to_tem_tm"]],
                       on="protein_id", how="left")

        # Determine the top-1 TM column based on scorer
        top1_col = None
        top1_label = "Top-1 Template TM (by scorer)"
        if args.scorer == "af2rank":
            top1_col = "top_1_tm_ref_template"
            top1_label = "Top-1 Template TM (AF2Rank, by composite)"
        elif args.scorer == "proteinebm":
            top1_col = "top_1_tm_ref_template"
            top1_label = "Top-1 Template TM (ProteinEBM, by energy)"
        elif args.scorer == "af2rank_on_proteinebm_topk":
            # Use min(pTM_1, pTM_2) top-1 prediction TM if available, else m1
            top1_col = "min_top_1_tm_ref_pred" if "min_top_1_tm_ref_pred" in df.columns else "top_1_tm_ref_pred"
            top1_label = "Top-1 Prediction TM (AF2Rank on ProteinEBM top-k)"

        if top1_col and top1_col in df.columns:
            create_diversity_vs_quality_plot(df, args.output_dir, top1_col, top1_label)
        else:
            logger.warning(f"Top-1 TM column '{top1_col}' not found in data, skipping diversity-vs-quality plot")

        # Update summary statistics with diversity info
        stats_path = os.path.join(args.output_dir, "cross_protein_statistics.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            stats = add_diversity_to_summary_statistics(stats, df)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Updated statistics with diversity metrics: {stats_path}")
    else:
        logger.info("No diversity data found, skipping diversity-vs-quality plot")

    logger.info("✅ Cross-protein analysis completed successfully!")
    logger.info(f"📈 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
