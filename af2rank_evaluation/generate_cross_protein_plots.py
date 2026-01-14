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


def find_proteinebm_score_files(inference_base_dir: str) -> List[str]:
    """
    Find all ProteinEBM per-protein score CSVs in the inference directory structure.

    Args:
        inference_base_dir: Base inference directory

    Returns:
        List of paths to ProteinEBM score CSV files
    """
    pattern = os.path.join(inference_base_dir, "*", "proteinebm_analysis", "proteinebm_scores_*.csv")
    score_files = glob.glob(pattern)
    logger.info(f"Found {len(score_files)} ProteinEBM score files")
    return score_files


def find_proteinebm_summaries(inference_base_dir: str) -> List[str]:
    """
    Find all ProteinEBM per-protein summary JSON files in the inference directory structure.
    """
    pattern = os.path.join(inference_base_dir, "*", "proteinebm_analysis", "proteinebm_summary_*.json")
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
    id_column: str,
    tms_column: str,
    top_k: int,
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

    If fewer than 5 entries exist, we use whatever is present.
    """
    data = []

    dataset_df = pd.read_csv(dataset_file)
    dataset_proteins = set(dataset_df[id_column].tolist())
    logger.info(f"Dataset contains {len(dataset_proteins)} proteins")

    for score_file in score_files:
        protein_id = Path(score_file).parent.parent.parent.name
        if protein_id not in dataset_proteins:
            continue

        df = pd.read_csv(score_file)
        needed = {"ptm", "tm_ref_pred", "tm_ref_template"}
        missing = sorted([c for c in needed if c not in df.columns])
        if missing:
            raise KeyError(f"Missing columns {missing} in {score_file}. Columns: {df.columns.tolist()}")

        df = df.dropna(subset=["ptm", "tm_ref_pred", "tm_ref_template"]).copy()
        if len(df) == 0:
            continue

        df["ptm"] = pd.to_numeric(df["ptm"], errors="coerce")
        df["tm_ref_pred"] = pd.to_numeric(df["tm_ref_pred"], errors="coerce")
        df["tm_ref_template"] = pd.to_numeric(df["tm_ref_template"], errors="coerce")
        df = (
            df.dropna(subset=["ptm", "tm_ref_pred", "tm_ref_template"])
            .sort_values("ptm", ascending=False)
            .reset_index(drop=True)
        )
        if len(df) == 0:
            continue

        k_eff_total = int(min(int(top_k), int(len(df))))
        if k_eff_total <= 0:
            continue
        df = df.iloc[:k_eff_total].reset_index(drop=True)

        n_scored = int(len(df))
        top_1_tm_ref_pred = float(df.iloc[0]["tm_ref_pred"])
        top_1_tm_ref_template = float(df.iloc[0]["tm_ref_template"])
        k_eff_5 = int(min(5, n_scored))
        df_top5 = df.iloc[:k_eff_5].reset_index(drop=True)
        best_idx = int(np.argmax(df_top5["tm_ref_pred"].to_numpy()))
        top_5_tm_ref_pred = float(df_top5.iloc[best_idx]["tm_ref_pred"])
        top_5_tm_ref_template = float(df_top5.iloc[best_idx]["tm_ref_template"])

        data.append(
            {
                "protein_id": protein_id,
                "n_scored": n_scored,
                "top_1_tm_ref_pred": top_1_tm_ref_pred,
                "top_5_tm_ref_pred": top_5_tm_ref_pred,
                "top_1_tm_ref_template": top_1_tm_ref_template,
                "top_5_tm_ref_template": top_5_tm_ref_template,
                "scores_csv": str(score_file),
            }
        )

    if len(data) == 0:
        # Return an empty frame with the expected schema so callers can handle it cleanly.
        return pd.DataFrame(
            columns=[
                "protein_id",
                "n_scored",
                "top_1_tm_ref_pred",
                "top_5_tm_ref_pred",
                "top_1_tm_ref_template",
                "top_5_tm_ref_template",
                "scores_csv",
                tms_column,
                "in_train",
                "length",
            ]
        )

    out = pd.DataFrame(data)
    logger.info(f"Loaded {len(out)} proteins from AF2Rank-on-ProteinEBM-topk results (filtered by dataset)")

    out = out.merge(
        dataset_df[[id_column, tms_column, "in_train", "length"]],
        left_on="protein_id",
        right_on=id_column,
        how="left",
    )
    out.drop(columns=[id_column], inplace=True)
    return out


def create_af2rank_on_proteinebm_topk_plots(df: pd.DataFrame, output_dir: str, tms_column: str, top_k: int) -> None:
    """
    Create cross-protein plots for AF2Rank-on-ProteinEBM-topk:
      - Reference TM vs Top-1 tm_ref_pred (rank by pTM)
      - Reference TM vs Top-5 tm_ref_pred (rank by pTM; uses min(5, n_scored))
      - Template TM vs Prediction TM for Top-1 (same entry as top_1_tm_ref_pred)
      - Template TM vs Prediction TM for Top-5 (entry achieving best tm_ref_pred among top-5 by pTM)
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    # Figure 1: reference TM vs prediction TM (top-1/top-5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    valid_1 = df.dropna(subset=[tms_column, "top_1_tm_ref_pred", "in_train", "length"])
    if len(valid_1) > 0:
        ax1.scatter(
            valid_1[tms_column],
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

    valid_2 = df.dropna(subset=[tms_column, "top_5_tm_ref_pred", "in_train", "length"])
    if len(valid_2) > 0:
        ax2.scatter(
            valid_2[tms_column],
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

    # Figure 2: template TM vs prediction TM (top-1/top-5)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 8))

    valid_3 = df.dropna(subset=["top_1_tm_ref_template", "top_1_tm_ref_pred", "in_train", "length"])
    if len(valid_3) > 0:
        ax3.scatter(
            valid_3["top_1_tm_ref_template"],
            valid_3["top_1_tm_ref_pred"],
            alpha=0.3,
            c=valid_3["in_train"].map({True: "blue", False: "red"}),
            s=valid_3["length"] / 1.5,
        )
        ax3.plot([0, 1], [0, 1], "r--", alpha=0.75, linewidth=2)
        ax3.set_xlabel("Top-1 Template TM (tm_ref_template) by pTM", fontsize=12)
        ax3.set_ylabel("Top-1 Prediction TM (tm_ref_pred) by pTM", fontsize=12)
        ax3.set_title(
            f"Template TM vs Prediction TM (Top-1 by pTM)\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)",
            fontsize=13,
        )
        ax3.grid(True, alpha=0.3)

    valid_4 = df.dropna(subset=["top_5_tm_ref_template", "top_5_tm_ref_pred", "in_train", "length"])
    if len(valid_4) > 0:
        ax4.scatter(
            valid_4["top_5_tm_ref_template"],
            valid_4["top_5_tm_ref_pred"],
            alpha=0.3,
            c=valid_4["in_train"].map({True: "blue", False: "red"}),
            s=valid_4["length"] / 1.5,
        )
        ax4.plot([0, 1], [0, 1], "r--", alpha=0.75, linewidth=2)
        ax4.set_xlabel("Top-5 Template TM (tm_ref_template for best tm_ref_pred in top-5 by pTM)", fontsize=12)
        ax4.set_ylabel("Top-5 Prediction TM (best tm_ref_pred in top-5 by pTM)", fontsize=12)
        ax4.set_title(
            f"Template TM vs Prediction TM (Top-5 by pTM)\n(AF2Rank on ProteinEBM top-{int(top_k)} templates)",
            fontsize=13,
        )
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(
        output_dir, f"cross_protein_af2rank_on_proteinebm_top_k_template_vs_prediction_top{int(top_k)}.png"
    )
    plt.savefig(out_path2, dpi=900, bbox_inches="tight")
    logger.info(f"Saved AF2Rank-on-ProteinEBM-topk plot: {out_path2}")
    plt.close()


def load_summary_data(summary_files: List[str], dataset_file: str, id_column: str, tms_column: str) -> pd.DataFrame:
    """
    Load and compile summary data from all AF2Rank JSON files.
    
    Args:
        summary_files: List of paths to summary JSON files
        dataset_file: Name of the dataset file to analyze
        id_column: Column to use as the protein ID
        tms_column: Column to use as the TM score
    Returns:
        DataFrame with compiled summary data
    """
    data = []
    
    # Load dataset file
    dataset_df = pd.read_csv(dataset_file)
    dataset_proteins = set(dataset_df[id_column].tolist())
    
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
    df = df.merge(dataset_df[[id_column, tms_column, 'in_train', 'length']], left_on='protein_id', right_on=id_column, how='left')
    df.drop(columns=[id_column], inplace=True)
    
    logger.info(f"Final dataset: {len(df)} proteins with complete metrics")
    return df


def load_proteinebm_data(score_files: List[str], dataset_file: str, id_column: str, tms_column: str) -> pd.DataFrame:
    """
    Load and compile per-protein energy statistics from ProteinEBM score CSVs.

    Each protein contributes a single row with summary energy statistics across its decoys.
    """
    data = []

    dataset_df = pd.read_csv(dataset_file)
    dataset_proteins = set(dataset_df[id_column].tolist())
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

    df = df.merge(dataset_df[[id_column, tms_column, "in_train", "length"]], left_on="protein_id", right_on=id_column, how="left")
    df.drop(columns=[id_column], inplace=True)

    logger.info(f"Final dataset: {len(df)} proteins with complete metrics")
    return df


def load_proteinebm_summary_data(summary_files: List[str], dataset_file: str, id_column: str, tms_column: str) -> pd.DataFrame:
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
    dataset_proteins = set(dataset_df[id_column].tolist())
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

    df = df.merge(dataset_df[[id_column, tms_column, "in_train", "length"]], left_on="protein_id", right_on=id_column, how="left")
    df.drop(columns=[id_column], inplace=True)
    return df


def create_proteinebm_plots(df: pd.DataFrame, output_dir: str, tms_column: str) -> None:
    """
    Create cross-protein ProteinEBM plots (energy statistics vs dataset metadata).
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 18))

    # Plot 1: Reference TM score vs min energy
    valid_1 = df.dropna(subset=[tms_column, "min_energy", "in_train", "length"])
    if len(valid_1) > 0:
        ax1.scatter(
            valid_1[tms_column],
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
            corr = np.corrcoef(valid_1[tms_column], valid_1["min_energy"])[0, 1]
            ax1.text(
                0.05,
                0.95,
                f"R = {corr:.3f}",
                transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # Plot 2: Reference TM score vs mean energy
    valid_2 = df.dropna(subset=[tms_column, "mean_energy", "in_train", "length"])
    if len(valid_2) > 0:
        ax2.scatter(
            valid_2[tms_column],
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
            corr = np.corrcoef(valid_2[tms_column], valid_2["mean_energy"])[0, 1]
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


def save_proteinebm_statistics(df: pd.DataFrame, output_dir: str, tms_column: str) -> None:
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

    if tms_column in df.columns:
        valid = df.dropna(subset=[tms_column, "min_energy", "mean_energy"])
        if len(valid) > 1:
            stats["correlations"] = {
                "reference_tms_vs_min_energy": float(np.corrcoef(valid[tms_column], valid["min_energy"])[0, 1]),
                "reference_tms_vs_mean_energy": float(np.corrcoef(valid[tms_column], valid["mean_energy"])[0, 1]),
            }

    stats_path = os.path.join(output_dir, "cross_protein_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved summary statistics: {stats_path}")


def create_summary_plots(df: pd.DataFrame, output_dir: str, tms_column: str, scorer: str) -> None:
    """
    Create cross-protein summary plots.
    
    Args:
        df: DataFrame with compiled summary data
        output_dir: Directory to save plots
        tms_column: Column to use as the TM score
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

        valid_data_9 = df.dropna(subset=[tms_column, 'top_1_tm_ref_template'])
        if len(valid_data_9) > 0:
            ax9.scatter(valid_data_9[tms_column], valid_data_9['top_1_tm_ref_template'], alpha=0.3, c=valid_data_9['in_train'].map({True: 'blue', False: 'red'}), s=valid_data_9['length'] / 1.5)
            ax9.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=2)
            ax9.set_xlabel('Reference TM Score', fontsize=12)
            ax9.set_ylabel(top1_label, fontsize=12)
            ax9.set_title(f'Reference TM vs {top1_label}', fontsize=15)
            ax9.grid(True, alpha=0.3)

        valid_data_10 = df.dropna(subset=[tms_column, 'top_5_tm_ref_template'])
        if len(valid_data_10) > 0:
            ax10.scatter(valid_data_10[tms_column], valid_data_10['top_5_tm_ref_template'], alpha=0.3, c=valid_data_10['in_train'].map({True: 'blue', False: 'red'}), s=valid_data_10['length'] / 1.5)
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
    valid_data_9 = df.dropna(subset=[tms_column, 'top_1_tm_ref_template'])
    
    if len(valid_data_9) > 0:
        scatter9 = ax9.scatter(
            valid_data_9[tms_column], 
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
    valid_data_10 = df.dropna(subset=[tms_column, 'top_5_tm_ref_template'])
    
    if len(valid_data_10) > 0:
        scatter10 = ax10.scatter(
            valid_data_10[tms_column], 
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
        valid_data_11 = df.dropna(subset=[tms_column, 'top_1_tm_ref_pred'])
        if len(valid_data_11) > 0:
            ax11.scatter(
                valid_data_11[tms_column],
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
        valid_data_12 = df.dropna(subset=[tms_column, 'top_5_tm_ref_pred'])
        if len(valid_data_12) > 0:
            ax12.scatter(
                valid_data_12[tms_column],
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
    parser.add_argument('--dataset_file', required=True,
                       help='Name of the dataset file to analyze')
    parser.add_argument('--id_column', default='natives_rcsb',
                       help='Column name to use as protein ID (default: natives_rcsb)')
    parser.add_argument('--tms_column', default='tms_single',
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
    logger.info(f"🔑 ID column: {args.id_column}")
    logger.info(f"🔑 TM score column: {args.tms_column}")

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
        df = load_summary_data(summary_files, args.dataset_file, args.id_column, args.tms_column)
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
            args.id_column,
            args.tms_column,
            top_k=int(args.af2rank_top_k),
        )
    else:
        if args.proteinebm_plot_mode == "tm":
            summary_files = find_proteinebm_summaries(args.inference_dir)
            if not summary_files:
                logger.error("No ProteinEBM summary files found")
                sys.exit(1)
            df = load_proteinebm_summary_data(summary_files, args.dataset_file, args.id_column, args.tms_column)
        else:
            score_files = find_proteinebm_score_files(args.inference_dir)
            if not score_files:
                logger.error("No ProteinEBM score files found")
                sys.exit(1)
            df = load_proteinebm_data(score_files, args.dataset_file, args.id_column, args.tms_column)
    
    if len(df) == 0:
        logger.error("No valid data found in scoring outputs")
        sys.exit(1)
    
    logger.info(f"📊 Analyzing {len(df)} proteins")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.scorer == "af2rank":
        # Generate plots
        create_summary_plots(df, args.output_dir, args.tms_column, scorer=args.scorer)
        # Save summary statistics
        save_summary_statistics(df, args.output_dir, scorer=args.scorer)
    elif args.scorer == "af2rank_on_proteinebm_topk":
        create_af2rank_on_proteinebm_topk_plots(df, args.output_dir, args.tms_column, int(args.af2rank_top_k))
        # Save the full dataset for debugging/inspection.
        csv_path = os.path.join(args.output_dir, "cross_protein_summary_data.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary data: {csv_path}")
    else:
        if args.proteinebm_plot_mode == "tm":
            # TM-based (AF2Rank-style) summaries.
            create_summary_plots(df, args.output_dir, args.tms_column, scorer=args.scorer)
            save_summary_statistics(df, args.output_dir, scorer=args.scorer)
        else:
            # Energy-based diagnostic plots.
            create_proteinebm_plots(df, args.output_dir, args.tms_column)
            save_proteinebm_statistics(df, args.output_dir, args.tms_column)
    
    logger.info("✅ Cross-protein analysis completed successfully!")
    logger.info(f"📈 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
