#!/usr/bin/env python3
"""
Backfill ProteinEBM per-protein plots for previously-scored proteins.

This is intentionally *non-scoring*: it reads existing
  proteinebm_analysis/proteinebm_scores_<protein>.csv
and writes PNG plots into the same folder, plus (optionally) updates
  proteinebm_analysis/proteinebm_summary_<protein>.json
to include a "plots" field pointing to those PNGs.

This exists because the scoring pipeline uses --filter_existing and will not
re-run proteins that already have score CSVs.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _rankdata(x: np.ndarray) -> np.ndarray:
    """
    Average-rank for ties (like scipy.stats.rankdata(method="average")).
    Returns 1..n ranks.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i + 1
        while j < len(sorted_x) and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = float(ranks[order[i:j]].mean())
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x.astype(np.float64))
    ry = _rankdata(y.astype(np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _read_scores_csv(scores_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    energies: List[float] = []
    tms: List[float] = []
    with scores_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV header: {scores_csv}")
        if "energy" not in reader.fieldnames or "tm_ref_template" not in reader.fieldnames:
            raise KeyError(f"CSV missing required columns energy/tm_ref_template: {scores_csv}")
        for row in reader:
            energies.append(float(row["energy"]))
            tms.append(float(row["tm_ref_template"]))
    if len(energies) == 0:
        raise ValueError(f"No rows found in {scores_csv}")
    return np.asarray(energies, dtype=np.float64), np.asarray(tms, dtype=np.float64)


def _plot_one(scores_csv: Path, output_dir: Path, overwrite: bool) -> Dict[str, str]:
    protein_id = scores_csv.stem.replace("proteinebm_scores_", "")

    energies, tm = _read_scores_csv(scores_csv)
    score = -energies
    rho = _spearmanr(tm, score)

    output_dir.mkdir(parents=True, exist_ok=True)

    score_plot = output_dir / f"proteinebm_{protein_id}_score_vs_true_quality.png"
    energy_hist = output_dir / f"proteinebm_{protein_id}_energy_hist.png"
    tm_hist = output_dir / f"proteinebm_{protein_id}_tm_hist.png"

    if (not overwrite) and score_plot.exists() and energy_hist.exists() and tm_hist.exists():
        return {
            "score_vs_true_quality": str(score_plot.resolve()),
            "energy_hist": str(energy_hist.resolve()),
            "tm_hist": str(tm_hist.resolve()),
        }

    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.scatter(tm, score, s=18, alpha=0.5)
    plt.title(f"ProteinEBM: score(-energy) vs TM(ref, decoy)\n{protein_id} | Spearman ρ={rho:.3f}")
    plt.xlabel("TM-score (Reference vs Decoy) [tm_ref_template]")
    plt.ylabel("ProteinEBM score (-energy, higher is better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(score_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.hist(energies, bins=30, alpha=0.8)
    plt.title(f"ProteinEBM energy histogram\n{protein_id}")
    plt.xlabel("Energy (lower is better)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(energy_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.hist(tm, bins=30, alpha=0.8)
    plt.title(f"TM(ref, decoy) histogram\n{protein_id}")
    plt.xlabel("TM-score (Reference vs Decoy)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(tm_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "score_vs_true_quality": str(score_plot.resolve()),
        "energy_hist": str(energy_hist.resolve()),
        "tm_hist": str(tm_hist.resolve()),
    }


def _maybe_update_summary(summary_json: Path, plots: Dict[str, str], overwrite_summary: bool) -> None:
    if not summary_json.exists():
        return
    with summary_json.open("r") as f:
        summary = json.load(f)
    if (not overwrite_summary) and ("plots" in summary) and isinstance(summary["plots"], dict) and len(summary["plots"]) > 0:
        return
    summary["plots"] = plots
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill ProteinEBM per-protein plots from existing CSVs")
    parser.add_argument("--inference_dir", required=True, help="Base inference directory containing per-protein folders")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG plots")
    parser.add_argument(
        "--overwrite_summary_plots",
        action="store_true",
        help="Overwrite existing summary JSON 'plots' field if present",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of proteins processed (0 = no limit)")
    args = parser.parse_args()

    base = Path(args.inference_dir)
    if not base.exists():
        raise FileNotFoundError(f"inference_dir not found: {base}")

    score_csvs = sorted(base.glob("*/proteinebm_analysis/proteinebm_scores_*.csv"))
    if len(score_csvs) == 0:
        raise FileNotFoundError(f"No proteinebm_scores_*.csv found under {base}")

    n_done = 0
    for scores_csv in score_csvs:
        protein_dir = scores_csv.parent
        protein_id = scores_csv.stem.replace("proteinebm_scores_", "")
        summary_json = protein_dir / f"proteinebm_summary_{protein_id}.json"

        plots = _plot_one(scores_csv, output_dir=protein_dir, overwrite=bool(args.overwrite))
        _maybe_update_summary(summary_json, plots=plots, overwrite_summary=bool(args.overwrite_summary_plots))

        n_done += 1
        if args.limit and n_done >= int(args.limit):
            break

    print(f"Backfilled plots for {n_done} proteins under {base}", flush=True)


if __name__ == "__main__":
    main()

