#!/usr/bin/env python3
"""
Proteina Sample Diversity Analysis

Computes all-to-all pairwise TMscores between Proteina-generated samples for each
protein chain, producing per-chain histograms and summary statistics.

Usage (standalone):
    python proteina_diversity.py --inference_dir <path> --protein_ids <id1> <id2> ...
    python proteina_diversity.py --inference_dir <path> --csv_file data.csv --csv_column id

Usage (as library):
    from proteinfoundation.af2rank_evaluation.proteina_diversity import (
        compute_pairwise_tm, compute_diversity_for_proteins, load_diversity_data,
    )
"""

import argparse
import glob
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proteinfoundation.af2rank_evaluation.proteinebm_scorer import (
    _extract_ca_coords,
    tmscore_ca_coords,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-protein diversity computation
# ---------------------------------------------------------------------------

def compute_pairwise_tm(
    protein_dir: str,
    protein_id: str,
    output_subdir: str = "proteina_diversity",
    skip_existing: bool = True,
) -> Optional[dict]:
    """
    Compute all-to-all pairwise TMscores for all Proteina samples of a protein.

    Since all samples share the same sequence length, TM1 == TM2 (symmetric),
    so we only run USalign once per unique pair (i < j) and take TM1.

    Returns summary dict or None on failure.
    """
    protein_dir = str(protein_dir)
    out_dir = os.path.join(protein_dir, output_subdir)
    summary_path = os.path.join(out_dir, f"diversity_summary_{protein_id}.json")

    if skip_existing and os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # recompute

    # Find all generated PDB files
    pdb_pattern = os.path.join(protein_dir, f"{protein_id}_*.pdb")
    pdb_files = sorted(glob.glob(pdb_pattern))

    if len(pdb_files) < 2:
        logger.warning(f"{protein_id}: found {len(pdb_files)} PDB files, need >= 2 for pairwise TM")
        return None

    # Extract CA coordinates from each PDB
    ca_coords: List[np.ndarray] = []
    valid_files: List[str] = []
    for pdb_path in pdb_files:
        try:
            coords = _extract_ca_coords(pdb_path, chain_id=None)
            ca_coords.append(coords)
            valid_files.append(pdb_path)
        except Exception as e:
            logger.warning(f"{protein_id}: failed to extract CA coords from {os.path.basename(pdb_path)}: {e}")

    n = len(ca_coords)
    if n < 2:
        logger.warning(f"{protein_id}: only {n} valid structures after CA extraction, need >= 2")
        return None

    # Compute pairwise TMscore for all unique pairs (i < j)
    tm_values: List[float] = []
    n_pairs = n * (n - 1) // 2
    logger.info(f"{protein_id}: computing {n_pairs} pairwise TMscores for {n} samples")

    for i, j in combinations(range(n), 2):
        try:
            result = tmscore_ca_coords(ca_coords[i], ca_coords[j])
            tm_values.append(result["tms"])
        except Exception as e:
            logger.warning(f"{protein_id}: TMscore failed for pair ({i},{j}): {e}")

    if not tm_values:
        logger.warning(f"{protein_id}: no successful pairwise TMscore computations")
        return None

    tm_arr = np.array(tm_values, dtype=np.float64)
    summary = {
        "protein_id": protein_id,
        "n_samples": n,
        "n_pairs": len(tm_values),
        "mean_tem_to_tem_tm": float(tm_arr.mean()),
        "std_tem_to_tem_tm": float(tm_arr.std()),
        "median_tem_to_tem_tm": float(np.median(tm_arr)),
        "min_tem_to_tem_tm": float(tm_arr.min()),
        "max_tem_to_tem_tm": float(tm_arr.max()),
    }

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"{protein_id}: diversity summary saved to {summary_path}")

    hist_path = os.path.join(out_dir, f"pairwise_tm_histogram_{protein_id}.png")
    plot_pairwise_tm_histogram(tm_values, hist_path, protein_id)

    return summary


def plot_pairwise_tm_histogram(
    tm_values: List[float],
    output_path: str,
    protein_id: str,
) -> None:
    """Save histogram of pairwise TMscores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tm_values, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("Pairwise TM-score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Sample-to-Sample TM-score Distribution — {protein_id}", fontsize=13)

    tm_arr = np.array(tm_values)
    annotation = (
        f"N={len(tm_values)} pairs\n"
        f"mean={tm_arr.mean():.3f}, median={np.median(tm_arr):.3f}\n"
        f"std={tm_arr.std():.3f}"
    )
    ax.annotate(
        annotation,
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Pairwise TM histogram saved to {output_path}")


# ---------------------------------------------------------------------------
# Batch processing across proteins
# ---------------------------------------------------------------------------

def compute_diversity_for_proteins(
    inference_dir: str,
    protein_ids: List[str],
    output_subdir: str = "proteina_diversity",
    skip_existing: bool = True,
) -> Dict[str, dict]:
    """
    Compute pairwise TM diversity for each protein sequentially.

    Returns {protein_id: summary_dict} for proteins with valid results.
    """
    results: Dict[str, dict] = {}
    for idx, protein_id in enumerate(protein_ids, 1):
        protein_dir = os.path.join(inference_dir, protein_id)
        if not os.path.isdir(protein_dir):
            logger.warning(f"[{idx}/{len(protein_ids)}] {protein_id}: directory not found, skipping")
            continue
        logger.info(f"[{idx}/{len(protein_ids)}] Processing {protein_id}")
        summary = compute_pairwise_tm(
            protein_dir, protein_id, output_subdir=output_subdir, skip_existing=skip_existing,
        )
        if summary is not None:
            results[protein_id] = summary
    logger.info(f"Diversity computed for {len(results)}/{len(protein_ids)} proteins")
    return results


# ---------------------------------------------------------------------------
# Loading diversity data (for cross-protein analysis)
# ---------------------------------------------------------------------------

def find_diversity_summaries(inference_dir: str, subdir: str = "proteina_diversity") -> List[str]:
    """Find all diversity summary JSON files under inference_dir."""
    pattern = os.path.join(inference_dir, "*", subdir, "diversity_summary_*.json")
    return sorted(glob.glob(pattern))


def load_diversity_data(inference_dir: str, subdir: str = "proteina_diversity") -> dict:
    """
    Load diversity summaries for all proteins.

    Returns {protein_id: summary_dict}.
    """
    summary_files = find_diversity_summaries(inference_dir, subdir)
    data: Dict[str, dict] = {}
    for path in summary_files:
        try:
            with open(path, "r") as f:
                summary = json.load(f)
            pid = summary.get("protein_id") or Path(path).parent.parent.name
            data[pid] = summary
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {path}: {e}")
    logger.info(f"Loaded diversity data for {len(data)} proteins")
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute Proteina sample diversity (all-to-all TMscore)")
    parser.add_argument("--inference_dir", required=True, help="Base inference directory")
    parser.add_argument("--protein_ids", nargs="*", help="Protein IDs to process")
    parser.add_argument("--csv_file", help="CSV file with protein IDs (alternative to --protein_ids)")
    parser.add_argument("--csv_column", default="id", help="Column name for protein ID in CSV (default: id)")
    parser.add_argument("--output_subdir", default="proteina_diversity",
                        help="Per-protein subdirectory for outputs (default: proteina_diversity)")
    parser.add_argument("--rerun", action="store_true", help="Recompute even if results exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.protein_ids:
        protein_ids = args.protein_ids
    elif args.csv_file:
        import pandas as pd
        df = pd.read_csv(args.csv_file)
        protein_ids = df[args.csv_column].dropna().astype(str).str.strip().unique().tolist()
    else:
        # Auto-detect from inference_dir subdirectories
        protein_ids = sorted([
            d for d in os.listdir(args.inference_dir)
            if os.path.isdir(os.path.join(args.inference_dir, d))
        ])

    if not protein_ids:
        logger.error("No protein IDs found")
        sys.exit(1)

    logger.info(f"Processing {len(protein_ids)} proteins from {args.inference_dir}")
    results = compute_diversity_for_proteins(
        args.inference_dir, protein_ids,
        output_subdir=args.output_subdir,
        skip_existing=not args.rerun,
    )

    logger.info(f"Done. {len(results)} proteins with diversity metrics.")


if __name__ == "__main__":
    main()
