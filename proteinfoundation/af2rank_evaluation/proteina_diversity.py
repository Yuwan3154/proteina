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
        resolve_num_workers,
    )
"""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proteinfoundation.af2rank_evaluation.proteinebm_scorer import (
    _extract_ca_coords,
    tmscore_pdb_paths,
)

logger = logging.getLogger(__name__)

# Matplotlib is not thread-safe; outer protein pool uses multiple threads.
_plot_lock = threading.Lock()

# Subprocess env for USalign when multiple USalign runs overlap (avoids OpenMP oversubscription).
_USALIGN_PARALLEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _parse_usalign_dir_stdout(text: str) -> List[float]:
    """
    Extract the first TM-score per alignment block from `USalign -dir` stdout
    (normalized by Structure_1), matching one value per unique pair.
    """
    out: List[float] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("Name of Structure_1:"):
            j = i + 1
            while j < len(lines):
                if lines[j].startswith("TM-score="):
                    rest = lines[j].split("=", 1)[1].strip()
                    out.append(float(rest.split()[0]))
                    break
                j += 1
            i = j
        i += 1
    return out


def _pairwise_tm_via_usalign_dir(
    protein_dir: str,
    basenames: List[str],
    env: Optional[Dict[str, str]],
) -> Optional[List[float]]:
    """
    Run one USalign process: all-against-all alignment for PDBs listed in a temp
    chain list (see USalign -dir). Returns TM-scores in encounter order, or None
    if USalign is missing or the command fails.
    """
    exe = shutil.which("USalign")
    if exe is None:
        return None
    list_f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    for b in basenames:
        list_f.write(b + "\n")
    list_f.close()
    list_path = list_f.name
    folder = os.path.abspath(protein_dir)
    if not folder.endswith(os.sep):
        folder = folder + os.sep
    cmd = [exe, "-dir", folder, list_path, "-TMscore", "5"]
    subprocess_env = os.environ.copy()
    if env is not None:
        subprocess_env.update(env)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env)
    os.unlink(list_path)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        logger.warning(f"USalign -dir failed (rc={proc.returncode}): {tail}")
        return None
    return _parse_usalign_dir_stdout(proc.stdout)


def resolve_num_workers(explicit: Optional[int]) -> int:
    """
    Resolve CPU worker budget for diversity (and pipeline) CPU-parallel steps.

    If explicit is None, uses clamp(os.cpu_count()). Otherwise clamps explicit to [1, 64].
    """
    if explicit is not None:
        return max(1, min(64, int(explicit)))
    n = os.cpu_count() or 1
    return max(1, min(64, n))


# ---------------------------------------------------------------------------
# Per-protein diversity computation
# ---------------------------------------------------------------------------

def compute_pairwise_tm(
    protein_dir: str,
    protein_id: str,
    output_subdir: str = "proteina_diversity",
    skip_existing: bool = True,
    pair_workers: int = 1,
    use_usalign_dir: bool = True,
) -> Optional[dict]:
    """
    Compute all-to-all pairwise TMscores for all Proteina samples of a protein.

    Since all samples share the same sequence length, TM1 == TM2 (symmetric),
    so we only run USalign once per unique pair (i < j) and take TM1.

    When use_usalign_dir is True and USalign is on PATH, uses a single
    ``USalign -dir <folder> <chain_list> -TMscore 5`` run (all-against-all
    built into USalign), which is much faster than one subprocess per pair.

    Otherwise falls back to per-pair ``USalign pdb_i pdb_j -TMscore 5`` (via
    tmscore_pdb_paths). pair_workers then
    applies to that fallback: max concurrent USalign subprocesses (>= 1).
    When > 1, each subprocess uses OMP_NUM_THREADS=1 in its environment.

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

    # Validate each PDB (CA-extractable); keep paths for -dir and per-pair USalign on files
    valid_pdbs: List[str] = []
    valid_basenames: List[str] = []
    for pdb_path in pdb_files:
        try:
            _extract_ca_coords(pdb_path, chain_id=None)
            valid_pdbs.append(pdb_path)
            valid_basenames.append(os.path.basename(pdb_path))
        except Exception as e:
            logger.warning(f"{protein_id}: failed to extract CA coords from {os.path.basename(pdb_path)}: {e}")

    n = len(valid_pdbs)
    if n < 2:
        logger.warning(f"{protein_id}: only {n} valid structures after CA extraction, need >= 2")
        return None

    n_pairs = n * (n - 1) // 2
    pw = max(1, int(pair_workers))
    pair_indices = list(combinations(range(n), 2))

    tm_values: List[float] = []
    pairwise_mode = "per_pair"
    tm_env = _USALIGN_PARALLEL_ENV if pw > 1 else None

    if use_usalign_dir and shutil.which("USalign"):
        logger.info(
            f"{protein_id}: computing {n_pairs} pairwise TMscores for {n} samples "
            f"(USalign -dir single pass)"
        )
        tm_values = _pairwise_tm_via_usalign_dir(protein_dir, valid_basenames, tm_env) or []
        if len(tm_values) != n_pairs:
            logger.warning(
                f"{protein_id}: USalign -dir parsed {len(tm_values)} TM-scores, expected {n_pairs}; "
                "falling back to per-pair USalign"
            )
            tm_values = []

    if not tm_values:
        pairwise_mode = "per_pair"
        logger.info(f"{protein_id}: computing {n_pairs} pairwise TMscores for {n} samples (pair_workers={pw})")
        use_parallel = pw > 1 and n_pairs > 0
        tm_env = _USALIGN_PARALLEL_ENV if use_parallel else None

        def _pair_tm_safe(ij: tuple) -> Optional[float]:
            i, j = ij
            try:
                result = tmscore_pdb_paths(valid_pdbs[i], valid_pdbs[j], env=tm_env)
                return float(result["tms"])
            except Exception as e:
                logger.warning(f"{protein_id}: TMscore failed for pair ({i},{j}): {e}")
                return None

        if use_parallel:
            with ThreadPoolExecutor(max_workers=pw) as ex:
                for val in ex.map(_pair_tm_safe, pair_indices):
                    if val is not None:
                        tm_values.append(val)
        else:
            for i, j in pair_indices:
                try:
                    result = tmscore_pdb_paths(valid_pdbs[i], valid_pdbs[j])
                    tm_values.append(result["tms"])
                except Exception as e:
                    logger.warning(f"{protein_id}: TMscore failed for pair ({i},{j}): {e}")
    else:
        pairwise_mode = "usalign_dir"

    if not tm_values:
        logger.warning(f"{protein_id}: no successful pairwise TMscore computations")
        return None

    tm_arr = np.array(tm_values, dtype=np.float64)
    summary = {
        "protein_id": protein_id,
        "n_samples": n,
        "n_pairs": len(tm_values),
        "pairwise_tm_mode": pairwise_mode,
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
    with _plot_lock:
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
        plt.close(fig)
    logger.info(f"Pairwise TM histogram saved to {output_path}")


# ---------------------------------------------------------------------------
# Batch processing across proteins
# ---------------------------------------------------------------------------

def compute_diversity_for_proteins(
    inference_dir: str,
    protein_ids: List[str],
    output_subdir: str = "proteina_diversity",
    skip_existing: bool = True,
    num_workers: Optional[int] = None,
    use_usalign_dir: bool = True,
) -> Dict[str, dict]:
    """
    Compute pairwise TM diversity for each protein.

    Uses nested ThreadPoolExecutors: outer proteins (up to outer workers) and
    inner pair_workers per protein so outer * inner <= max_workers.

    num_workers: optional explicit cap; if None, uses resolve_num_workers(None) (clamped os.cpu_count()).

    Returns {protein_id: summary_dict} for proteins with valid results.
    """
    results: Dict[str, dict] = {}
    w = resolve_num_workers(num_workers)
    n_list = len(protein_ids)
    outer = min(w, max(1, n_list))
    inner = max(1, w // outer)
    logger.info(
        f"Diversity: num_workers={w} outer={outer} inner={inner} "
        f"(proteins={n_list})"
    )

    def _one_protein(protein_id: str) -> tuple:
        protein_dir = os.path.join(inference_dir, protein_id)
        if not os.path.isdir(protein_dir):
            return protein_id, None, "missing_dir"
        summary = compute_pairwise_tm(
            protein_dir,
            protein_id,
            output_subdir=output_subdir,
            skip_existing=skip_existing,
            pair_workers=inner,
            use_usalign_dir=use_usalign_dir,
        )
        return protein_id, summary, None

    if outer > 1 and n_list > 1:
        with ThreadPoolExecutor(max_workers=outer) as ex:
            futures = {ex.submit(_one_protein, pid): pid for pid in protein_ids}
            for fut in as_completed(futures):
                pid, summary, err = fut.result()
                if err == "missing_dir":
                    logger.warning(f"{pid}: directory not found, skipping")
                elif summary is not None:
                    results[pid] = summary
    else:
        for idx, protein_id in enumerate(protein_ids, 1):
            protein_dir = os.path.join(inference_dir, protein_id)
            if not os.path.isdir(protein_dir):
                logger.warning(f"[{idx}/{len(protein_ids)}] {protein_id}: directory not found, skipping")
                continue
            logger.info(f"[{idx}/{len(protein_ids)}] Processing {protein_id}")
            summary = compute_pairwise_tm(
                protein_dir,
                protein_id,
                output_subdir=output_subdir,
                skip_existing=skip_existing,
                pair_workers=inner,
                use_usalign_dir=use_usalign_dir,
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Max parallel CPU workers (clamped 1–64); default is clamped os.cpu_count()",
    )
    parser.add_argument(
        "--no_usalign_dir",
        action="store_true",
        help="Disable USalign -dir all-against-all (use per-pair USalign on PDB files only)",
    )
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
        num_workers=args.num_workers,
        use_usalign_dir=not args.no_usalign_dir,
    )

    logger.info(f"Done. {len(results)} proteins with diversity metrics.")


if __name__ == "__main__":
    main()
