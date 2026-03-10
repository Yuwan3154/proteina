#!/usr/bin/env python3
"""
Prediction Pipeline (No Ground-Truth)

Given protein sequences (CSV or FASTA), this pipeline:
1. Creates PT files from sequences
2. Runs Proteina inference (structure generation)
3. Scores with ProteinEBM (energy-based ranking)
4. Refines top-k with AF2Rank (pTM-based quality)
5. Collects best predictions and generates summary
6. Plots pTM distribution across all proteins

Usage:
    python run_prediction_pipeline.py \
        --input sequences.fasta \
        --inference_config config_name \
        --output_dir /path/to/output \
        --num_gpus 4
"""

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTEINA_BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
AF2RANK_EVAL_DIR = os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation")


def run_with_conda_env(env_name: str, command_list: list, cwd: str | None = None) -> bool:
    """Run a command with conda environment activation using shell script wrappers."""
    wrapper_map = {
        "proteina": os.path.join(AF2RANK_EVAL_DIR, "run_with_proteina_env.sh"),
        "colabdesign": os.path.join(AF2RANK_EVAL_DIR, "run_with_colabdesign_env.sh"),
        "proteinebm": os.path.join(AF2RANK_EVAL_DIR, "run_with_proteinebm_env.sh"),
    }
    wrapper_script = wrapper_map.get(env_name)
    if not wrapper_script or not os.path.exists(wrapper_script):
        logger.error(f"Wrapper script not found for environment: {env_name}")
        return False

    cmd = [wrapper_script] + command_list
    logger.info(f"Running in {env_name} environment: {' '.join(command_list)}")

    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to run command: {e}")
        return False


# ── Step 1: Parse input and create PT files ──────────────────────────────────

def step_parse_input(input_file: str, id_column: str, sequence_column: str, output_dir: str):
    """Parse input file, create PT files, and write working CSV."""
    from prediction_pipeline.input_parser import create_pt_files, create_working_csv, parse_input

    df = parse_input(input_file, id_column=id_column, sequence_column=sequence_column)
    create_pt_files(df)
    working_csv = create_working_csv(df, os.path.join(output_dir, "working_proteins.csv"))
    return df, working_csv


# ── Step 2: Proteina inference ────────────────────────────────────────────────

def step_proteina_inference(
    working_csv: str,
    inference_config: str,
    num_gpus: int,
    force_compile: bool = True,
) -> bool:
    """Run Proteina inference on all proteins."""
    logger.info("Starting Proteina inference...")
    cmd = [
        "python", os.path.join(AF2RANK_EVAL_DIR, "parallel_proteina_inference.py"),
        "--csv_file", working_csv,
        "--csv_column", "id",
        "--inference_config", inference_config,
        "--num_gpus", str(num_gpus),
        "--skip_existing",
        "--skip_pt_conversion",
    ]
    if force_compile:
        cmd.append("--force_compile")
    return run_with_conda_env("proteina", cmd)


# ── Step 3: ProteinEBM scoring ────────────────────────────────────────────────

def step_proteinebm_scoring(
    working_csv: str,
    inference_config: str,
    num_gpus: int,
    proteinebm_config: str,
    proteinebm_checkpoint: str,
    proteinebm_t: float = 0.05,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    batch_size: int = 32,
) -> bool:
    """Run ProteinEBM scoring (energy only, no ground-truth TM-score)."""
    logger.info("Starting ProteinEBM scoring...")
    cmd = [
        "python", os.path.join(AF2RANK_EVAL_DIR, "parallel_proteinebm_scoring.py"),
        "--csv_file", working_csv,
        "--csv_column", "id",
        "--inference_config", inference_config,
        "--num_gpus", str(num_gpus),
        "--filter_existing",
        "--proteinebm_config", proteinebm_config,
        "--proteinebm_checkpoint", proteinebm_checkpoint,
        "--proteinebm_t", str(proteinebm_t),
        "--proteinebm_analysis_subdir", proteinebm_analysis_subdir,
        "--batch_size", str(batch_size),
        # No --cif_dir: skips TM-score computation
    ]
    return run_with_conda_env("proteinebm", cmd)


# ── Step 4: AF2Rank on ProteinEBM top-k ──────────────────────────────────────

def step_af2rank_topk(
    inference_config: str,
    top_k: int,
    recycles: int,
    num_gpus: int,
    backend: str = "colabdesign",
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
) -> bool:
    """Run AF2Rank on ProteinEBM top-k templates."""
    logger.info(f"Starting AF2Rank on ProteinEBM top-{top_k} (backend={backend})...")
    inference_dir = os.path.join(PROTEINA_BASE_DIR, "inference", inference_config)
    cmd = [
        "python", os.path.join(AF2RANK_EVAL_DIR, "run_af2rank_on_proteinebm_topk.py"),
        "--inference_dir", inference_dir,
        "--top_k", str(top_k),
        "--recycles", str(recycles),
        "--num_gpus", str(num_gpus),
        "--backend", backend,
        "--proteinebm_analysis_subdir", proteinebm_analysis_subdir,
        "--filter_existing",
        # No --dataset_file: no ground-truth comparison
    ]
    return run_with_conda_env("proteina", cmd)


# ── Step 5: Collect results ───────────────────────────────────────────────────

def step_collect_results(
    protein_ids: list,
    inference_config: str,
    output_dir: str,
    ptm_cutoff: float,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
) -> list:
    """
    Read AF2Rank scores, select best structure per protein, copy to output.

    Returns list of per-protein result dicts.
    """
    import pandas as pd

    logger.info("Collecting results...")
    inference_base = os.path.join(PROTEINA_BASE_DIR, "inference", inference_config)
    structures_dir = os.path.join(output_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)

    results = []

    for protein_id in protein_ids:
        protein_dir = Path(inference_base) / protein_id
        topk_dir = protein_dir / "af2rank_on_proteinebm_top_k"
        m1_csv = topk_dir / "af2rank_analysis" / f"af2rank_scores_{protein_id}.csv"
        m2_csv = topk_dir / "af2rank_analysis_model_2_ptm" / f"af2rank_scores_{protein_id}.csv"

        # Count generated structures
        num_generated = len(list(protein_dir.glob(f"{protein_id}_*.pdb")))

        # Read sequence length from PT file
        data_path = os.environ.get("DATA_PATH", os.path.join(PROTEINA_BASE_DIR, "data"))
        pt_path = os.path.join(data_path, "pdb_train", "processed", f"{protein_id}.pt")
        try:
            import torch
            pt = torch.load(pt_path, weights_only=False, map_location="cpu")
            seq_len = len(pt.residue_type)
        except Exception:
            seq_len = 0

        if not m1_csv.exists() or not m2_csv.exists():
            logger.warning(f"AF2Rank scores not found for {protein_id}, skipping")
            results.append({
                "protein_id": protein_id,
                "sequence_length": seq_len,
                "num_generated": num_generated,
                "best_ptm": float("nan"),
                "best_plddt": float("nan"),
                "best_energy": float("nan"),
                "best_structure": "",
                "passes_cutoff": False,
            })
            continue

        m1_df = pd.read_csv(m1_csv)
        m2_df = pd.read_csv(m2_csv)

        # Merge across models: take min pTM per structure (conservative)
        merged = m1_df[["structure_file", "ptm", "plddt"]].merge(
            m2_df[["structure_file", "ptm"]], on="structure_file", suffixes=("_m1", "_m2"),
        )
        merged["min_ptm"] = merged[["ptm_m1", "ptm_m2"]].min(axis=1)

        # Also get energy from ProteinEBM scores
        ebm_csv = protein_dir / proteinebm_analysis_subdir / f"proteinebm_scores_{protein_id}.csv"
        energy_map = {}
        if ebm_csv.exists():
            ebm_df = pd.read_csv(ebm_csv)
            energy_map = dict(zip(ebm_df["structure_file"].astype(str), ebm_df["energy"].astype(float)))

        merged["energy"] = merged["structure_file"].map(energy_map).fillna(float("nan"))

        if merged.empty:
            results.append({
                "protein_id": protein_id,
                "sequence_length": seq_len,
                "num_generated": num_generated,
                "best_ptm": float("nan"),
                "best_plddt": float("nan"),
                "best_energy": float("nan"),
                "best_structure": "",
                "passes_cutoff": False,
            })
            continue

        # Select best structure by highest min_ptm
        best_idx = merged["min_ptm"].idxmax()
        best_row = merged.loc[best_idx]
        best_ptm = float(best_row["min_ptm"])
        best_plddt = float(best_row.get("plddt", float("nan")))
        best_energy = float(best_row.get("energy", float("nan")))
        best_file = str(best_row["structure_file"])
        passes = best_ptm >= ptm_cutoff

        # Find and copy the best structure file
        # The structure is in the staged_topk_templates directory
        staged_pdb = topk_dir / "staged_topk_templates" / best_file
        dest_pdb = os.path.join(structures_dir, f"{protein_id}.pdb")
        if staged_pdb.exists():
            shutil.copy2(str(staged_pdb), dest_pdb)
        else:
            # Fallback: look in inference dir
            fallback = protein_dir / best_file
            if fallback.exists():
                shutil.copy2(str(fallback), dest_pdb)
            else:
                logger.warning(f"Could not find structure file {best_file} for {protein_id}")
                dest_pdb = ""

        results.append({
            "protein_id": protein_id,
            "sequence_length": seq_len,
            "num_generated": num_generated,
            "best_ptm": best_ptm,
            "best_plddt": best_plddt,
            "best_energy": best_energy,
            "best_structure": os.path.basename(dest_pdb) if dest_pdb else "",
            "passes_cutoff": passes,
        })

    # Write prediction_summary.csv
    summary_csv_path = os.path.join(output_dir, "prediction_summary.csv")
    fieldnames = ["protein_id", "sequence_length", "num_generated", "best_ptm", "best_plddt", "best_energy", "best_structure", "passes_cutoff"]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Summary CSV written to {summary_csv_path}")

    # Write prediction_summary.json
    ptm_values = [r["best_ptm"] for r in results if not (isinstance(r["best_ptm"], float) and math.isnan(r["best_ptm"]))]
    num_passing = sum(1 for r in results if r["passes_cutoff"])
    summary_json = {
        "total_proteins": len(results),
        "num_passing_cutoff": num_passing,
        "fraction_passing": num_passing / len(results) if results else 0.0,
        "ptm_cutoff": ptm_cutoff,
        "ptm_mean": float(np.mean(ptm_values)) if ptm_values else None,
        "ptm_median": float(np.median(ptm_values)) if ptm_values else None,
        "ptm_std": float(np.std(ptm_values)) if ptm_values else None,
        "ptm_min": float(np.min(ptm_values)) if ptm_values else None,
        "ptm_max": float(np.max(ptm_values)) if ptm_values else None,
    }
    summary_json_path = os.path.join(output_dir, "prediction_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info(f"Summary JSON written to {summary_json_path}")

    # Write best_ptm_scores.csv
    ptm_csv_path = os.path.join(output_dir, "best_ptm_scores.csv")
    with open(ptm_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["protein_id", "best_ptm", "passes_cutoff"])
        writer.writeheader()
        for r in results:
            writer.writerow({"protein_id": r["protein_id"], "best_ptm": r["best_ptm"], "passes_cutoff": r["passes_cutoff"]})
    logger.info(f"Best pTM scores written to {ptm_csv_path}")

    return results


# ── Step 6: Generate distribution plot ────────────────────────────────────────

def step_plot_distribution(results: list, output_dir: str, ptm_cutoff: float) -> None:
    """Plot histogram of best pTM scores with cutoff line and fraction annotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ptm_values = [r["best_ptm"] for r in results if not (isinstance(r["best_ptm"], float) and math.isnan(r["best_ptm"]))]
    if not ptm_values:
        logger.warning("No valid pTM scores to plot")
        return

    num_passing = sum(1 for v in ptm_values if v >= ptm_cutoff)
    total = len(ptm_values)
    frac = num_passing / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ptm_values, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.axvline(ptm_cutoff, color="red", linestyle="--", linewidth=2, label=f"pTM cutoff = {ptm_cutoff}")
    ax.set_xlabel("Best pTM (min across models)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Best pTM Scores", fontsize=14)

    annotation = f"{num_passing}/{total} proteins ({frac:.0%}) pass pTM >= {ptm_cutoff}"
    ax.annotate(
        annotation,
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "ptm_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"pTM distribution plot saved to {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prediction Pipeline (No Ground-Truth)")
    parser.add_argument("--input", required=True, help="Input CSV or FASTA file with protein sequences")
    parser.add_argument("--id_column", default="id", help="Column name for protein ID in CSV (default: id)")
    parser.add_argument("--sequence_column", default="sequence", help="Column name for sequence in CSV (default: sequence)")
    parser.add_argument("--inference_config", required=True, help="Proteina inference configuration name")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--output_dir", required=True, help="Output directory for predictions and summary")
    parser.add_argument("--ptm_cutoff", type=float, default=0.7, help="pTM threshold for filtering (default: 0.7)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top ProteinEBM templates for AF2Rank (default: 5)")
    parser.add_argument("--backend", choices=["colabdesign", "openfold"], default="colabdesign",
                        help="AF2Rank backend: colabdesign (JAX) or openfold (PyTorch)")
    parser.add_argument("--recycles", type=int, default=3, help="AF2 recycles for AF2Rank (default: 3)")
    parser.add_argument("--proteinebm_config", default="/home/ubuntu/ProteinEBM/protein_ebm/config/base_pretrain.yaml",
                        help="Path to ProteinEBM config YAML")
    parser.add_argument("--proteinebm_checkpoint", default="/home/ubuntu/ProteinEBM/weights/model_1_frozen_1m_md.pt",
                        help="Path to ProteinEBM checkpoint")
    parser.add_argument("--proteinebm_t", type=float, default=0.05, help="Diffusion time for ProteinEBM (default: 0.05)")
    parser.add_argument("--proteinebm_analysis_subdir", default="proteinebm_v2_cathmd_analysis",
                        help="Per-protein subdir for ProteinEBM outputs")
    parser.add_argument("--proteinebm_batch_size", type=int, default=32,
                        help="Batch size for ProteinEBM inference (default: 32). Auto-reduces on OOM.")
    parser.add_argument("--skip_inference", action="store_true", help="Skip Proteina inference step")
    parser.add_argument("--skip_scoring", action="store_true", help="Skip ProteinEBM scoring step")
    parser.add_argument("--skip_af2rank", action="store_true", help="Skip AF2Rank step")
    parser.add_argument(
        "--force_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile for Proteina inference (default: True)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    success = True

    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE (No Ground-Truth)")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Inference config: {args.inference_config}")
    logger.info(f"GPUs: {args.num_gpus}")
    logger.info(f"pTM cutoff: {args.ptm_cutoff}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Output: {args.output_dir}")

    # ── Step 1: Parse input and create PT files ──
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: PARSE INPUT & CREATE PT FILES")
    logger.info("=" * 60)
    df, working_csv = step_parse_input(args.input, args.id_column, args.sequence_column, args.output_dir)
    protein_ids = df["id"].tolist()
    logger.info(f"Parsed {len(protein_ids)} proteins")

    # ── Step 2: Proteina inference ──
    if not args.skip_inference and success:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: PROTEINA INFERENCE")
        logger.info("=" * 60)
        if not step_proteina_inference(working_csv, args.inference_config, args.num_gpus, args.force_compile):
            logger.error("Proteina inference failed")
            success = False
        else:
            logger.info("Proteina inference completed successfully")
    elif args.skip_inference:
        logger.info("Skipping Proteina inference")

    # ── Step 3: ProteinEBM scoring ──
    if not args.skip_scoring and success:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: PROTEINEBM SCORING")
        logger.info("=" * 60)
        if not step_proteinebm_scoring(
            working_csv, args.inference_config, args.num_gpus,
            args.proteinebm_config, args.proteinebm_checkpoint,
            args.proteinebm_t, args.proteinebm_analysis_subdir,
            args.proteinebm_batch_size,
        ):
            logger.error("ProteinEBM scoring failed")
            success = False
        else:
            logger.info("ProteinEBM scoring completed successfully")
    elif args.skip_scoring:
        logger.info("Skipping ProteinEBM scoring")

    # ── Step 4: AF2Rank on ProteinEBM top-k ──
    if not args.skip_af2rank and success:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: AF2RANK ON PROTEINEBM TOP-K")
        logger.info("=" * 60)
        if not step_af2rank_topk(
            args.inference_config, args.top_k, args.recycles,
            args.num_gpus, args.backend, args.proteinebm_analysis_subdir,
        ):
            logger.error("AF2Rank step failed")
            success = False
        else:
            logger.info("AF2Rank step completed successfully")
    elif args.skip_af2rank:
        logger.info("Skipping AF2Rank step")

    # ── Step 5: Collect results ──
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: COLLECT RESULTS")
        logger.info("=" * 60)
        results = step_collect_results(
            protein_ids, args.inference_config, args.output_dir,
            args.ptm_cutoff, args.proteinebm_analysis_subdir,
        )

        # ── Step 6: Plot distribution ──
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: PTM DISTRIBUTION PLOT")
        logger.info("=" * 60)
        step_plot_distribution(results, args.output_dir, args.ptm_cutoff)

    # ── Summary ──
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    if success:
        num_passing = sum(1 for r in results if r["passes_cutoff"])
        logger.info(f"Pipeline completed in {total_time:.1f}s")
        logger.info(f"Proteins processed: {len(protein_ids)}")
        logger.info(f"Passing pTM >= {args.ptm_cutoff}: {num_passing}/{len(results)}")
        logger.info(f"Results: {args.output_dir}")
    else:
        logger.error(f"Pipeline failed after {total_time:.1f}s")

    logger.info("=" * 60)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
