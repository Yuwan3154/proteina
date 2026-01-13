#!/usr/bin/env python3
"""
Parallel ProteinEBM Scoring Script

Scores Proteina-generated decoy PDBs using ProteinEBM across multiple GPUs.

Outputs (per protein):
  <PROTEINA_BASE_DIR>/inference/<inference_config>/<protein_id>/proteinebm_analysis/
    - proteinebm_scores_<protein_id>.csv
    - proteinebm_summary_<protein_id>.json
"""

import argparse
import builtins
import csv
import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _terminate_process_group(signum: int) -> None:
    """
    Terminate this job's whole process group (main + worker processes + scorer subprocesses).
    This prevents orphaned GPU processes after Ctrl+C / SIGTERM.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.killpg(os.getpgrp(), signum)


def signal_handler(signum, frame):
    """Handle interrupt signals by terminating the whole process group."""
    sig_name = signal.Signals(signum).name
    logger.warning(f"\n⚠️  Received {sig_name}. Terminating all worker/scorer processes...")
    _terminate_process_group(signum)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_proteina_base_dir():
    """Auto-detect proteina base directory."""
    possible_dirs = [
        os.path.expanduser("~/proteina"),
        os.path.join(os.getcwd(), ".."),  # Assume we're in af2rank_evaluation
    ]
    for dir_path in possible_dirs:
        abs_path = os.path.abspath(dir_path)
        if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, "proteinfoundation")):
            return abs_path
    return os.path.abspath(os.path.join(os.getcwd(), ".."))


PROTEINA_BASE_DIR = get_proteina_base_dir()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate_protein_output_dir(inference_config: str, protein_name: str) -> str:
    return os.path.join(PROTEINA_BASE_DIR, "inference", inference_config, protein_name)


def find_reference_cif(protein_name: str, cif_dir: str) -> str:
    """Find the reference CIF file for a protein (same logic as parallel_af2rank_scoring.py)."""
    pdb_id = protein_name.split("_")[0]

    cif_path = Path(cif_dir)
    for subdir in cif_path.iterdir():
        if subdir.is_dir():
            potential_cif = subdir / f"{pdb_id}.cif"
            if potential_cif.exists():
                return str(potential_cif)

    potential_cif = cif_path / f"{pdb_id}.cif"
    if potential_cif.exists():
        return str(potential_cif)

    raise FileNotFoundError(f"CIF file not found for {pdb_id} in {cif_dir}")


def get_protein_names(csv_file: str, csv_column: str):
    """Extract protein names from CSV file without pandas dependency."""
    proteins = []
    seen = set()

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []

        field_map = {name.strip(): name for name in reader.fieldnames}
        actual_col = field_map.get(csv_column.strip())
        if actual_col is None:
            raise KeyError(f"Column '{csv_column}' not found in CSV header: {reader.fieldnames}")

        for row in reader:
            val = (row.get(actual_col) or "").strip()
            if val and val not in seen:
                seen.add(val)
                proteins.append(val)

    return proteins


def find_proteins_needing_proteinebm(csv_file: str, csv_column: str, inference_config: str):
    csv_proteins = get_protein_names(csv_file, csv_column)
    inference_base_dir = os.path.join(PROTEINA_BASE_DIR, "inference", inference_config)
    if not os.path.exists(inference_base_dir):
        return []

    proteins_needing_work = []
    for protein_name in csv_proteins:
        protein_dir = Path(inference_base_dir) / protein_name
        if not protein_dir.exists() or not protein_dir.is_dir():
            continue

        pdb_files = list(protein_dir.glob(f"{protein_name}_*.pdb"))
        if not pdb_files:
            continue

        scores_csv = protein_dir / "proteinebm_analysis" / f"proteinebm_scores_{protein_name}.csv"
        if not scores_csv.exists():
            proteins_needing_work.append(protein_name)
        else:
            # Treat old outputs (missing TM columns) as incomplete so we can regenerate summaries.
            with open(scores_csv, "r") as f:
                header = (f.readline() or "").strip()
            if "tm_ref_template" not in header:
                proteins_needing_work.append(protein_name)
            else:
                logger.info(f"ProteinEBM scoring already completed for {protein_name}, skipping")

    return proteins_needing_work


def run_proteinebm_scoring_subprocess(
    protein_name: str,
    inference_output_dir: str,
    reference_cif: str,
    proteinebm_config: str,
    proteinebm_checkpoint: str,
    template_self_condition: bool,
    t: float = 0.1,
):
    wrapper_script = os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation", "run_with_proteinebm_env.sh")
    scorer_script = os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation", "proteinebm_scorer.py")
    output_dir = os.path.join(inference_output_dir, "proteinebm_analysis")

    cmd = [
        wrapper_script,
        "python",
        scorer_script,
        "--protein_id",
        protein_name,
        "--reference_cif",
        reference_cif,
        "--inference_output_dir",
        inference_output_dir,
        "--output_dir",
        output_dir,
        "--proteinebm_config",
        proteinebm_config,
        "--proteinebm_checkpoint",
        proteinebm_checkpoint,
        "--t",
        str(t),
    ]

    if not template_self_condition:
        cmd.append("--no-proteinebm_template_self_condition")

    return subprocess.run(cmd, cwd=os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation"))


def process_single_protein_proteinebm(args):
    protein_name, cif_dir, inference_config, gpu_id, proteinebm_config, proteinebm_checkpoint, template_self_condition = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(f"[GPU {gpu_id}] Starting ProteinEBM scoring for {protein_name}")

    inference_output_dir = generate_protein_output_dir(inference_config, protein_name)
    if not os.path.exists(inference_output_dir):
        raise FileNotFoundError(f"Inference output directory not found: {inference_output_dir}")

    pdb_files = list(Path(inference_output_dir).glob(f"{protein_name}_*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {inference_output_dir}")

    reference_cif = find_reference_cif(protein_name, cif_dir)
    logger.info(f"[GPU {gpu_id}] Found reference CIF: {reference_cif}")

    result = run_proteinebm_scoring_subprocess(
        protein_name=protein_name,
        inference_output_dir=inference_output_dir,
        reference_cif=reference_cif,
        proteinebm_config=proteinebm_config,
        proteinebm_checkpoint=proteinebm_checkpoint,
        template_self_condition=template_self_condition,
        t=0.1,
    )

    if result.returncode != 0:
        raise Exception(f"ProteinEBM scoring failed with returncode {result.returncode}")

    analysis_dir = os.path.join(inference_output_dir, "proteinebm_analysis")
    summary_file = os.path.join(analysis_dir, f"proteinebm_summary_{protein_name}.json")
    summary_info = {}
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary_info = json.load(f)

    logger.info(f"[GPU {gpu_id}] ✅ ProteinEBM scoring completed for {protein_name}")

    return {
        "protein": protein_name,
        "gpu": gpu_id,
        "status": "success",
        "output_dir": analysis_dir,
        "summary": summary_info,
    }


def worker_init_proteinebm(counter, lock, num_gpus):
    """Initialize worker with a specific GPU assignment."""
    with lock:
        worker_id = counter.value
        counter.value += 1

    gpu_id = worker_id % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    builtins._worker_gpu_id = gpu_id
    logger.info(f"ProteinEBM Worker {worker_id} initialized with GPU {gpu_id}")


def process_single_protein_proteinebm_wrapper(args_tuple):
    protein_name, cif_dir, inference_config, proteinebm_config, proteinebm_checkpoint, template_self_condition = args_tuple
    gpu_id = getattr(builtins, "_worker_gpu_id", 0)
    full_args = (protein_name, cif_dir, inference_config, gpu_id, proteinebm_config, proteinebm_checkpoint, template_self_condition)
    return process_single_protein_proteinebm(full_args)


def main():
    parser = argparse.ArgumentParser(description="Parallel ProteinEBM scoring pipeline")
    parser.add_argument("--csv_file", required=True, help="Path to CSV file with protein data")
    parser.add_argument("--csv_column", required=True, help="Column name in CSV file to use for protein selection")
    parser.add_argument("--cif_dir", required=True, help="Directory containing reference CIF files (for TMscore ground-truth)")
    parser.add_argument("--inference_config", required=True, help="Inference configuration name")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--filter_existing",
        action="store_true",
        help="Only process proteins that have completed inference but do not yet have ProteinEBM scores",
    )
    parser.add_argument("--proteinebm_config", required=True, help="Path to ProteinEBM base_pretrain.yaml")
    parser.add_argument("--proteinebm_checkpoint", required=True, help="Path to ProteinEBM checkpoint (.pt)")
    parser.add_argument(
        "--proteinebm_template_self_condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use template coordinates for self-conditioning input (default: True)",
    )

    args = parser.parse_args()

    if args.filter_existing:
        protein_names = find_proteins_needing_proteinebm(args.csv_file, args.csv_column, args.inference_config)
        logger.info(f"Found {len(protein_names)} proteins needing ProteinEBM scoring (from CSV file)")
    else:
        protein_names = get_protein_names(args.csv_file, args.csv_column)
        logger.info(f"Found {len(protein_names)} proteins in CSV file")

    if not protein_names:
        logger.warning("No proteins to process")
        sys.exit(0)

    logger.info(f"Using {args.num_gpus} GPU(s) for parallel ProteinEBM scoring")

    start_time = time.time()
    all_results = []

    manager = mp.Manager()
    worker_counter = manager.Value("i", 0)
    worker_lock = manager.Lock()

    executor = ProcessPoolExecutor(
        max_workers=args.num_gpus,
        initializer=worker_init_proteinebm,
        initargs=(worker_counter, worker_lock, args.num_gpus),
    )

    try:
        work_items = [
            (protein_name, args.cif_dir, args.inference_config, args.proteinebm_config, args.proteinebm_checkpoint, args.proteinebm_template_self_condition)
            for protein_name in protein_names
        ]

        future_to_protein = {executor.submit(process_single_protein_proteinebm_wrapper, item): item[0] for item in work_items}

        for future in as_completed(future_to_protein):
            protein_name = future_to_protein[future]
            result = future.result()
            all_results.append(result)

            completed = len([r for r in all_results if r["status"] == "success"])
            total = len(work_items)
            elapsed = time.time() - start_time

            summary = result.get("summary", {})
            successful_scores = summary.get("successful_scores", 0)
            total_structures = summary.get("total_structures", 0)
            logger.info(
                f"✅ Progress: {completed}/{total} proteins completed ({successful_scores}/{total_structures} structures scored, {elapsed:.1f}s elapsed)"
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        logger.info("✓ Executor shut down cleanly")

    total_time = time.time() - start_time
    successful = len([r for r in all_results if r["status"] == "success"])

    logger.info("\n📊 ProteinEBM Processing Results:")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s")

    if successful:
        logger.info(f"🎉 Successfully processed {successful} proteins!")
        logger.info(f"📁 Results saved to: {os.path.join(PROTEINA_BASE_DIR, 'inference', args.inference_config)}/")

    sys.exit(0)


if __name__ == "__main__":
    main()



