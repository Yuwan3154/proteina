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
import csv
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List


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


def find_proteins_needing_proteinebm(csv_file: str, csv_column: str, inference_config: str, analysis_subdir: str = "proteinebm_v2_cathmd_analysis"):
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

        scores_csv = protein_dir / analysis_subdir / f"proteinebm_scores_{protein_name}.csv"
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


def _get_protein_length(inference_output_dir: str, protein_name: str) -> int:
    """Fast protein length estimate by counting CA atoms in the first available PDB."""
    pdb_files = sorted(Path(inference_output_dir).glob(f"{protein_name}_*.pdb"))
    if not pdb_files:
        return 0
    count = 0
    with open(pdb_files[0]) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                count += 1
    return count


def _build_protein_configs(
    protein_names: List[str],
    cif_dir: str | None,
    inference_config: str,
    analysis_subdir: str,
) -> List[Dict]:
    """Build per-protein config dicts, sorted by sequence length (ascending)."""
    configs = []
    for protein_name in protein_names:
        inference_output_dir = generate_protein_output_dir(inference_config, protein_name)
        if not os.path.exists(inference_output_dir):
            logger.warning(f"Inference directory not found, skipping: {inference_output_dir}")
            continue
        reference_cif = None
        reference_chain = None
        if cif_dir is not None:
            try:
                reference_cif = find_reference_cif(protein_name, cif_dir)
                if "_" in protein_name:
                    reference_chain = protein_name.split("_", 1)[1]
            except FileNotFoundError as e:
                logger.warning(f"CIF not found for {protein_name}: {e}")
        output_dir = os.path.join(inference_output_dir, analysis_subdir)
        length = _get_protein_length(inference_output_dir, protein_name)
        configs.append({
            "protein_id": protein_name,
            "inference_output_dir": inference_output_dir,
            "output_dir": output_dir,
            "reference_cif": reference_cif,
            "reference_chain": reference_chain,
            "length": length,
        })

    # Sort by length ascending so the batch_size_cache in the scorer is maximally effective
    configs.sort(key=lambda c: c["length"])
    return configs


def run_gpu_worker_subprocess(
    gpu_id: int,
    protein_configs: List[Dict],
    proteinebm_config: str,
    proteinebm_checkpoint: str,
    template_self_condition: bool,
    t: float,
    batch_size: int,
) -> subprocess.CompletedProcess:
    """Spawn one proteinebm_scorer.py subprocess for a single GPU, processing all
    assigned proteins in a single model load (multi-protein mode)."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(protein_configs, tmp)
    tmp.flush()
    proteins_json_path = tmp.name
    tmp.close()

    wrapper_script = os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation", "run_with_proteinebm_env.sh")
    scorer_script = os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation", "proteinebm_scorer.py")

    cmd = [
        wrapper_script, "python", scorer_script,
        "--proteins_json", proteins_json_path,
        "--proteinebm_config", proteinebm_config,
        "--proteinebm_checkpoint", proteinebm_checkpoint,
        "--t", str(t),
        "--batch_size", str(batch_size),
    ]
    if not template_self_condition:
        cmd.append("--no-proteinebm_template_self_condition")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation"),
            env=env,
        )
    finally:
        os.unlink(proteins_json_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Parallel ProteinEBM scoring pipeline")
    parser.add_argument("--csv_file", required=True, help="Path to CSV file with protein data")
    parser.add_argument("--csv_column", required=True, help="Column name in CSV file to use for protein selection")
    parser.add_argument("--cif_dir", default=None, help="Directory containing reference CIF files (for TMscore ground-truth). Optional; if omitted, TM-score metrics are skipped.")
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
    parser.add_argument(
        "--proteinebm_analysis_subdir",
        default="proteinebm_v2_cathmd_analysis",
        help="Per-protein subdir for ProteinEBM outputs (default: proteinebm_v2_cathmd_analysis)",
    )
    parser.add_argument(
        "--proteinebm_t",
        type=float,
        default=0.05,
        help="Diffusion time t for ProteinEBM scoring (default: 0.05)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for ProteinEBM inference per protein (default: 32). Auto-reduces on OOM.",
    )

    args = parser.parse_args()

    if args.filter_existing:
        protein_names = find_proteins_needing_proteinebm(args.csv_file, args.csv_column, args.inference_config, args.proteinebm_analysis_subdir)
        logger.info(f"Found {len(protein_names)} proteins needing ProteinEBM scoring (from CSV file)")
    else:
        protein_names = get_protein_names(args.csv_file, args.csv_column)
        logger.info(f"Found {len(protein_names)} proteins in CSV file")

    if not protein_names:
        logger.warning("No proteins to process")
        sys.exit(0)

    logger.info(f"Using {args.num_gpus} GPU(s) for parallel ProteinEBM scoring")
    logger.info("Sorting proteins by sequence length and building per-GPU work lists...")

    # Build full config list sorted by length (ascending)
    all_configs = _build_protein_configs(
        protein_names, args.cif_dir, args.inference_config, args.proteinebm_analysis_subdir
    )
    if not all_configs:
        logger.warning("No valid protein directories found")
        sys.exit(0)

    logger.info(
        f"Length range: {all_configs[0]['length']}–{all_configs[-1]['length']} residues "
        f"across {len(all_configs)} proteins"
    )

    # Split into num_gpus contiguous chunks (contiguous = same-length proteins stay together,
    # maximising batch_size_cache hits within each GPU subprocess)
    num_gpus = args.num_gpus
    gpu_chunks: List[List[Dict]] = [[] for _ in range(num_gpus)]
    for i, cfg in enumerate(all_configs):
        gpu_chunks[i % num_gpus].append(cfg)
    # After round-robin assignment each chunk is still roughly sorted by length
    # (every num_gpus-th element of a sorted list). Sort each chunk explicitly.
    for chunk in gpu_chunks:
        chunk.sort(key=lambda c: c["length"])

    start_time = time.time()

    # Spawn one subprocess per GPU (each loads the model once and processes its full chunk)
    procs: List[subprocess.Popen] = []
    for gpu_id, chunk in enumerate(gpu_chunks):
        if not chunk:
            continue
        logger.info(f"GPU {gpu_id}: {len(chunk)} proteins, lengths {chunk[0]['length']}–{chunk[-1]['length']}")

    # Run GPU workers in parallel — one Popen per GPU, wait for all.
    popen_procs: List[tuple] = []  # (gpu_id, Popen, proteins_json_path)

    wrapper_script = os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation", "run_with_proteinebm_env.sh")
    scorer_script = os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation", "proteinebm_scorer.py")

    for gpu_id, chunk in enumerate(gpu_chunks):
        if not chunk:
            continue
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(chunk, tmp)
        tmp.flush()
        proteins_json_path = tmp.name
        tmp.close()

        cmd = [
            wrapper_script, "python", scorer_script,
            "--proteins_json", proteins_json_path,
            "--proteinebm_config", args.proteinebm_config,
            "--proteinebm_checkpoint", args.proteinebm_checkpoint,
            "--t", str(args.proteinebm_t),
            "--batch_size", str(args.batch_size),
        ]
        if not args.proteinebm_template_self_condition:
            cmd.append("--no-proteinebm_template_self_condition")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.join(PROTEINA_BASE_DIR, "af2rank_evaluation"),
            env=env,
        )
        popen_procs.append((gpu_id, proc, proteins_json_path))
        logger.info(f"Launched GPU {gpu_id} worker (pid={proc.pid}) for {len(chunk)} proteins")

    # Wait for all GPU workers
    failed_gpus = []
    for gpu_id, proc, proteins_json_path in popen_procs:
        returncode = proc.wait()
        try:
            os.unlink(proteins_json_path)
        except OSError:
            pass
        if returncode != 0:
            logger.error(f"GPU {gpu_id} worker exited with code {returncode}")
            failed_gpus.append(gpu_id)
        else:
            logger.info(f"GPU {gpu_id} worker finished successfully")

    total_time = time.time() - start_time
    n_proteins = sum(len(c) for c in gpu_chunks)
    logger.info(f"\nProteinEBM scoring complete: {n_proteins} proteins in {total_time:.1f}s")
    if failed_gpus:
        logger.error(f"Failed GPU workers: {failed_gpus}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()



