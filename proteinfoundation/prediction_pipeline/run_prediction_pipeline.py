#!/usr/bin/env python3
"""
Prediction Pipeline (unified prediction + evaluation driver).

Given either a sequence file (CSV/FASTA via --input) or a dataset CSV with
pre-existing PT files (--dataset_file), this pipeline:

  1. (optional) Parses sequences and creates PT files   -- --input mode only
  2. Runs Proteina inference (structure generation)
  3. Scores with ProteinEBM (energy) OR AF2Rank, per --scorer
  4. Optionally refines ProteinEBM top-k with AF2Rank (run_af2rank_prediction.py)
  5. Runs central per-protein analysis (proteina_analysis.py)
  6. Collects best predictions and writes prediction_summary.{csv,json}
  7. Generates cross-protein plots (only when --cif_dir is supplied)
  8. Plots the pTM distribution histogram

Ground-truth analysis turns on when --cif_dir is supplied. Without --cif_dir
the pipeline runs in pure prediction mode: no GT comparisons, no cross-protein
plots.

Usage (prediction mode, no GT):
    python run_prediction_pipeline.py \\
        --input sequences.fasta --inference_config <config> \\
        --output_dir /path/to/output --num_gpus 4

Usage (evaluation mode, with GT):
    python run_prediction_pipeline.py \\
        --dataset_file data.csv --id_col pdb \\
        --cif_dir /path/to/cif --tms_col tms_single \\
        --inference_config <config> --output_dir /path/to/output --num_gpus 4

The AF2Rank top-k refinement step (run_af2rank_prediction.py) is OpenFold-only;
--af2rank_backend is honoured by the dedicated AF2Rank scoring step
(parallel_af2rank_scoring.py) when --scorer=af2rank.
"""

import argparse
import csv
import io
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from proteinfoundation.prediction_pipeline.pipeline_cli_utils import parallel_incremental_filter_args
from proteinfoundation.prediction_pipeline.sharding_utils import (
    add_shard_args,
    build_shard_cli_args,
    lengths_from_csv,
    resolve_shard_args,
    shard_proteins,
    wait_for_completion,
    wait_for_step,
)
from proteinfoundation.prediction_pipeline.protein_tar_utils import (
    ensure_protein_tar,
    pack_protein_dirs,
    protein_glob_members,
    read_protein_text,
    restore_selected_protein_dirs,
)
from proteinfoundation.prediction_pipeline.input_parser import build_discontinuous_pt, build_segment_pt_worker, create_pt_files, create_working_csv, parse_input
from proteinfoundation.prediction_pipeline.cif_to_pt_converter import convert_from_csv, sequence_to_pt_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import proteinfoundation.prediction_pipeline as _pf_pp

# prediction_pipeline lives inside the installed proteinfoundation package
PRED_PIPELINE_DIR = os.path.dirname(_pf_pp.__file__)
# Assume the script is always run from the proteina root directory
PROTEINA_BASE_DIR = os.getcwd()


def _split_shard_args(shard_args: list | None) -> tuple[int | None, int | None, list]:
    """Inverse of sharding_utils.build_shard_cli_args. Returns
    (external_shard_index, external_num_shards, passthrough) where passthrough
    preserves --len_col and any other unrecognized flags."""
    if not shard_args:
        return None, None, []
    ext_idx, ext_n, passthrough = None, None, []
    it = iter(shard_args)
    for tok in it:
        if tok == "--shard_index":
            ext_idx = int(next(it))
        elif tok == "--num_shards":
            ext_n = int(next(it))
        else:
            passthrough.append(tok)
    return ext_idx, ext_n, passthrough


def _inference_base(inference_config):
    """Build the inference base dir, optionally appending a conditioning-mode
    subdir from the PROTEINA_CONDITIONING_MODE env var. Mirrors the layout
    written by parallel_proteina_inference.py's --conditioning_mode flag:
        PROTEINA_CONDITIONING_MODE=seq      -> inference/{config}/seq_cond/
        PROTEINA_CONDITIONING_MODE=seq_cath -> inference/{config}/seq_cath_cond/
        PROTEINA_CONDITIONING_MODE=legacy   -> inference/{config}/legacy/
        (unset / other)                     -> inference/{config}/
    """
    parts = [PROTEINA_BASE_DIR, "inference", inference_config]
    # "legacy" mirrors parallel_proteina_inference._conditioning_label(None) so the
    # analysis can read the un-namespaced legacy layout when inference wrote there.
    label = {"seq": "seq_cond", "seq_cath": "seq_cath_cond", "legacy": "legacy"}.get(
        os.environ.get("PROTEINA_CONDITIONING_MODE", ""), ""
    )
    # Segment (joint-discontinuous) runs sample DIFFERENT PDBs than whole-chain for
    # the same conditioning, so they live under a distinct <mode_base> subdir.
    if label == "seq_cond" and os.environ.get("PROTEINA_SEGMENT_MODE", "") == "joint":
        label = "seq_cond_segment"
    if label:
        parts.append(label)
    return os.path.join(*parts)


def _available_gpu_count(default: int = 1) -> int:
    """Return number of visible GPUs via nvidia-smi, or `default` if detection fails."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode()
        return out.count("\n")
    except Exception:
        return default


def _tar_cli_args(enabled: bool) -> list[str]:
    return ["--tar_protein_dirs"] if enabled else ["--no-tar_protein_dirs"]


def _dynamic_cli_args(enabled: bool, progress_check_workers: int | None) -> list[str]:
    args = ["--dynamic_resharding"] if enabled else ["--no-dynamic_resharding"]
    if progress_check_workers is not None:
        args.extend(["--progress_check_workers", str(progress_check_workers)])
    return args


def _cleanup_shard_sentinels(output_dir: Path, num_shards: int) -> None:
    removed = 0
    for shard_idx in range(num_shards):
        sentinel = output_dir / f".shard_{shard_idx}_of_{num_shards}_complete"
        if sentinel.exists():
            sentinel.unlink()
            removed += 1
    for sentinel in output_dir.glob(".step_*_shard_*_of_*_complete"):
        sentinel.unlink()
        removed += 1
    logger.info("Cleaned up %d shard completion sentinel file(s).", removed)


def run_with_conda_env(env_name: str, command_list: list, cwd: str | None = None, direct_python: bool = False) -> bool:
    """Run a command. If direct_python, use current Python; else use shell script wrappers."""
    if direct_python:
        cmd = [sys.executable] + command_list[1:]
        logger.info(f"Running with current Python: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=cwd, check=False)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to run command: {e}")
            return False

    wrapper_map = {
        "proteina": os.path.join(PRED_PIPELINE_DIR, "run_with_proteina_env.sh"),
        "colabdesign": os.path.join(PRED_PIPELINE_DIR, "run_with_colabdesign_env.sh"),
        "proteinebm": os.path.join(PRED_PIPELINE_DIR, "run_with_proteinebm_env.sh"),
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

def _wait_for_pts(processed_dir, ids, label="", poll=15, timeout=7200):
    """Non-builder shards wait until shard 0 has built the PT files (shared processed/).
    Waits for >=99% of ids to have a {id}.pt (tolerates a few skips), or timeout."""
    import time as _time
    need = max(1, int(0.99 * len(ids)))
    waited = 0
    while waited < timeout:
        have = sum(1 for i in ids if os.path.exists(os.path.join(processed_dir, f"{i}.pt")))
        if have >= need:
            logger.info(f"_wait_for_pts({label}): {have}/{len(ids)} PTs present, proceeding")
            return
        logger.info(f"_wait_for_pts({label}): {have}/{len(ids)} PTs present, waiting...")
        _time.sleep(poll)
        waited += poll
    logger.warning(f"_wait_for_pts({label}): timeout after {timeout}s; proceeding with whatever PTs exist")


def step_build_segment_pts(dataset_file, id_col, sequence_col, segments_col, segment_min_len, workers=1):
    """Build per-protein PTs from a pre-annotated CSV (sequence + ordered_segments).
    Proteins with >=1 ordered segment >= segment_min_len -> discontinuous (joint) PT.
    Proteins with no qualifying segment -> WHOLE-CHAIN fallback PT (experimentally ordered
    but AIUPred-disordered: small folds / structured-upon-binding). The af2rank step's
    per-protein segment lookup returns None for these -> whole-chain scoring (no change there).
    `workers` parallelizes the (CPU-bound, side-effect-free, skip-existing) build.
    Returns all built/skipped ids (segment + fallback) so every protein is sampled."""
    data_path = os.environ.get("DATA_PATH", os.path.join(PROTEINA_BASE_DIR, "data"))
    processed_dir = os.path.join(data_path, "pdb_train", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    df = pd.read_csv(dataset_file)
    df.columns = df.columns.str.strip()
    for col in (sequence_col, segments_col):
        if col not in df.columns:
            raise KeyError(f"segment_mode=joint requires column '{col}' in {dataset_file}; columns={list(df.columns)}")
    tasks = [
        (str(row[id_col]).strip(), row[sequence_col], row[segments_col], segment_min_len,
         os.path.join(processed_dir, f"{str(row[id_col]).strip()}.pt"))
        for _, row in df.iterrows()
    ]
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(build_segment_pt_worker, tasks))
    else:
        results = [build_segment_pt_worker(t) for t in tasks]
    built = [pid for pid, kind in results if kind in ("segment", "fallback", "skip")]
    n_seg = sum(1 for _, k in results if k == "segment")
    n_fb = sum(1 for _, k in results if k == "fallback")
    n_skip = sum(1 for _, k in results if k == "skip")
    n_bad = sum(1 for _, k in results if k == "bad")
    logger.info("Segment PTs: %d built/present (%d segment, %d fallback, %d pre-existing), %d bad-row skipped in %s (workers=%d)"
                % (len(built), n_seg, n_fb, n_skip, n_bad, processed_dir, workers))
    return built


def step_parse_input(input_file: str, id_col: str, sequence_col: str, output_dir: str):
    """Parse input file, create PT files, and write working CSV."""
    df = parse_input(input_file, id_col=id_col, sequence_col=sequence_col)
    create_pt_files(df)
    working_csv = create_working_csv(df, os.path.join(output_dir, "working_proteins.csv"))
    return df, working_csv


# ── Step 2: Proteina inference ────────────────────────────────────────────────

def step_proteina_inference(
    csv_file: str,
    csv_col: str,
    inference_config: str,
    num_gpus: int,
    cif_dir: str | None = None,
    skip_pt_conversion: bool = True,
    usalign_path: str | None = None,
    proteina_force_compile: bool = True,
    shard_args: list | None = None,
    direct_python: bool = False,
    rerun: bool = False,
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
) -> bool:
    """Run Proteina inference on all proteins.

    When skip_pt_conversion=True (default), PT files are assumed to already
    exist in DATA_PATH/pdb_train/processed/. This is the --input mode flow.
    When skip_pt_conversion=False, --cif_dir is required and the inference
    script converts CIF → PT before inference.
    """
    logger.info("Starting Proteina inference...")
    cmd = [
        "python", os.path.join(PRED_PIPELINE_DIR, "parallel_proteina_inference.py"),
        "--csv_file", csv_file,
        "--csv_col", csv_col,
        "--inference_config", inference_config,
        "--num_gpus", str(num_gpus),
    ]
    if skip_pt_conversion:
        cmd.append("--skip_pt_conversion")
    else:
        if not cif_dir:
            logger.error("step_proteina_inference: --cif_dir required when skip_pt_conversion=False")
            return False
        cmd.extend(["--cif_dir", cif_dir])
    if usalign_path:
        cmd.extend(["--usalign_path", usalign_path])
    if not rerun:
        cmd.append("--skip_existing")
    if proteina_force_compile:
        cmd.append("--force_compile")
    # Propagate PROTEINA_CONDITIONING_MODE env var to --conditioning_mode flag.
    # parallel_proteina_inference.py uses this to load per-protein cath_codes
    # from the CSV and route them into inference.py's cath conditioning path.
    _cm = os.environ.get("PROTEINA_CONDITIONING_MODE", "").strip()
    if _cm in ("seq", "seq_cath"):
        cmd.extend(["--conditioning_mode", _cm])
    # Force subprocess-per-protein mode. The in_process ProcessPoolExecutor
    # path sets CUDA_VISIBLE_DEVICES via worker_init, but by then torch +
    # deepspeed have already initialized CUDA seeing all GPUs in the child
    # process (because module-level imports run before the initializer), so
    # all workers default to cuda:0. Subprocess mode gives each protein a
    # fresh child with CUDA_VISIBLE_DEVICES set in env at exec time -- proper
    # per-GPU isolation.
    cmd.append("--no-in_process")
    if shard_args:
        cmd.extend(shard_args)
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))
    return run_with_conda_env("proteina", cmd, direct_python=direct_python)


# ── Step 3: ProteinEBM scoring ────────────────────────────────────────────────

def step_proteinebm_scoring(
    csv_file: str,
    csv_col: str,
    inference_config: str,
    num_gpus: int,
    proteinebm_config: str,
    proteinebm_checkpoint: str,
    cif_dir: str | None = None,
    proteinebm_t: float = 0.05,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    proteinebm_batch_size: int = 32,
    proteinebm_template_self_condition: bool = True,
    num_workers: int | None = None,
    shard_args: list | None = None,
    direct_python: bool = False,
    rerun: bool = False,
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
) -> bool:
    """Run ProteinEBM scoring. When cif_dir is given, ground-truth TM-score
    columns are added to the per-protein scores CSVs."""
    logger.info("Starting ProteinEBM scoring...")
    cmd = [
        "python", os.path.join(PRED_PIPELINE_DIR, "parallel_proteinebm_scoring.py"),
        "--csv_file", csv_file,
        "--csv_col", csv_col,
        "--inference_config", inference_config,
        "--num_gpus", str(num_gpus),
        *parallel_incremental_filter_args(not rerun),
        "--proteinebm_config", proteinebm_config,
        "--proteinebm_checkpoint", proteinebm_checkpoint,
        "--proteinebm_t", str(proteinebm_t),
        "--proteinebm_analysis_subdir", proteinebm_analysis_subdir,
        "--proteinebm_batch_size", str(proteinebm_batch_size),
    ]
    if cif_dir:
        cmd.extend(["--cif_dir", cif_dir])
    if num_workers is not None:
        cmd.extend(["--num_workers", str(num_workers)])
    if not proteinebm_template_self_condition:
        cmd.append("--no-proteinebm_template_self_condition")
    if direct_python:
        cmd.append("--direct_python")
    if shard_args:
        cmd.extend(shard_args)
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))
    return run_with_conda_env("proteinebm", cmd, direct_python=direct_python)


# ── Step 3 (alt): AF2Rank scoring (when --scorer=af2rank) ────────────────────

def step_af2rank_scoring(
    csv_file: str,
    csv_col: str,
    cif_dir: str,
    inference_config: str,
    num_gpus: int,
    recycles: int = 3,
    regenerate_plots: bool = False,
    backend: str = "colabdesign",
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = True,
    use_cuequivariance_multiplicative_update: bool = True,
    shard_args: list | None = None,
    direct_python: bool = False,
    rerun: bool = False,
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
) -> bool:
    """Run AF2Rank scoring (parallel_af2rank_scoring.py). Requires --cif_dir
    because AF2Rank always needs a reference structure for scoring."""
    logger.info(f"Starting AF2Rank scoring (backend={backend})...")
    cmd = [
        "python", os.path.join(PRED_PIPELINE_DIR, "parallel_af2rank_scoring.py"),
        "--csv_file", csv_file,
        "--csv_col", csv_col,
        "--cif_dir", cif_dir,
        "--inference_config", inference_config,
        "--num_gpus", str(num_gpus),
        "--recycles", str(recycles),
        *parallel_incremental_filter_args(not rerun),
        "--af2rank_backend", backend,
    ]
    if regenerate_plots:
        cmd.append("--regenerate_plots")
    if backend == "openfold":
        if not use_deepspeed_evoformer_attention:
            cmd.append("--no-use_deepspeed_evoformer_attention")
        if not use_cuequivariance_attention:
            cmd.append("--no-use_cuequivariance_attention")
        if not use_cuequivariance_multiplicative_update:
            cmd.append("--no-use_cuequivariance_multiplicative_update")
    if shard_args:
        cmd.extend(shard_args)
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))
    env_name = "proteina" if backend == "openfold" else "colabdesign"
    return run_with_conda_env(env_name, cmd, direct_python=direct_python)


# ── Step 4: AF2Rank on ProteinEBM top-k ──────────────────────────────────────

def step_af2rank_topk(
    inference_config: str,
    af2rank_top_k: int,
    recycles: int,
    num_gpus: int,
    csv_file: str,
    csv_col: str = "id",
    cif_dir: str | None = None,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = True,
    use_cuequivariance_multiplicative_update: bool = True,
    shard_args: list | None = None,
    direct_python: bool = False,
    filter_existing: bool = True,
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
    force_regenerate_topk_summary: bool = False,
    segment_mode: str = "off",
    segments_col: str = "ordered_segments",
    segment_min_len: int = 50,
    mask_inter_segment: bool = True,
    af2rank_subdir: str = "af2rank_on_proteinebm_top_k",
) -> bool:
    """Run AF2Rank on ProteinEBM top-k templates.

    Uses run_af2rank_prediction.py which initialises each AF2Rank model variant
    exactly once across all proteins (instead of once per protein), and reads
    top-k templates directly from the ProteinEBM scores CSV without requiring
    a summary JSON file. When --cif_dir is given, ground-truth TM-score columns
    are added to the output CSVs.
    """
    logger.info(f"Starting AF2Rank on ProteinEBM top-{af2rank_top_k} ...")
    inference_dir = _inference_base(inference_config)
    cmd = [
        "python", os.path.join(PRED_PIPELINE_DIR, "run_af2rank_prediction.py"),
        "--inference_dir", inference_dir,
        "--csv_file", csv_file,
        "--csv_col", csv_col,
        "--top_k", str(af2rank_top_k),
        "--recycles", str(recycles),
        "--proteinebm_analysis_subdir", proteinebm_analysis_subdir,
        "--af2rank_subdir", af2rank_subdir,
    ]
    if segment_mode == "joint":
        cmd.extend(["--segment_mode", "joint", "--segments_col", segments_col,
                    "--segment_min_len", str(segment_min_len)])
        cmd.append("--mask_inter_segment" if mask_inter_segment else "--no-mask_inter_segment")
    if cif_dir:
        cmd.extend(["--cif_dir", cif_dir])
    if filter_existing:
        cmd.append("--filter_existing")
    else:
        cmd.append("--no-filter_existing")
    if force_regenerate_topk_summary:
        cmd.append("--force_regenerate_topk_summary")
    if not use_deepspeed_evoformer_attention:
        cmd.append("--no-use_deepspeed_evoformer_attention")
    if not use_cuequivariance_attention:
        cmd.append("--no-use_cuequivariance_attention")
    if not use_cuequivariance_multiplicative_update:
        cmd.append("--no-use_cuequivariance_multiplicative_update")
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))

    requested_gpus = num_gpus if num_gpus and num_gpus > 0 else 1
    detected_gpus = _available_gpu_count(default=requested_gpus)
    if detected_gpus < requested_gpus:
        logger.warning(
            f"Requested {requested_gpus} GPUs but only {detected_gpus} available; using {detected_gpus}."
        )
    effective_num_gpus = min(requested_gpus, detected_gpus)

    if direct_python:
        python_cmd = [sys.executable]
    else:
        python_cmd = [os.path.join(PRED_PIPELINE_DIR, "run_with_proteina_env.sh"), "python"]
    base_cmd = python_cmd + cmd[1:]

    ext_idx, ext_n, passthrough = _split_shard_args(shard_args)
    base_ext_idx = ext_idx if ext_idx is not None else 0
    base_ext_n = ext_n if ext_n is not None else 1

    if effective_num_gpus == 1 and ext_idx is None and ext_n is None:
        sub_cmds = [(0, list(base_cmd) + passthrough)]
    else:
        sub_cmds = []
        for gpu_id in range(effective_num_gpus):
            eff_index = base_ext_idx * effective_num_gpus + gpu_id
            eff_shards = base_ext_n * effective_num_gpus
            sub_cmd = list(base_cmd) + passthrough + [
                "--shard_index", str(eff_index),
                "--num_shards", str(eff_shards),
            ]
            sub_cmds.append((gpu_id, sub_cmd))

    procs: list[tuple[int, subprocess.Popen]] = []
    for gpu_id, sub_cmd in sub_cmds:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if effective_num_gpus > 1 or ext_idx is not None:
            logger.info(
                f"GPU {gpu_id}: launching AF2Rank top-k worker "
                f"(shard {base_ext_idx * effective_num_gpus + gpu_id}/{base_ext_n * effective_num_gpus})"
            )
        else:
            logger.info(f"GPU {gpu_id}: launching AF2Rank top-k worker (single-GPU)")
        proc = subprocess.Popen(sub_cmd, env=env)
        procs.append((gpu_id, proc))

    failed: list[int] = []
    try:
        for gpu_id, proc in procs:
            rc = proc.wait()
            if rc != 0:
                logger.error(f"GPU {gpu_id} AF2Rank top-k worker exited with code {rc}")
                failed.append(gpu_id)
            else:
                logger.info(f"GPU {gpu_id} AF2Rank top-k worker finished successfully")
    except KeyboardInterrupt:
        for _, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        raise

    return not failed


def step_central_analysis(
    inference_config: str,
    csv_file: str,
    csv_col: str = "id",
    cif_dir: str | None = None,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    num_workers: int | None = None,
    shard_args: list | None = None,
    direct_python: bool = False,
    rerun: bool = False,
    skip_diversity: bool = False,
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
    af2rank_subdir: str = "af2rank_on_proteinebm_top_k",
) -> bool:
    """Run centralized TM analysis after scorer outputs and prediction PDBs exist.
    When cif_dir is provided, GT TM-score columns are added to the per-protein
    enriched CSVs."""
    inference_dir = _inference_base(inference_config)
    cmd = [
        "python",
        os.path.join(PRED_PIPELINE_DIR, "proteina_analysis.py"),
        "--inference_dir",
        inference_dir,
        "--csv_file",
        csv_file,
        "--csv_col",
        csv_col,
        "--proteinebm_analysis_subdir",
        proteinebm_analysis_subdir,
        "--af2rank_subdir",
        af2rank_subdir,
    ]
    if cif_dir:
        cmd.extend(["--cif_dir", cif_dir])
    if num_workers is not None:
        cmd.extend(["--num_workers", str(num_workers)])
    if shard_args:
        cmd.extend(shard_args)
    if rerun:
        cmd.append("--rerun")
    if skip_diversity:
        cmd.append("--skip_diversity")
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))
    return run_with_conda_env("proteina", cmd, direct_python=direct_python)


# ── Step 5b: Cross-protein plots (only when --cif_dir is supplied) ───────────

def step_cross_protein_plots(
    inference_config: str,
    output_dir: str,
    scorer: str,
    dataset_file: str,
    id_col: str,
    tms_col: str,
    af2rank_top_k: int,
    proteinebm_plot_mode: str = "tm",
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    direct_python: bool = False,
    skip_af2rank_on_top_k: bool = False,
    af2rank_subdir: str = "af2rank_on_proteinebm_top_k",
) -> bool:
    """Generate cross-protein plots (TM-vs-pTM / TM-vs-energy scatters).

    Always requires ground-truth (a dataset_file with tms_col). Runs once for
    the primary scorer; if scorer=proteinebm and af2rank_top_k > 0, runs a
    second time for the af2rank_on_proteinebm_topk plots.
    """
    inference_dir = _inference_base(inference_config)

    def _one(scorer_name: str) -> bool:
        cmd = [
            "python",
            os.path.join(PRED_PIPELINE_DIR, "generate_cross_protein_plots.py"),
            "--inference_dir", inference_dir,
            "--output_dir", output_dir,
            "--scorer", scorer_name,
            "--dataset_file", dataset_file,
            "--id_col", id_col,
            "--tms_col", tms_col,
            "--proteinebm_analysis_subdir", proteinebm_analysis_subdir,
        ]
        if scorer_name == "proteinebm":
            cmd.extend(["--proteinebm_plot_mode", proteinebm_plot_mode])
        if scorer_name == "af2rank_on_proteinebm_topk":
            cmd.extend(["--af2rank_top_k", str(int(af2rank_top_k))])
            cmd.extend(["--af2rank_subdir", af2rank_subdir])
        return run_with_conda_env("proteina", cmd, cwd=PRED_PIPELINE_DIR, direct_python=direct_python)

    ok = _one(scorer)
    if not ok:
        logger.error("Cross-protein plotting failed (primary scorer)")
        return False
    if scorer == "proteinebm" and int(af2rank_top_k) > 0 and not skip_af2rank_on_top_k:
        ok2 = _one("af2rank_on_proteinebm_topk")
        if not ok2:
            logger.warning("Cross-protein plotting skipped/failed (af2rank_on_proteinebm_topk)")
    return True


# ── Step 5: Collect results ───────────────────────────────────────────────────

def step_collect_results(
    protein_ids: list,
    inference_config: str,
    output_dir: str,
    ptm_cutoff: float,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    tar_protein_dirs: bool = False,
    af2rank_subdir: str = "af2rank_on_proteinebm_top_k",
) -> list:
    """
    Read AF2Rank scores, select best structure per protein, copy to output.

    Returns list of per-protein result dicts.
    """
    logger.info("Collecting results...")
    inference_base = _inference_base(inference_config)
    best_templates_dir = os.path.join(output_dir, "best_templates")
    best_predictions_dir = os.path.join(output_dir, "best_predictions")
    os.makedirs(best_templates_dir, exist_ok=True)
    os.makedirs(best_predictions_dir, exist_ok=True)

    results = []
    restored_for_copy = []

    for protein_id in protein_ids:
        protein_dir = Path(inference_base) / protein_id
        topk_dir = protein_dir / af2rank_subdir
        m1_csv = topk_dir / "af2rank_analysis" / f"af2rank_scores_{protein_id}.csv"
        m2_csv = topk_dir / "af2rank_analysis_model_2_ptm" / f"af2rank_scores_{protein_id}.csv"

        # Count generated structures
        if tar_protein_dirs:
            num_generated = len(protein_glob_members(inference_base, protein_id, f"{protein_id}_*.pdb"))
        else:
            num_generated = len(list(protein_dir.glob(f"{protein_id}_*.pdb")))

        # Read sequence length from PT file
        data_path = os.environ.get("DATA_PATH", os.path.join(PROTEINA_BASE_DIR, "data"))
        pt_path = os.path.join(data_path, "pdb_train", "processed", f"{protein_id}.pt")
        try:
            pt = torch.load(pt_path, weights_only=False, map_location="cpu")
            seq_len = len(pt.residue_type)
        except Exception:
            seq_len = 0

        if tar_protein_dirs:
            m1_text = read_protein_text(
                inference_base,
                protein_id,
                Path(af2rank_subdir) / "af2rank_analysis" / f"af2rank_scores_{protein_id}.csv",
            )
            m2_text = read_protein_text(
                inference_base,
                protein_id,
                Path(af2rank_subdir) / "af2rank_analysis_model_2_ptm" / f"af2rank_scores_{protein_id}.csv",
            )
        else:
            m1_text = m1_csv.read_text() if m1_csv.exists() else None
            m2_text = m2_csv.read_text() if m2_csv.exists() else None

        if m1_text is None or m2_text is None:
            logger.warning(f"AF2Rank scores not found for {protein_id}, skipping")
            results.append({
                "protein_id": protein_id,
                "sequence_length": seq_len,
                "num_generated": num_generated,
                "best_ptm": float("nan"),
                "best_plddt": float("nan"),
                "best_energy": float("nan"),
                "best_proteinebm_ptm": float("nan"),
                "best_proteinebm_mean_pae": float("nan"),
                "best_ref_pred_tm": float("nan"),
                "best_template": "",
                "best_prediction": "",
                "passes_cutoff": False,
            })
            continue

        m1_df = pd.read_csv(io.StringIO(m1_text))
        m2_df = pd.read_csv(io.StringIO(m2_text))

        # Merge across models: take min pTM per structure (conservative).
        # Also bring in tm_ref_pred (GT TM-score for AF2Rank's prediction) when
        # present — populated by proteina_analysis.py --cif_dir.
        m1_cols = ["structure_file", "ptm", "plddt"]
        if "tm_ref_pred" in m1_df.columns:
            m1_cols.append("tm_ref_pred")
        m2_cols = ["structure_file", "ptm"]
        if "tm_ref_pred" in m2_df.columns:
            m2_cols.append("tm_ref_pred")
        merged = m1_df[m1_cols].merge(
            m2_df[m2_cols], on="structure_file", suffixes=("_m1", "_m2"),
        )
        merged["min_ptm"] = merged[["ptm_m1", "ptm_m2"]].min(axis=1)

        # Also get energy / ProteinEBM pTM / ProteinEBM mean pAE from ProteinEBM scores
        ebm_csv = protein_dir / proteinebm_analysis_subdir / f"proteinebm_scores_{protein_id}.csv"
        energy_map: dict = {}
        proteinebm_ptm_map: dict = {}
        proteinebm_mean_pae_map: dict = {}
        proteinebm_best_ptm = float("nan")
        proteinebm_best_mean_pae = float("nan")
        ebm_text = read_protein_text(
            inference_base,
            protein_id,
            Path(proteinebm_analysis_subdir) / f"proteinebm_scores_{protein_id}.csv",
        ) if tar_protein_dirs else (ebm_csv.read_text() if ebm_csv.exists() else None)
        if ebm_text is not None:
            ebm_df = pd.read_csv(io.StringIO(ebm_text))
            energy_map = dict(zip(ebm_df["structure_file"].astype(str), ebm_df["energy"].astype(float)))
            if "ptm" in ebm_df.columns:
                proteinebm_ptm_map = dict(zip(
                    ebm_df["structure_file"].astype(str),
                    pd.to_numeric(ebm_df["ptm"], errors="coerce").astype(float),
                ))
                ptm_values_all = pd.to_numeric(ebm_df["ptm"], errors="coerce").dropna()
                if len(ptm_values_all) > 0:
                    proteinebm_best_ptm = float(ptm_values_all.max())
            if "mean_pae" in ebm_df.columns:
                proteinebm_mean_pae_map = dict(zip(
                    ebm_df["structure_file"].astype(str),
                    pd.to_numeric(ebm_df["mean_pae"], errors="coerce").astype(float),
                ))
                pae_values_all = pd.to_numeric(ebm_df["mean_pae"], errors="coerce").dropna()
                if len(pae_values_all) > 0:
                    proteinebm_best_mean_pae = float(pae_values_all.min())

        merged["energy"] = merged["structure_file"].map(energy_map).fillna(float("nan"))

        if merged.empty:
            results.append({
                "protein_id": protein_id,
                "sequence_length": seq_len,
                "num_generated": num_generated,
                "best_ptm": float("nan"),
                "best_plddt": float("nan"),
                "best_energy": float("nan"),
                "best_proteinebm_ptm": proteinebm_best_ptm,
                "best_proteinebm_mean_pae": proteinebm_best_mean_pae,
                "best_ref_pred_tm": float("nan"),
                "best_template": "",
                "best_prediction": "",
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
        # Match ref_pred_TM to the model whose pTM equals min_ptm, so
        # (best_ptm, best_ref_pred_tm) come from the same AF2Rank model.
        best_ref_pred_tm = float("nan")
        if "tm_ref_pred_m1" in best_row.index and "tm_ref_pred_m2" in best_row.index:
            ptm_m1_val = float(best_row.get("ptm_m1", float("inf")))
            ptm_m2_val = float(best_row.get("ptm_m2", float("inf")))
            use_m1 = ptm_m1_val <= ptm_m2_val
            chosen = best_row["tm_ref_pred_m1"] if use_m1 else best_row["tm_ref_pred_m2"]
            try:
                best_ref_pred_tm = float(chosen)
            except (TypeError, ValueError):
                best_ref_pred_tm = float("nan")

        # ── Save best cg2all template ──────────────────────────────────────────
        if tar_protein_dirs and not protein_dir.exists():
            stats = restore_selected_protein_dirs(inference_base, [protein_id])
            logger.info("tar_restore collect_results copy %s: %s", protein_id, stats)
            if protein_id not in restored_for_copy:
                restored_for_copy.append(protein_id)
        elif tar_protein_dirs and protein_id not in restored_for_copy:
            restored_for_copy.append(protein_id)
        best_stem = Path(best_file).stem
        cg2all_pdb = topk_dir / "cg2all_topk_structures" / f"{best_stem}_allatom.pdb"
        dest_template = os.path.join(best_templates_dir, f"{protein_id}.pdb")
        if cg2all_pdb.exists():
            shutil.copy2(str(cg2all_pdb), dest_template)
        else:
            # Fallback: staged CA template or raw inference dir
            staged_pdb = topk_dir / "staged_topk_templates" / best_file
            fallback = protein_dir / best_file
            if staged_pdb.exists():
                shutil.copy2(str(staged_pdb), dest_template)
                logger.warning(f"{protein_id}: cg2all template not found, fell back to staged CA template")
            elif fallback.exists():
                shutil.copy2(str(fallback), dest_template)
                logger.warning(f"{protein_id}: cg2all template not found, fell back to raw inference PDB")
            else:
                logger.warning(f"{protein_id}: could not find any template for {best_file}")
                dest_template = ""

        # ── Save best AF2Rank prediction ──────────────────────────────────────
        # Pick the model with the higher individual pTM for the best template
        ptm_m1 = float(best_row.get("ptm_m1", float("nan")))
        ptm_m2 = float(best_row.get("ptm_m2", float("nan")))
        use_m1 = (not math.isnan(ptm_m1)) and (math.isnan(ptm_m2) or ptm_m1 >= ptm_m2)
        if use_m1:
            pred_src = topk_dir / "af2rank_analysis" / "predicted_structures" / best_file
        else:
            pred_src = topk_dir / "af2rank_analysis_model_2_ptm" / "predicted_structures" / best_file
        dest_prediction = os.path.join(best_predictions_dir, f"{protein_id}.pdb")
        if pred_src.exists():
            shutil.copy2(str(pred_src), dest_prediction)
        else:
            logger.warning(f"{protein_id}: AF2Rank prediction not found at {pred_src}")
            dest_prediction = ""

        results.append({
            "protein_id": protein_id,
            "sequence_length": seq_len,
            "num_generated": num_generated,
            "best_ptm": best_ptm,
            "best_plddt": best_plddt,
            "best_energy": best_energy,
            "best_proteinebm_ptm": proteinebm_best_ptm,
            "best_proteinebm_mean_pae": proteinebm_best_mean_pae,
            "best_ref_pred_tm": best_ref_pred_tm,
            "best_template": os.path.basename(dest_template) if dest_template else "",
            "best_prediction": os.path.basename(dest_prediction) if dest_prediction else "",
            "passes_cutoff": passes,
        })

    if tar_protein_dirs and restored_for_copy:
        stats = pack_protein_dirs(inference_base, restored_for_copy, delete_after=True)
        logger.info("tar_pack collect_results copy: %s", stats)

    # Write prediction_summary.csv
    summary_csv_path = os.path.join(output_dir, "prediction_summary.csv")
    fieldnames = [
        "protein_id", "sequence_length", "num_generated",
        "best_ptm", "best_plddt", "best_energy",
        "best_proteinebm_ptm", "best_proteinebm_mean_pae",
        "best_ref_pred_tm",
        "best_template", "best_prediction", "passes_cutoff",
    ]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Summary CSV written to {summary_csv_path}")

    # Write prediction_summary.json
    def _finite(values: list) -> list:
        return [v for v in values if not (isinstance(v, float) and math.isnan(v))]

    ptm_values = _finite([r["best_ptm"] for r in results])
    proteinebm_ptm_values = _finite([r.get("best_proteinebm_ptm", float("nan")) for r in results])
    proteinebm_mean_pae_values = _finite([r.get("best_proteinebm_mean_pae", float("nan")) for r in results])
    ref_pred_tm_values = _finite([r.get("best_ref_pred_tm", float("nan")) for r in results])
    num_passing = sum(1 for r in results if r["passes_cutoff"])

    def _stats(values: list):
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # Yield at the configured cutoff: among proteins with pTM >= cutoff,
    # what fraction have ref_pred_TM >= cutoff (same threshold τ on both sides)?
    # Both vals must be present (GT mode).
    calibration_pairs = [
        (float(r["best_ptm"]), float(r.get("best_ref_pred_tm", float("nan"))))
        for r in results
        if not (isinstance(r["best_ptm"], float) and math.isnan(r["best_ptm"]))
        and not (isinstance(r.get("best_ref_pred_tm", float("nan")), float)
                 and math.isnan(r.get("best_ref_pred_tm", float("nan"))))
    ]
    n_pass_c = sum(1 for ptm, _ in calibration_pairs if ptm >= ptm_cutoff)
    n_cal_c = sum(1 for ptm, gt in calibration_pairs if ptm >= ptm_cutoff and gt >= ptm_cutoff)
    calibration_block = (
        {
            "ptm_cutoff": ptm_cutoff,
            "num_passing_cutoff": n_pass_c,
            "num_meeting_actual_threshold": n_cal_c,
            "fraction_meeting_actual_threshold": (n_cal_c / n_pass_c) if n_pass_c else None,
        }
        if calibration_pairs
        else None
    )

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
        "proteinebm_ptm": _stats(proteinebm_ptm_values),
        "proteinebm_mean_pae": _stats(proteinebm_mean_pae_values),
        "ref_pred_tm": _stats(ref_pred_tm_values),
        "calibration_at_cutoff": calibration_block,
    }

    # Add analysis pairwise TM metrics if available
    from proteinfoundation.prediction_pipeline.proteina_analysis import load_analysis_data
    inference_base = _inference_base(inference_config)
    analysis_data = load_analysis_data(inference_base)
    if analysis_data:
        div_medians = [v["median_tem_to_tem_tm"] for v in analysis_data.values() if v.get("median_tem_to_tem_tm") is not None]
        div_means = [v["mean_tem_to_tem_tm"] for v in analysis_data.values() if v.get("mean_tem_to_tem_tm") is not None]
        div_stds = [v["std_tem_to_tem_tm"] for v in analysis_data.values() if v.get("std_tem_to_tem_tm") is not None]
        analysis_stats = {"num_proteins_with_analysis": len(analysis_data)}
        if div_medians:
            analysis_stats.update({
                "mean_tem_to_tem_tm": {
                    "mean": float(np.mean(div_means)),
                    "median": float(np.median(div_means)),
                    "std": float(np.std(div_means)),
                },
                "std_tem_to_tem_tm": {
                    "mean": float(np.mean(div_stds)),
                    "median": float(np.median(div_stds)),
                    "std": float(np.std(div_stds)),
                },
                "median_tem_to_tem_tm": {
                    "mean": float(np.mean(div_medians)),
                    "median": float(np.median(div_medians)),
                    "std": float(np.std(div_medians)),
                },
                "num_proteins_with_diversity": len(div_medians),
            })
        summary_json["analysis_statistics"] = analysis_stats
        summary_json["diversity_statistics"] = analysis_stats
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


def step_plot_ptm_calibration_curve(results: list, output_dir: str, ptm_cutoff: float) -> None:
    """Calibration / yield curve: among proteins passing the pTM filter
    (best_ptm ≥ τ), what fraction actually achieve quality at or above the
    same threshold (best_ref_pred_tm ≥ τ)?

    Y(τ) = P(ref_pred_TM ≥ τ | pTM ≥ τ)

    Both thresholds are the **same** value τ — the curve answers "does the
    pTM filter buy me proteins that hit the corresponding actual-TM bar?".
    Skipped silently when no protein has both best_ptm and best_ref_pred_tm
    (no-GT mode). Writes ``ptm_calibration_curve.png`` in ``output_dir``.
    """
    pairs = [
        (float(r["best_ptm"]), float(r.get("best_ref_pred_tm", float("nan"))))
        for r in results
        if not (isinstance(r["best_ptm"], float) and math.isnan(r["best_ptm"]))
        and not (isinstance(r.get("best_ref_pred_tm", float("nan")), float)
                 and math.isnan(r.get("best_ref_pred_tm", float("nan"))))
    ]
    if not pairs:
        logger.info("Skipping calibration curve: no GT-aware best_ref_pred_tm available")
        return

    ptms = np.array([p for p, _ in pairs], dtype=float)
    gts = np.array([g for _, g in pairs], dtype=float)
    taus = np.linspace(0.0, 1.0, 101)
    fracs = np.full_like(taus, np.nan, dtype=float)
    n_passing = np.zeros_like(taus, dtype=int)
    for i, tau in enumerate(taus):
        mask = ptms >= tau
        n = int(mask.sum())
        n_passing[i] = n
        if n > 0:
            fracs[i] = float((gts[mask] >= tau).sum()) / n

    # Fraction at the configured cutoff (for the highlighted marker + annotation)
    mask_c = ptms >= ptm_cutoff
    n_pass_c = int(mask_c.sum())
    n_cal_c = int((gts[mask_c] >= ptm_cutoff).sum()) if n_pass_c else 0
    frac_c = (n_cal_c / n_pass_c) if n_pass_c else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, fracs, color="#4C72B0", linewidth=2,
            label="P(ref_pred_TM ≥ τ | pTM ≥ τ)")
    ax.axvline(ptm_cutoff, color="red", linestyle="--", linewidth=2,
               label=f"pTM cutoff = {ptm_cutoff}")
    if not math.isnan(frac_c):
        ax.plot([ptm_cutoff], [frac_c], "ro", markersize=8, zorder=5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("pTM cutoff τ", fontsize=12)
    ax.set_ylabel("Fraction with ref_pred_TM ≥ τ\n(among proteins with pTM ≥ τ)", fontsize=12)
    ax.set_title("Yield at threshold: actual TM ≥ τ given predicted pTM ≥ τ", fontsize=13)
    annotation = (
        f"At τ = {ptm_cutoff}:\n  {n_cal_c}/{n_pass_c} proteins ({frac_c:.0%}) have ref_pred_TM ≥ {ptm_cutoff}"
        if n_pass_c
        else f"At τ = {ptm_cutoff}: no proteins pass cutoff"
    )
    ax.annotate(annotation, xy=(0.04, 0.96), xycoords="axes fraction",
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ptm_calibration_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"pTM calibration curve saved to {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Unified CLI: prediction mode (--input) or evaluation mode (--dataset_file + --cif_dir)."""
    parser = argparse.ArgumentParser(description="Prediction Pipeline (unified prediction + evaluation driver)")

    # Input: --input (FASTA/CSV of sequences) XOR --dataset_file (CSV with pre-built PT files)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", default=None,
                             help="Input CSV/FASTA with protein sequences (prediction mode). "
                                  "Triggers PT-file creation via input_parser.py.")
    input_group.add_argument("--dataset_file", default=None,
                             help="Dataset CSV with pre-existing PT files (evaluation mode). "
                                  "Mutually exclusive with --input.")
    parser.add_argument("--id_col", default="id",
                        help="Column name for protein ID in --input or --dataset_file (default: id)")
    parser.add_argument("--sequence_col", default="sequence",
                        help="Column name for sequence in --input CSV (default: sequence, ignored for FASTA)")

    # Ground-truth mode: optional --cif_dir + --tms_col. When --cif_dir is set,
    # downstream steps add GT TM-score columns; when --tms_col is also set (with
    # --dataset_file), cross-protein plots are generated.
    parser.add_argument("--cif_dir", default=None,
                        help="Optional: directory of reference CIF files. Presence enables ground-truth "
                             "analysis (GT TM-score columns added to per-protein CSVs).")
    parser.add_argument("--tms_col", default=None,
                        help="Dataset column with reference TM-scores. Required (with --cif_dir + --dataset_file) "
                             "for cross-protein plot generation.")

    parser.add_argument("--scorer", choices=["af2rank", "proteinebm"], default="proteinebm",
                        help="Primary scoring backend (default: proteinebm). When proteinebm + "
                             "--af2rank_top_k>0, an AF2Rank top-k refinement step runs afterward.")
    parser.add_argument("--inference_config", required=True, help="Proteina inference configuration name")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--output_dir", required=True, help="Output directory for predictions and summary")
    parser.add_argument("--ptm_cutoff", type=float, default=0.7, help="pTM threshold for filtering (default: 0.7)")
    parser.add_argument(
        "--proteinebm_cross_protein_plot_mode",
        choices=["tm", "energy"],
        default="tm",
        help="When --scorer=proteinebm and cross-protein plots run, which mode to use (default: tm).",
    )
    parser.add_argument("--usalign_path", default=None,
                        help="Optional: path to USalign binary (passed to parallel_proteina_inference).")
    parser.add_argument(
        "--af2rank_top_k",
        "--top_k",
        dest="af2rank_top_k",
        type=int,
        default=5,
        help="Number of top ProteinEBM templates for AF2Rank (default: 5). Alias: --top_k.",
    )
    parser.add_argument(
        "--af2rank_backend",
        "--backend",
        dest="af2rank_backend",
        choices=["colabdesign", "openfold"],
        default="openfold",
        help="AF2Rank backend name (shared with full pipeline). The prediction AF2Rank step is OpenFold-only; "
        "only openfold is used. Alias: --backend.",
    )
    parser.add_argument("--use_deepspeed_evoformer_attention", action=argparse.BooleanOptionalAction, default=False,
                        help="Use DeepSpeed evoformer attention (openfold backend, default: False)")
    parser.add_argument("--use_cuequivariance_attention", action=argparse.BooleanOptionalAction, default=False,
                        help="Use cuEquivariance attention kernels (openfold backend, default: False)")
    parser.add_argument("--use_cuequivariance_multiplicative_update", action=argparse.BooleanOptionalAction, default=False,
                        help="Use cuEquivariance multiplicative update (openfold backend, default: False)")
    parser.add_argument("--recycles", type=int, default=3, help="AF2 recycles for AF2Rank (default: 3)")
    parser.add_argument("--proteinebm_config", default="/home/ubuntu/ProteinEBM/protein_ebm/config/pae_config.yaml",
                        help="Path to ProteinEBM config YAML (default: pae_config.yaml — adds PAE head for pTM/mean_pae)")
    parser.add_argument("--proteinebm_checkpoint", default="/home/ubuntu/ProteinEBM/weights/pae.ckpt",
                        help="Path to ProteinEBM checkpoint (default: pae.ckpt — same EBM trunk as v2_cathmd, plus a PAE head)")
    parser.add_argument("--proteinebm_t", type=float, default=0.05, help="Diffusion time for ProteinEBM (default: 0.05)")
    parser.add_argument("--proteinebm_analysis_subdir", default="proteinebm_v2_cathmd_analysis",
                        help="Per-protein subdir for ProteinEBM outputs")
    parser.add_argument(
        "--proteinebm_template_self_condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use template coordinates for self-conditioning (matches parallel_proteinebm_scoring).",
    )
    parser.add_argument("--proteinebm_batch_size", type=int, default=32,
                        help="Batch size for ProteinEBM inference (default: 32). Auto-reduces on OOM.")
    parser.add_argument(
        "--af2rank_topk_filter_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip AF2Rank top-k work when outputs already cover all desired templates (default: True).",
    )
    parser.add_argument("--skip_inference", action="store_true", help="Skip Proteina inference step")
    parser.add_argument("--skip_diversity", action="store_true",
                        help="Skip template-to-template diversity computation in central analysis")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip the central post-scoring TM analysis stage")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Max parallel CPU workers for central analysis and similar CPU-bound steps; "
             "default is clamped os.cpu_count() (1–64).",
    )
    parser.add_argument("--skip_scoring", action="store_true", help="Skip ProteinEBM scoring step")
    parser.add_argument(
        "--skip_af2rank_on_top_k",
        "--skip_af2rank",
        dest="skip_af2rank_on_top_k",
        action="store_true",
        help="Skip AF2Rank-on-ProteinEBM-top-k step. Alias: --skip_af2rank.",
    )
    parser.add_argument("--skip_cross_protein_plots", action="store_true",
                        help="Skip cross-protein plot generation (even when --cif_dir is set).")
    parser.add_argument("--skip_distribution_plot", action="store_true",
                        help="Skip pTM distribution histogram.")
    parser.add_argument("--skip_collect_results", action="store_true",
                        help="Skip prediction_summary.{csv,json} + best_templates/ + best_predictions/.")
    parser.add_argument("--regenerate_plots", action="store_true",
                        help="Regenerate AF2Rank plots even if scoring already completed (--scorer=af2rank)")
    parser.add_argument("--rerun_proteina", action="store_true",
                        help="Force re-run Proteina inference even if outputs already exist")
    parser.add_argument("--rerun_score", action="store_true",
                        help="Force re-run ProteinEBM scoring even if outputs already exist")
    parser.add_argument("--rerun_af2rank_on_top_k", action="store_true",
                        help="Force re-run AF2Rank on ProteinEBM top-k even if outputs already exist")
    parser.add_argument("--rerun_analysis", action="store_true",
                        help="Force re-run central analysis even if analysis summaries already exist")
    parser.add_argument(
        "--proteina_force_compile",
        "--force_compile",
        dest="proteina_force_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile for Proteina inference (default: True). Alias: --force_compile.",
    )
    add_shard_args(parser)
    parser.add_argument("--shard_poll_interval", type=int, default=60,
                        help="Seconds between polls when shard 0 waits (default: 60)")
    parser.add_argument("--shard_timeout", type=int, default=86400,
                        help="Max seconds for shard 0 to wait (default: 86400)")
    parser.add_argument("--direct_python", action="store_true",
                        help="Use current Python interpreter for subprocesses instead of shell script wrappers. "
                             "Useful on HPC where conda env activation is slow; requires all deps in current env.")
    parser.add_argument(
        "--tar_protein_dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-protein inference directories as uncompressed <protein_id>.tar archives (default: True).",
    )
    parser.add_argument(
        "--dynamic_resharding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At each step, filter global unfinished proteins before sharding to reduce idle shards (default: True).",
    )
    parser.add_argument(
        "--progress_check_workers",
        type=int,
        default=None,
        help="Thread workers for per-step progress checks (default: child script uses min(32, cpu_count * 4)).",
    )
    parser.add_argument('--segment_mode', choices=['off', 'joint'], default='off',
                        help='Joint-discontinuous segment mode (default: off = full-sequence/whole-chain).')
    parser.add_argument('--segments_col', default='ordered_segments',
                        help='CSV column with ordered_segments JSON (segment_mode=joint).')
    parser.add_argument('--segment_min_len', type=int, default=50,
                        help='Drop ordered segments shorter than this (segment_mode=joint, default 50).')
    parser.add_argument('--mask_inter_segment', action=argparse.BooleanOptionalAction, default=True,
                        help='Mask cross-segment template pairs in AF2Rank (segment_mode=joint, default True).')
    parser.add_argument('--af2rank_subdir', default=None,
                        help='Per-protein AF2Rank-on-top-k output subdir name. Default: auto '
                             '(af2rank_on_proteinebm_top_k for whole-chain; ..._mask / ..._nomask for '
                             'segment_mode=joint with/without --mask_inter_segment) so mask vs nomask '
                             'runs coexist under the same shared inference dir. Override to force a name.')
    parser.add_argument('--pt_build_workers', type=int, default=max(1, (os.cpu_count() or 8) - 2),
                        help='CPU workers for the up-front parallel PT-build pre-step (CIF->PT / '
                             'discontinuous PT). PTs are built before GPU inference so workers never '
                             'block on the CPU step. Default: cpu_count-2.')
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    # Resolve the per-protein AF2Rank-on-top-k output subdir. For segment mode the
    # mask and nomask cells share ONE inference + ProteinEBM pass and differ only in
    # the AF2Rank scoring step, so they must write to distinct subdirs to coexist.
    if args.af2rank_subdir:
        af2rank_subdir = args.af2rank_subdir
    elif args.segment_mode == "joint":
        af2rank_subdir = "af2rank_on_proteinebm_top_k_" + ("mask" if args.mask_inter_segment else "nomask")
    else:
        af2rank_subdir = "af2rank_on_proteinebm_top_k"

    # Drive the inference <mode_base> (seq_cond vs seq_cond_segment) for every
    # _inference_base call (here + the child scripts, which inherit os.environ).
    os.environ["PROTEINA_SEGMENT_MODE"] = args.segment_mode

    # Resolve input mode and validate flag combinations
    if args.input is not None:
        input_mode = "input"
        input_path = args.input
        # In --input mode, the working_csv is created downstream; tms_col/cif_dir
        # are still allowed but cross-protein plots cannot run (no tms_col on a
        # working_csv).
    else:
        input_mode = "dataset"
        input_path = args.dataset_file
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    if args.cif_dir and not os.path.exists(args.cif_dir):
        logger.error(f"CIF directory not found: {args.cif_dir}")
        sys.exit(1)

    has_gt = bool(args.cif_dir)
    can_cross_protein_plot = bool(
        has_gt
        and input_mode == "dataset"
        and args.tms_col
        and not args.skip_cross_protein_plots
    )
    if has_gt and input_mode == "input" and not args.skip_cross_protein_plots:
        logger.warning(
            "--cif_dir is set with --input (sequence file); cross-protein plots "
            "require --dataset_file + --tms_col and will not run."
        )

    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)
    shard_cli_args = build_shard_cli_args(shard_index, num_shards, len_col=args.len_col)
    if shard_index is not None:
        logger.info(f"Sharding enabled: shard {shard_index} of {num_shards}")

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if shard_index is not None and num_shards is not None:
        own_sentinel = output_dir_path / f".shard_{shard_index}_of_{num_shards}_complete"
        own_sentinel.unlink(missing_ok=True)
        for _step_s in output_dir_path.glob(f".step_*_shard_{shard_index}_of_{num_shards}_complete"):
            _step_s.unlink()
        logger.info(f"Cleared own shard completion sentinel (shard {shard_index}/{num_shards}).")
    start_time = time.time()
    success = True

    logger.info("=" * 60)
    logger.info(f"PREDICTION PIPELINE ({'WITH GT' if has_gt else 'NO GT'})")
    logger.info("=" * 60)
    logger.info(f"Input mode: {input_mode}  ({input_path})")
    logger.info(f"Scorer: {args.scorer}")
    logger.info(f"CIF dir: {args.cif_dir or '(none — prediction mode)'}")
    if has_gt:
        logger.info(f"TM-score column: {args.tms_col or '(none)'}")
        logger.info(f"Cross-protein plots: {'enabled' if can_cross_protein_plot else 'disabled'}")
    logger.info(f"Inference config: {args.inference_config}")
    logger.info(f"GPUs: {args.num_gpus}")
    logger.info(f"pTM cutoff: {args.ptm_cutoff}")
    logger.info(f"AF2Rank top-k: {args.af2rank_top_k}")
    logger.info(f"AF2Rank backend: {args.af2rank_backend}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Tar protein dirs: {args.tar_protein_dirs}")
    logger.info(f"Dynamic re-sharding: {args.dynamic_resharding}")
    from proteinfoundation.prediction_pipeline.proteina_analysis import resolve_num_workers
    logger.info(f"num_workers (CPU, analysis etc.): {resolve_num_workers(args.num_workers)}")
    skip_analysis = args.skip_analysis

    # ── Step 1: Resolve working CSV (parse --input or use --dataset_file directly) ──
    if input_mode == "input":
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: PARSE INPUT & CREATE PT FILES")
        logger.info("=" * 60)
        df, working_csv = step_parse_input(args.input, args.id_col, args.sequence_col, args.output_dir)
        protein_ids = df["id"].tolist()
        working_csv_id_col = "id"
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: LOADING DATASET CSV")
        logger.info("=" * 60)
        df = pd.read_csv(args.dataset_file)
        if args.id_col not in df.columns:
            logger.error(f"Dataset CSV missing --id_col '{args.id_col}'. Columns: {sorted(df.columns)}")
            sys.exit(1)
        protein_ids = [str(p).strip() for p in df[args.id_col].dropna().unique() if str(p).strip()]
        working_csv = args.dataset_file
        working_csv_id_col = args.id_col
        # Up-front PT pre-build is SHARD-SAFE: only shard 0 (or non-sharded run) builds
        # the PTs; other array shards wait for the PT files to exist (they share the same
        # processed/ dir). Without this guard every shard would redundantly build the FULL
        # dataset with pt_build_workers each -> N_shards x oversubscription + concurrent
        # writes to the same .pt files (corruption risk).
        _build_pts = (shard_index is None) or (shard_index == 0)
        _processed_dir = os.path.join(
            os.environ.get("DATA_PATH", os.path.join(PROTEINA_BASE_DIR, "data")),
            "pdb_train", "processed",
        )
        if args.segment_mode == "joint":
            os.environ["PROTEINA_CONDITIONING_MODE"] = "seq"
            # Eligible ids (>=1 segment >= min_len) computed by every shard from the CSV.
            _eligible = set()
            for _, _row in df.iterrows():
                _seq, _raw = _row.get(args.sequence_col), _row.get(args.segments_col)
                if isinstance(_seq, str) and isinstance(_raw, str):
                    _eligible.add(str(_row[args.id_col]).strip())  # has seq+segments; build decides seg vs fallback
            if _build_pts:
                step_build_segment_pts(args.dataset_file, args.id_col, args.sequence_col,
                                       args.segments_col, args.segment_min_len,
                                       workers=args.pt_build_workers)
            else:
                _wait_for_pts(_processed_dir, _eligible, label="segment")
            protein_ids = [p for p in protein_ids if p in _eligible]
            logger.info(f"segment_mode=joint: {len(protein_ids)} proteins (build_pts={_build_pts}); conditioning=seq")
        elif has_gt and not args.skip_inference:
            # Whole-chain dataset mode: build full-chain PTs UP FRONT in parallel (CPU),
            # so GPU inference workers never block on per-protein CIF->PT (skip_pt forced True below).
            if _build_pts:
                logger.info(f"Whole-chain PT pre-build (CIF->PT) for {len(protein_ids)} proteins, workers={args.pt_build_workers} ...")
                convert_from_csv(args.dataset_file, args.id_col, args.cif_dir,
                                 os.path.join(PROTEINA_BASE_DIR, "data"),
                                 workers=args.pt_build_workers)
            else:
                _wait_for_pts(_processed_dir, set(protein_ids), label="whole-chain")
    logger.info(f"Loaded {len(protein_ids)} proteins")

    inference_base = _inference_base(args.inference_config)
    shard_protein_ids_for_tar = list(protein_ids)
    if shard_index is not None:
        lengths = lengths_from_csv(working_csv, working_csv_id_col, args.len_col)
        if lengths is not None:
            shard_protein_ids_for_tar = shard_proteins(protein_ids, shard_index, num_shards, lengths=lengths)
        else:
            data_dir = os.environ.get("DATA_PATH", os.path.join(PROTEINA_BASE_DIR, "data"))
            shard_protein_ids_for_tar = shard_proteins(protein_ids, shard_index, num_shards, data_dir=data_dir)
    if args.tar_protein_dirs:
        for protein_id in protein_ids:
            ensure_protein_tar(inference_base, str(protein_id))
        logger.info("Initialized per-protein tar files under %s", inference_base)

    # ── Step 2: Proteina inference ──
    if not args.skip_inference and success:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: PROTEINA INFERENCE")
        logger.info("=" * 60)
        # PTs are now ALWAYS built up front (parallel): --input mode via step_parse_input;
        # dataset+segment via step_build_segment_pts; dataset+whole-chain via convert_from_csv
        # above. So GPU workers always skip CIF->PT (never block on the CPU step).
        skip_pt = True
        if not step_proteina_inference(
            csv_file=working_csv,
            csv_col=working_csv_id_col,
            inference_config=args.inference_config,
            num_gpus=args.num_gpus,
            cif_dir=args.cif_dir,
            skip_pt_conversion=skip_pt,
            usalign_path=args.usalign_path,
            proteina_force_compile=args.proteina_force_compile,
            shard_args=shard_cli_args,
            direct_python=args.direct_python,
            rerun=args.rerun_proteina,
            tar_protein_dirs=args.tar_protein_dirs,
            dynamic_resharding=args.dynamic_resharding,
            progress_check_workers=args.progress_check_workers,
        ):
            logger.error("Proteina inference failed")
            success = False
        else:
            logger.info("Proteina inference completed successfully")
    elif args.skip_inference:
        logger.info("Skipping Proteina inference")
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            output_dir_path, "inference", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    # ── Step 3: Scoring (AF2Rank or ProteinEBM, per --scorer) ──
    if not args.skip_scoring and success:
        logger.info("\n" + "=" * 60)
        if args.scorer == "af2rank":
            logger.info("STEP 3: AF2RANK SCORING")
        else:
            logger.info("STEP 3: PROTEINEBM SCORING")
        logger.info("=" * 60)
        if args.scorer == "af2rank":
            if not has_gt:
                logger.error("--scorer=af2rank requires --cif_dir (AF2Rank needs a reference).")
                success = False
            else:
                scoring_ok = step_af2rank_scoring(
                    csv_file=working_csv,
                    csv_col=working_csv_id_col,
                    cif_dir=args.cif_dir,
                    inference_config=args.inference_config,
                    num_gpus=args.num_gpus,
                    recycles=args.recycles,
                    regenerate_plots=args.regenerate_plots,
                    backend=args.af2rank_backend,
                    use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
                    use_cuequivariance_attention=args.use_cuequivariance_attention,
                    use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
                    shard_args=shard_cli_args,
                    direct_python=args.direct_python,
                    rerun=args.rerun_score,
                    tar_protein_dirs=args.tar_protein_dirs,
                    dynamic_resharding=args.dynamic_resharding,
                    progress_check_workers=args.progress_check_workers,
                )
                if not scoring_ok:
                    logger.error("AF2Rank scoring failed")
                    success = False
        else:
            if not step_proteinebm_scoring(
                csv_file=working_csv,
                csv_col=working_csv_id_col,
                inference_config=args.inference_config,
                num_gpus=args.num_gpus,
                proteinebm_config=args.proteinebm_config,
                proteinebm_checkpoint=args.proteinebm_checkpoint,
                cif_dir=args.cif_dir,
                proteinebm_t=args.proteinebm_t,
                proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
                proteinebm_batch_size=args.proteinebm_batch_size,
                proteinebm_template_self_condition=args.proteinebm_template_self_condition,
                num_workers=args.num_workers,
                shard_args=shard_cli_args,
                direct_python=args.direct_python,
                rerun=args.rerun_score,
                tar_protein_dirs=args.tar_protein_dirs,
                dynamic_resharding=args.dynamic_resharding,
                progress_check_workers=args.progress_check_workers,
            ):
                logger.error("ProteinEBM scoring failed")
                success = False
            else:
                logger.info("ProteinEBM scoring completed successfully")
    elif args.skip_scoring:
        logger.info(f"Skipping scoring stage ({args.scorer})")
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            output_dir_path, "scoring", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    # ── Step 4: AF2Rank on ProteinEBM top-k (only when --scorer=proteinebm + top-k>0) ──
    if (success and not args.skip_scoring and not args.skip_af2rank_on_top_k
            and args.scorer == "proteinebm" and int(args.af2rank_top_k) > 0):
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: AF2RANK ON PROTEINEBM TOP-K")
        logger.info("=" * 60)
        if args.af2rank_backend != "openfold":
            logger.warning(
                f"--af2rank_backend {args.af2rank_backend!r} is ignored at step 4: "
                "the top-k refinement uses run_af2rank_prediction.py which is openfold-only."
            )
        if not step_af2rank_topk(
            inference_config=args.inference_config,
            af2rank_top_k=args.af2rank_top_k,
            recycles=args.recycles,
            num_gpus=args.num_gpus,
            csv_file=working_csv,
            csv_col=working_csv_id_col,
            cif_dir=args.cif_dir,
            proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
            use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=args.use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
            shard_args=shard_cli_args,
            direct_python=args.direct_python,
            filter_existing=bool(args.af2rank_topk_filter_existing) and not args.rerun_af2rank_on_top_k,
            tar_protein_dirs=args.tar_protein_dirs,
            dynamic_resharding=args.dynamic_resharding,
            progress_check_workers=args.progress_check_workers,
            # When ProteinEBM was re-scored, the per-decoy CSV gains new
            # ptm/mean_pae columns. Force-regenerate the top-k summary CSV
            # + per-protein af2rank-vs-proteinebm pTM scatter even for
            # proteins whose AF2Rank scoring is already complete.
            force_regenerate_topk_summary=bool(args.rerun_score),
            segment_mode=args.segment_mode,
            segments_col=args.segments_col,
            segment_min_len=args.segment_min_len,
            mask_inter_segment=args.mask_inter_segment,
            af2rank_subdir=af2rank_subdir,
        ):
            logger.error("AF2Rank top-k step failed")
            success = False
        else:
            logger.info("AF2Rank top-k step completed successfully")
    elif args.skip_af2rank_on_top_k:
        logger.info("Skipping AF2Rank top-k step")
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            output_dir_path, "af2rank_topk", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    # ── Step 5: Central analysis (every shard runs its own subset) ──
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: CENTRAL ANALYSIS")
        logger.info("=" * 60)
        if not skip_analysis:
            if not step_central_analysis(
                inference_config=args.inference_config,
                csv_file=working_csv,
                csv_col=working_csv_id_col,
                cif_dir=args.cif_dir,
                proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
                num_workers=args.num_workers,
                shard_args=shard_cli_args,
                direct_python=args.direct_python,
                rerun=args.rerun_proteina or args.rerun_score or args.rerun_af2rank_on_top_k or args.rerun_analysis,
                skip_diversity=args.skip_diversity,
                tar_protein_dirs=args.tar_protein_dirs,
                dynamic_resharding=args.dynamic_resharding,
                progress_check_workers=args.progress_check_workers,
                af2rank_subdir=af2rank_subdir,
            ):
                logger.error("Central analysis failed")
                success = False
        else:
            logger.info("Skipping central analysis")
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            output_dir_path, "analysis", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    if args.tar_protein_dirs and shard_index is not None and num_shards is not None:
        stats = pack_protein_dirs(inference_base, shard_protein_ids_for_tar, delete_after=True)
        logger.info("Shard-owned tar finalization: %s", stats)

    if shard_index is not None and num_shards is not None:
        sentinel = output_dir_path / f".shard_{shard_index}_of_{num_shards}_complete"
        sentinel.write_text("0" if success else "1")
        logger.info(f"Wrote shard {shard_index} completion sentinel ({'success' if success else 'failure'}).")

    if shard_index is not None and shard_index != 0:
        logger.info("Per-protein work complete (after central analysis), exiting.")
        sys.exit(0 if success else 1)

    if success and shard_index is not None:
        other_shard_indices = list(range(1, num_shards))
        if other_shard_indices:
            def _check_shard_done(shard_idx: int) -> bool:
                sentinel_path = output_dir_path / f".shard_{shard_idx}_of_{num_shards}_complete"
                return sentinel_path.exists()

            if not wait_for_completion(
                other_shard_indices,
                _check_shard_done,
                poll_interval=args.shard_poll_interval,
                timeout=args.shard_timeout,
                item_name="shards",
            ):
                logger.error("Timeout waiting for all shards to complete")
                success = False

            failed_shards = []
            for shard_idx in other_shard_indices:
                sentinel_path = output_dir_path / f".shard_{shard_idx}_of_{num_shards}_complete"
                if sentinel_path.exists() and sentinel_path.read_text().strip() != "0":
                    failed_shards.append(shard_idx)
            if failed_shards:
                logger.error("%d shard(s) reported failure: %s", len(failed_shards), failed_shards)
                success = False

    results = []

    # ── Step 6: Collect results (best per protein + summary CSV/JSON) ──
    if success and not args.skip_collect_results and args.scorer == "proteinebm":
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: COLLECT RESULTS")
        logger.info("=" * 60)
        results = step_collect_results(
            protein_ids, args.inference_config, args.output_dir,
            args.ptm_cutoff, args.proteinebm_analysis_subdir,
            tar_protein_dirs=args.tar_protein_dirs,
            af2rank_subdir=af2rank_subdir,
        )
    elif args.skip_collect_results:
        logger.info("Skipping result collection / summary writeout")
    elif args.scorer != "proteinebm":
        logger.info(
            f"Skipping result collection (--scorer={args.scorer} — collect_results "
            "expects AF2Rank-on-ProteinEBM-topk outputs)"
        )

    # ── Step 7: Cross-protein plots (only when --cif_dir + --tms_col + --dataset_file) ──
    if success and can_cross_protein_plot:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 7: CROSS-PROTEIN PLOTS")
        logger.info("=" * 60)
        if not step_cross_protein_plots(
            inference_config=args.inference_config,
            output_dir=args.output_dir,
            scorer=args.scorer,
            dataset_file=args.dataset_file,
            id_col=args.id_col,
            tms_col=args.tms_col,
            af2rank_top_k=int(args.af2rank_top_k),
            proteinebm_plot_mode=args.proteinebm_cross_protein_plot_mode,
            proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
            direct_python=args.direct_python,
            skip_af2rank_on_top_k=args.skip_af2rank_on_top_k,
            af2rank_subdir=af2rank_subdir,
        ):
            logger.error("Cross-protein plotting failed")
            success = False
    elif has_gt and args.skip_cross_protein_plots:
        logger.info("Skipping cross-protein plots (--skip_cross_protein_plots)")

    # ── Step 8: pTM distribution histogram + calibration curve ──
    if success and results and not args.skip_distribution_plot:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 8: PTM DISTRIBUTION PLOT")
        logger.info("=" * 60)
        step_plot_distribution(results, args.output_dir, args.ptm_cutoff)
        # Calibration curve runs only when ground-truth TM-scores are present
        # in the per-protein results (i.e. GT-evaluation mode); the helper
        # silently skips otherwise.
        step_plot_ptm_calibration_curve(results, args.output_dir, args.ptm_cutoff)
    elif args.skip_distribution_plot:
        logger.info("Skipping pTM distribution plot")

    # ── Summary ──
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    if success:
        if results:
            num_passing = sum(1 for r in results if r["passes_cutoff"])
            logger.info(f"Pipeline completed in {total_time:.1f}s")
            logger.info(f"Proteins processed: {len(protein_ids)}")
            logger.info(f"Passing pTM >= {args.ptm_cutoff}: {num_passing}/{len(results)}")
        else:
            logger.info(f"Pipeline completed in {total_time:.1f}s")
            logger.info(f"Proteins processed: {len(protein_ids)}")
        logger.info(f"Results: {args.output_dir}")
    else:
        logger.error(f"Pipeline failed after {total_time:.1f}s")

    logger.info("=" * 60)
    if shard_index == 0 and num_shards is not None:
        _cleanup_shard_sentinels(output_dir_path, num_shards)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
