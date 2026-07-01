#!/usr/bin/env python3
"""
Parallel Proteina Inference Script

This script handles parallelized protein structure generation using Proteina across multiple GPUs.
It processes all proteins from a CSV file and distributes them across available GPUs.
"""

import argparse
import builtins
import logging
import sys
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from proteinfoundation.prediction_pipeline.cif_to_pt_converter import convert_from_csv
from proteinfoundation.prediction_pipeline.sharding_utils import (
    add_shard_args,
    default_progress_check_workers,
    filter_proteins_threaded,
    lengths_from_csv,
    load_lengths_from_pt,
    resolve_shard_args,
    shard_proteins,
)
from proteinfoundation.prediction_pipeline.protein_tar_utils import (
    pack_protein_dirs,
    protein_glob_members,
    restore_selected_protein_dirs,
)

def _terminate_process_group(signum: int) -> None:
    """
    Terminate this job's whole process group (main + worker processes + inference subprocesses).
    This avoids lingering orphaned GPU processes after Ctrl+C / SIGTERM (e.g. from `timeout`).
    """
    # Restore default handlers to avoid re-entering our handler when we signal the group.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.killpg(os.getpgrp(), signum)


def signal_handler(signum, frame):
    """Handle interrupt signals by terminating the whole process group."""
    sig_name = signal.Signals(signum).name
    logger.warning(f"\n⚠️  Received {sig_name}. Terminating all worker/inference processes...")
    _terminate_process_group(signum)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Auto-detect proteina base directory
def get_proteina_base_dir():
    """Auto-detect proteina base directory."""
    # Try common locations
    possible_dirs = [
        os.path.expanduser('~/proteina'),
        os.path.join(os.getcwd(), '..')  # Assume we're in prediction_pipeline
    ]
    
    for dir_path in possible_dirs:
        abs_path = os.path.abspath(dir_path)
        if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, 'proteinfoundation')):
            return abs_path
    
    # Fallback to current directory's parent
    return os.path.abspath(os.path.join(os.getcwd(), '..'))

from proteinfoundation.prediction_pipeline.conditioning_paths import conditioning_label

PROTEINA_BASE_DIR = get_proteina_base_dir()

env_path = os.path.join(PROTEINA_BASE_DIR, '.env')
if load_dotenv and os.path.exists(env_path):
    load_dotenv(env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker-process model cache (in-process single-load).
#
# ProcessPoolExecutor reuses the same worker process across many submitted
# tasks. By caching the loaded model + compiled graph as MODULE-LEVEL state
# we get true single-load: the first protein on a worker pays the checkpoint
# load + first-compile cost, every subsequent protein on the SAME worker
# reuses the cached objects.
# ---------------------------------------------------------------------------
_WORKER_MODEL = None        # (model, nn_ag, trainer, cfg, config_name) or None
_WORKER_MODEL_KEY = None    # (inference_config, conditioning_mode, force_compile, dynamic_shapes)


def _get_or_load_worker_model(inference_config, conditioning_mode, force_compile,
                              dynamic_shapes, verbose=False, use_cueq=True):
    """Cached per-worker model loader. Returns (model, nn_ag, trainer, cfg, config_name).

    The first call on a worker process loads the checkpoint and (when
    force_compile=True) sets up torch.compile with the chosen dynamic-shapes
    flag. Subsequent calls with the same key return the cached objects so
    every protein on that worker reuses the model + compiled graphs.
    """
    global _WORKER_MODEL, _WORKER_MODEL_KEY
    key = (inference_config, conditioning_mode, bool(force_compile), bool(dynamic_shapes))
    if _WORKER_MODEL is not None and _WORKER_MODEL_KEY == key:
        return _WORKER_MODEL

    # First call (or key change): build a synthetic args namespace to drive
    # the shared compose_inference_cfg from inference.py. cath_code is
    # per-protein, not per-worker — for seq_cath mode we feed a placeholder
    # here so the initial cfg merge succeeds; the real cath_code is supplied
    # to run_one_protein_in_process below.
    from proteinfoundation.inference import compose_inference_cfg, load_model_for_worker
    import types
    a = types.SimpleNamespace(
        config_subdir=None,
        config_number=-1,
        config_name=inference_config,
        max_nsamples=None,
        nsamples_per_protein=None,
        conditioning_mode=conditioning_mode,
        cath_code=("x.x.x.x" if conditioning_mode == "seq_cath" else None),
    )
    cfg, config_name, _ = compose_inference_cfg(a)
    model, nn_ag, trainer = load_model_for_worker(
        cfg,
        force_compile=force_compile,
        dynamic_shapes=dynamic_shapes,
        verbose=verbose,
        use_cueq=use_cueq,
    )
    _WORKER_MODEL = (model, nn_ag, trainer, cfg, config_name)
    _WORKER_MODEL_KEY = key
    logger.info(f"[worker pid={os.getpid()}] Cached model "
                f"(inference_config={inference_config}, conditioning_mode={conditioning_mode}, "
                f"force_compile={force_compile}, dynamic_shapes={dynamic_shapes})")
    return _WORKER_MODEL


def _conditioning_label(conditioning_mode):
    """Map a conditioning_mode CLI value to the output-dir segment (delegates to the
    shared conditioning_paths source of truth; REQUIRED, raises on unset)."""
    return conditioning_label(conditioning_mode)


def generate_protein_output_dir(inference_config, protein_name, conditioning_mode=None):
    """Generate consistent output directory path for a protein.

    Layout: inference/{inference_config}/{conditioning_label}/{protein_name}/
    """
    label = _conditioning_label(conditioning_mode)
    return os.path.join(PROTEINA_BASE_DIR, 'inference', inference_config, label, protein_name)

def create_single_protein_csv(csv_file, csv_col, protein_name, output_dir):
    """Create a single-protein CSV file for individual processing."""
    df = pd.read_csv(csv_file)
    
    # Filter for this specific protein
    protein_df = df[df[csv_col] == protein_name]
    
    if protein_df.empty:
        raise ValueError(f"Protein {protein_name} not found in CSV file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write single-protein CSV
    single_csv_path = os.path.join(output_dir, 'single_protein.csv')
    protein_df.to_csv(single_csv_path, index=False)
    
    return single_csv_path

def run_cif_to_pt_conversion(csv_file, csv_col, cif_dir):
    """
    Run CIF to PT conversion step.
    Directly imports and calls the converter to avoid subprocess overhead.
    """
    try:
        convert_from_csv(
            csv_file=csv_file,
            csv_col=csv_col,
            cif_dir=cif_dir,
            output_dir=os.path.join(PROTEINA_BASE_DIR, 'data')  # Uses DATA_PATH internally
        )
        
        # Create a mock result object for compatibility
        class MockResult:
            returncode = 0
            stdout = "CIF to PT conversion completed"
            stderr = ""
        
        return MockResult()
        
    except Exception as e:
        logger.error(f"CIF to PT conversion failed: {e}")
        class MockResult:
            returncode = 1
            stdout = ""
            stderr = str(e)
        return MockResult()

def run_proteina_inference(protein_name, inference_config, force_compile: bool = False,
                           max_nsamples: int = None,
                           conditioning_mode: str = None,
                           cath_code: str = None,
                           nsamples_per_protein: int = None):
    """
    Run Proteina inference directly.
    Only uses subprocess for calling proteinfoundation/inference.py.

    Note: No timeout - inference can take hours for long proteins.
    Environment activation is handled by the caller (shell script wrapper).
    """
    cmd = [
        sys.executable, 'proteinfoundation/inference.py',
        '--pt', protein_name,
        '--config_name', inference_config,
    ]
    # inference.py's own --force_compile defaults to True (BooleanOptionalAction), so omitting
    # the flag here does NOT disable it there -- must pass --no-force_compile explicitly.
    cmd.append('--force_compile' if force_compile else '--no-force_compile')
    if max_nsamples is not None:
        cmd.extend(['--max_nsamples', str(max_nsamples)])
    if conditioning_mode is not None:
        cmd.extend(['--conditioning_mode', conditioning_mode])
    if cath_code is not None:
        cmd.extend(['--cath_code', cath_code])
    if nsamples_per_protein is not None:
        cmd.extend(['--nsamples_per_protein', str(nsamples_per_protein)])
    logger.info(f"subprocess argv: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=PROTEINA_BASE_DIR,
        capture_output=True,
        text=True
    )
    return result

def _step2_inference_in_process(protein_name, gpu_id, inference_config, conditioning_mode,
                                 cath_code, nsamples_per_protein, force_compile, dynamic_shapes,
                                 protein_output_dir):
    """Step 2 (in-process path): use the cached worker-process model to generate samples
    for one protein, reusing the loaded checkpoint and any compiled graphs.

    Persists OOM-reduced max_nsamples to `builtins._worker_max_nsamples` so it
    sticks across proteins on the same worker — matches the subprocess-path
    behavior.
    """
    from proteinfoundation.inference import run_one_protein_in_process

    model, nn_ag, trainer, cfg, config_name = _get_or_load_worker_model(
        inference_config, conditioning_mode, force_compile, dynamic_shapes
    )

    # Apply per-protein cath_codes_override
    if conditioning_mode == "seq":
        cath_codes_override = ["x.x.x.x"]
    elif conditioning_mode == "seq_cath":
        if cath_code is None:
            return {'protein': protein_name, 'gpu': gpu_id, 'status': 'failed',
                    'error_type': 'BAD_INPUT',
                    'error': "--conditioning_mode seq_cath but no cath_code for this protein",
                    'output_dir': protein_output_dir}
        from proteinfoundation.inference import _normalize_cath_code
        cath_codes_override = [_normalize_cath_code(cath_code)]
    else:
        cath_codes_override = None

    # Apply per-protein nsamples_per_protein override (and OOM-persisted max_nsamples)
    if nsamples_per_protein is not None:
        cfg.nsamples_per_len = int(nsamples_per_protein)
    worker_max = getattr(builtins, '_worker_max_nsamples', None)
    if worker_max is not None:
        cfg.max_nsamples = int(worker_max)

    logger.info(f"[GPU {gpu_id}] Step 2 (in-process): {protein_name} "
                f"cath_code={cath_codes_override} max_nsamples={cfg.max_nsamples} "
                f"nsamples_per_len={cfg.nsamples_per_len}")

    result = run_one_protein_in_process(
        model, nn_ag, trainer, cfg,
        pt_name=protein_name,
        config_name=config_name,
        conditioning_mode=conditioning_mode,
        cath_codes_override=cath_codes_override,
    )

    # Persist any OOM-reduced max_nsamples for subsequent proteins on this worker
    if result.get("final_max_nsamples") is not None:
        if worker_max is None or result["final_max_nsamples"] < worker_max:
            builtins._worker_max_nsamples = int(result["final_max_nsamples"])

    if result.get("status") == "FAILED":
        return {'protein': protein_name, 'gpu': gpu_id, 'status': 'failed',
                'error_type': result.get("error_type", "INFERENCE_ERROR"),
                'error': result.get("error", "unknown"),
                'output_dir': protein_output_dir}

    logger.info(f"[GPU {gpu_id}] ✅ In-process inference completed for {protein_name} "
                f"(samples_written={result.get('n_samples_written', 0)})")
    return {'protein': protein_name, 'gpu': gpu_id, 'status': 'success',
            'output_dir': protein_output_dir}


def process_single_protein(args):
    """
    Process a single protein through the entire Proteina pipeline.
    Includes proper error handling and GPU memory cleanup.

    Dispatches to the in-process or subprocess-per-protein path based on the
    last tuple element (in_process_mode + dynamic_shapes). When tuple length
    is 12 (legacy), uses subprocess; when length is 14, uses in-process if
    requested.
    """
    if len(args) >= 14:
        (protein_name, csv_file, csv_col, cif_dir, inference_config, usalign_path,
         gpu_id, force_compile, skip_pt_conversion,
         conditioning_mode, cath_code, nsamples_per_protein,
         in_process_mode, dynamic_shapes) = args
    else:
        (protein_name, csv_file, csv_col, cif_dir, inference_config, usalign_path,
         gpu_id, force_compile, skip_pt_conversion,
         conditioning_mode, cath_code, nsamples_per_protein) = args
        in_process_mode = False
        dynamic_shapes = True

    try:
        # Set GPU for this process. gpu_id is a local worker index; if the caller already
        # restricted CUDA_VISIBLE_DEVICES to specific physical ids (shared, non-SLURM host),
        # index into that list instead of clobbering it with a raw 0-based id.
        _inherited_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
        _phys_gpus = _inherited_cvd.split(',') if _inherited_cvd else None
        os.environ['CUDA_VISIBLE_DEVICES'] = _phys_gpus[gpu_id] if _phys_gpus and gpu_id < len(_phys_gpus) else str(gpu_id)

        logger.info(f"[GPU {gpu_id}] Starting processing for {protein_name}")

        # Generate output directory for this protein (includes conditioning_mode segment)
        protein_output_dir = generate_protein_output_dir(inference_config, protein_name, conditioning_mode)

        logger.info(f"[GPU {gpu_id}] {protein_name} -> {protein_output_dir}")

        # Create single-protein CSV file
        logger.info(f"[GPU {gpu_id}] Creating single_protein.csv for {protein_name}")
        single_csv_path = create_single_protein_csv(csv_file, csv_col, protein_name, protein_output_dir)

        # Step 1: CIF to PT conversion (skip if PT files already exist)
        if skip_pt_conversion:
            logger.info(f"[GPU {gpu_id}] Skipping CIF to PT conversion for {protein_name} (--skip_pt_conversion)")
        else:
            logger.info(f"[GPU {gpu_id}] Step 1: CIF to PT conversion for {protein_name}")
            result = run_cif_to_pt_conversion(single_csv_path, csv_col, cif_dir)

            if result.returncode != 0:
                logger.error(f"[GPU {gpu_id}] CIF to PT conversion failed for {protein_name}")
                logger.error(f"[GPU {gpu_id}] STDERR: {result.stderr}")
                return {
                    'protein': protein_name,
                    'gpu': gpu_id,
                    'status': 'failed',
                    'error': f"CIF to PT conversion failed: {result.stderr}",
                    'output_dir': protein_output_dir
                }

            logger.info(f"[GPU {gpu_id}] ✅ CIF to PT conversion completed for {protein_name}")

        # Step 2: Proteina inference — branch on in_process_mode.
        if in_process_mode:
            return _step2_inference_in_process(
                protein_name, gpu_id, inference_config, conditioning_mode,
                cath_code, nsamples_per_protein, force_compile, dynamic_shapes,
                protein_output_dir,
            )
        # Legacy: subprocess-per-protein
        # Step 2: Proteina inference with adaptive batch size on OOM
        # Read persisted max_nsamples from prior OOM in this worker (if any)
        current_max_nsamples = getattr(builtins, '_worker_max_nsamples', None)

        while True:
            bs_info = f" (max_nsamples={current_max_nsamples})" if current_max_nsamples else ""
            logger.info(f"[GPU {gpu_id}] Step 2: Running Proteina inference for {protein_name}{bs_info}")
            result = run_proteina_inference(protein_name, inference_config,
                                            force_compile=force_compile,
                                            max_nsamples=current_max_nsamples,
                                            conditioning_mode=conditioning_mode,
                                            cath_code=cath_code,
                                            nsamples_per_protein=nsamples_per_protein)

            if result.returncode == 0:
                break  # success

            error_msg = result.stderr

            if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
                if current_max_nsamples is None:
                    current_max_nsamples = 16  # start from config default
                if current_max_nsamples <= 1:
                    logger.error(f"[GPU {gpu_id}] OOM with max_nsamples=1 for {protein_name}, giving up")
                    return {
                        'protein': protein_name, 'gpu': gpu_id, 'status': 'failed',
                        'error_type': 'GPU_OOM', 'error': error_msg,
                        'output_dir': protein_output_dir
                    }
                current_max_nsamples = current_max_nsamples // 2
                logger.warning(f"[GPU {gpu_id}] OOM for {protein_name}, "
                               f"retrying with max_nsamples={current_max_nsamples}")
                # Persist reduced batch size for all future proteins on this worker
                builtins._worker_max_nsamples = current_max_nsamples
                continue
            else:
                # Non-OOM failure
                logger.error(f"[GPU {gpu_id}] Proteina inference failed for {protein_name}")
                logger.error(f"[GPU {gpu_id}] STDOUT: {result.stdout}")
                logger.error(f"[GPU {gpu_id}] STDERR: {error_msg}")
                return {
                    'protein': protein_name, 'gpu': gpu_id, 'status': 'failed',
                    'error_type': 'INFERENCE_ERROR', 'error': error_msg,
                    'output_dir': protein_output_dir
                }

        logger.info(f"[GPU {gpu_id}] Proteina inference completed for {protein_name}")
        
        # Note: USalign evaluation is not implemented - would be handled separately if needed
        
        logger.info(f"[GPU {gpu_id}] ✅ Successfully completed {protein_name}")
        return {
            'protein': protein_name, 
            'gpu': gpu_id, 
            'status': 'success', 
            'output_dir': protein_output_dir
        }
        
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] ❌ Unexpected error for {protein_name}: {e}")
        logger.error(f"[GPU {gpu_id}] Traceback: {traceback.format_exc()}")
        
        return {
            'protein': protein_name,
            'gpu': gpu_id,
            'status': 'failed',
            'error_type': 'EXCEPTION',
            'error': str(e),
            'output_dir': protein_output_dir if 'protein_output_dir' in locals() else 'unknown'
        }
    
    finally:
        # Cleanup: Clear GPU memory cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass  # Silently fail if torch not available or other issues

# Worker initialization function (must be at module level for pickling)
def worker_init_proteina(counter, lock, num_gpus):
    """Initialize worker with a specific GPU assignment."""
    with lock:
        worker_id = counter.value
        counter.value += 1
    
    gpu_id = worker_id % num_gpus
    _inherited_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    _phys_gpus = _inherited_cvd.split(',') if _inherited_cvd else None
    os.environ['CUDA_VISIBLE_DEVICES'] = _phys_gpus[gpu_id] if _phys_gpus and gpu_id < len(_phys_gpus) else str(gpu_id)
    # Store GPU ID in a process-local variable
    builtins._worker_gpu_id = gpu_id
    logger.info(f"Proteina Worker {worker_id} initialized with GPU {gpu_id}")

# Wrapper function (must be at module level for pickling)
def process_single_protein_wrapper(args_tuple):
    """Wrapper that uses the GPU assigned during worker init.

    Supports both legacy 11-tuple (subprocess mode) and 13-tuple (in-process mode).
    """
    if len(args_tuple) >= 13:
        (protein_name, csv_file, csv_col, cif_dir, inference_config, usalign_path,
         force_compile, skip_pt_conversion,
         conditioning_mode, cath_code, nsamples_per_protein,
         in_process_mode, dynamic_shapes) = args_tuple
    else:
        (protein_name, csv_file, csv_col, cif_dir, inference_config, usalign_path,
         force_compile, skip_pt_conversion,
         conditioning_mode, cath_code, nsamples_per_protein) = args_tuple
        in_process_mode = False
        dynamic_shapes = True

    # Get GPU ID from process-local variable set by worker_init
    gpu_id = getattr(builtins, '_worker_gpu_id', 0)

    # Call the actual processing function (14-tuple for in-process dispatch)
    full_args = (protein_name, csv_file, csv_col, cif_dir, inference_config, usalign_path,
                 gpu_id, force_compile, skip_pt_conversion,
                 conditioning_mode, cath_code, nsamples_per_protein,
                 in_process_mode, dynamic_shapes)
    return process_single_protein(full_args)

def has_multi_char_chain_id(protein_name):
    """
    Check if protein name has multi-character chain ID.
    PDB format only supports single-character chain IDs.
    
    Args:
        protein_name: Protein name in format "PDBID_CHAIN" (e.g., "6SG9_Ci")
    
    Returns:
        True if chain ID has more than 1 character
    """
    parts = protein_name.split('_')
    if len(parts) >= 2:
        chain_id = parts[1]
        return len(chain_id) > 1
    return False

def get_protein_names(csv_file, csv_col):
    """Extract protein names from CSV file."""
    df = pd.read_csv(csv_file)
    proteins = df[csv_col].dropna().unique().tolist()
    return [p for p in proteins if p.strip()]


def _get_expected_nsamples(inference_config):
    """Read the expected nsamples_per_len from the inference config YAML.

    Tries the locations in priority order:
      1. Unified config: cfg.inference.nsamples_per_len  (nested under inference:)
      2. Legacy / inference-only config: cfg.nsamples_per_len  (top-level)
      3. Hydra defaults chain: any base in cfg.defaults that has nsamples_per_len top-level
    Returns None when nothing matches.
    """
    try:
        from omegaconf import OmegaConf
        config_dir = os.path.join(PROTEINA_BASE_DIR, 'configs', 'experiment_config')
        yaml_path = os.path.join(config_dir, f"{inference_config}.yaml")
        if os.path.exists(yaml_path):
            cfg = OmegaConf.load(yaml_path)
            # 1. Unified config: nested under `inference:`
            inf = cfg.get("inference")
            if inf is not None and "nsamples_per_len" in inf:
                return int(inf.nsamples_per_len)
            # 2. Legacy / inference-only config: top-level
            if "nsamples_per_len" in cfg:
                return int(cfg.nsamples_per_len)
            # 3. Defaults chain
            defaults = cfg.get("defaults", [])
            for d in defaults:
                if isinstance(d, str):
                    base_path = os.path.join(config_dir, f"{d}.yaml")
                    if os.path.exists(base_path):
                        base_cfg = OmegaConf.load(base_path)
                        if "nsamples_per_len" in base_cfg:
                            return int(base_cfg.nsamples_per_len)
    except Exception as e:
        logger.warning(f"Could not read nsamples_per_len from config: {e}")
    return None  # Unknown — caller decides policy


def find_proteins_needing_inference(
    csv_file,
    csv_col,
    inference_config,
    candidate_proteins=None,
    tar_protein_dirs: bool = False,
    max_workers=None,
    conditioning_mode=None,
):
    """Find proteins from CSV that need inference (incomplete or missing).

    A protein is skipped only when it already has at least ``nsamples_per_len``
    PDB files (i.e. fully completed).  Partially completed proteins are
    included so that ``inference.py`` can resume from where they left off.
    """
    # Get proteins from CSV file
    csv_proteins = list(candidate_proteins) if candidate_proteins is not None else get_protein_names(csv_file, csv_col)

    inference_base_dir = os.path.join(
        PROTEINA_BASE_DIR, 'inference', inference_config, _conditioning_label(conditioning_mode)
    )

    expected_nsamples = _get_expected_nsamples(inference_config)
    if expected_nsamples is not None:
        logger.info(f"Expected nsamples_per_len from config: {expected_nsamples}")

    def _inference_complete(protein_name: str) -> bool:
        protein_dir = Path(inference_base_dir) / protein_name

        if tar_protein_dirs:
            n_existing = len(protein_glob_members(inference_base_dir, protein_name, f"{protein_name}_*.pdb"))
            exists_for_check = n_existing > 0 or protein_dir.exists() or (Path(inference_base_dir) / f"{protein_name}.tar").exists()
        else:
            exists_for_check = protein_dir.exists() and protein_dir.is_dir()
            pdb_files = list(protein_dir.glob(f"{protein_name}_*.pdb")) if exists_for_check else []
            n_existing = len(pdb_files)

        if exists_for_check:
            if n_existing == 0:
                return False
            elif expected_nsamples is not None and n_existing < expected_nsamples:
                logger.info(f"Partial inference for {protein_name} ({n_existing}/{expected_nsamples} PDB files), will resume")
                return False
            else:
                logger.info(f"Inference already completed for {protein_name} ({n_existing} PDB files), skipping")
                return True
        return False

    needing_inference, _ = filter_proteins_threaded(csv_proteins, _inference_complete, max_workers=max_workers)
    return needing_inference

def main():
    parser = argparse.ArgumentParser(description='Parallel Proteina Inference Pipeline')
    parser.add_argument('--csv_file', required=True, help='Path to CSV file with protein data')
    parser.add_argument('--csv_col', required=True, help='Column name in CSV file to use for protein selection')
    parser.add_argument('--cif_dir', default=None, help='Directory containing CIF files (not required when --skip_pt_conversion is set)')
    parser.add_argument('--inference_config', required=True, help='Inference configuration name')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--usalign_path', help='Path to USalign executable')
    parser.add_argument('--skip_existing', action='store_true', 
                       help='Skip proteins that already have inference completed')
    parser.add_argument(
        '--force_compile',
        action='store_true',
        help='Force torch.compile during inference (default: off).',
    )
    parser.add_argument(
        '--skip_pt_conversion',
        action='store_true',
        help='Skip CIF-to-PT conversion (use when PT files already exist, e.g. created from sequences).',
    )
    parser.add_argument(
        "--tar_protein_dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-protein inference directories as uncompressed <protein_id>.tar archives (default: True).",
    )
    parser.add_argument("--progress_check_workers", type=int, default=default_progress_check_workers(),
                        help="Thread workers for progress checks (default: min(32, cpu_count * 4)).")
    parser.add_argument("--dynamic_resharding", action=argparse.BooleanOptionalAction, default=True,
                        help="Filter global progress before sharding each step to reduce idle shards (default: True).")
    parser.add_argument("--conditioning_mode", choices=["seq", "seq_cath", "legacy"], default=None,
                        help="Which conditioning to apply this run (REQUIRED; no silent fallback — omitting it raises). "
                             "'seq' = sequence only (forces cath='x.x.x.x', fold_cond=False). "
                             "'seq_cath' = sequence + top-1 CATH (requires a 'cath_code' column in --csv_file). "
                             "'legacy' = read/write the old un-namespaced 'legacy/' subdir (explicit only).")
    parser.add_argument("--nsamples_per_protein", type=int, default=None,
                        help="Override nsamples_per_len (total samples per protein) per inference subprocess.")
    parser.add_argument("--in_process", action=argparse.BooleanOptionalAction, default=True,
                        help="Default: each worker process loads the model ONCE and reuses it across "
                             "every assigned protein (single-load). --no-in_process falls back to "
                             "the legacy subprocess-per-protein path (one Python startup + ckpt load "
                             "per chain).")
    parser.add_argument("--dynamic_shapes", action=argparse.BooleanOptionalAction, default=True,
                        help="When --force_compile is set: torch.compile with dynamic=True so a single "
                             "compiled graph handles all protein lengths. --no-dynamic_shapes restores "
                             "static-shape compilation (one specialized graph per (length, batch).")
    add_shard_args(parser)

    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    if not args.skip_pt_conversion:
        if not args.cif_dir:
            logger.error("--cif_dir is required when --skip_pt_conversion is not set")
            sys.exit(1)
        if not os.path.exists(args.cif_dir):
            logger.error(f"CIF directory not found: {args.cif_dir}")
            sys.exit(1)
    
    # Static shard ownership is kept only for pipeline-level final tar cleanup.
    # Dynamic re-sharding filters the global unfinished set, then partitions it
    # disjointly per step so slow shards can pick up remaining work at boundaries.
    global_protein_names = get_protein_names(args.csv_file, args.csv_col)
    protein_names = list(global_protein_names)
    logger.info(f"Found {len(global_protein_names)} proteins in CSV file")

    # Note: Multi-character chain IDs are now supported in AF2Rank via chain extraction
    # Proteina PT conversion also handles them correctly

    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)
    lengths = lengths_from_csv(args.csv_file, args.csv_col, args.len_col)
    data_dir = os.environ.get("DATA_PATH", os.path.join(PROTEINA_BASE_DIR, "data"))
    if shard_index is not None:
        shard_protein_names_for_tar = shard_proteins(
            global_protein_names, shard_index, num_shards, lengths=lengths, data_dir=None if lengths is not None else data_dir
        )
        if not args.dynamic_resharding:
            protein_names = list(shard_protein_names_for_tar)
    else:
        # Non-sharded: sort short-to-long so OOM batch-size reductions stay conservative
        if lengths is None:
            pt_lengths = load_lengths_from_pt(protein_names, data_dir)
            if pt_lengths:
                lengths = pt_lengths
        if lengths:
            protein_names = sorted(protein_names, key=lambda p: lengths.get(p, 0))
            logger.info("Sorted proteins short-to-long for OOM-conservative batch-size adaptation.")
        shard_protein_names_for_tar = list(protein_names)

    inference_base_dir = os.path.join(
        PROTEINA_BASE_DIR, "inference", args.inference_config, _conditioning_label(args.conditioning_mode)
    )

    # Resolve per-protein cath_codes when conditioning_mode is seq_cath.
    cath_by_name = {}
    if args.conditioning_mode == "seq_cath":
        cath_df = pd.read_csv(args.csv_file)
        if "cath_code" not in cath_df.columns:
            logger.error(f"--conditioning_mode seq_cath requires a 'cath_code' column in {args.csv_file}")
            sys.exit(1)
        cath_by_name = dict(zip(cath_df[args.csv_col].astype(str), cath_df["cath_code"].astype(str)))
        n_null = sum(1 for v in cath_by_name.values() if v.strip().lower() in ("", "nan", "x.x.x.x"))
        logger.info(f"Loaded {len(cath_by_name)} cath_codes from {args.csv_file} (null/x.x.x.x count: {n_null})")

    check_candidates = global_protein_names if args.dynamic_resharding and shard_index is not None else protein_names
    if args.skip_existing:
        check_start = time.perf_counter()
        needing_inference = find_proteins_needing_inference(
            args.csv_file,
            args.csv_col,
            args.inference_config,
            candidate_proteins=check_candidates,
            tar_protein_dirs=args.tar_protein_dirs,
            max_workers=args.progress_check_workers,
            conditioning_mode=args.conditioning_mode,
        )
        if args.dynamic_resharding and shard_index is not None:
            protein_names = shard_proteins(
                needing_inference, shard_index, num_shards, lengths=lengths, data_dir=None if lengths is not None else data_dir
            )
        else:
            needing_set = set(needing_inference)
            protein_names = [p for p in protein_names if p in needing_set]
        logger.info(
            "tar_check inference: checked=%d complete=%d needing_work=%d elapsed_seconds=%.3f",
            len(check_candidates), len(check_candidates) - len(needing_inference), len(protein_names),
            time.perf_counter() - check_start,
        )
        logger.info(f"Found {len(protein_names)} proteins in this shard needing inference")
    else:
        if args.dynamic_resharding and shard_index is not None:
            protein_names = shard_proteins(
                global_protein_names, shard_index, num_shards, lengths=lengths, data_dir=None if lengths is not None else data_dir
            )
        logger.info(f"Found {len(protein_names)} proteins in this shard to process")

    protein_names_for_tar = list(protein_names)
    if args.tar_protein_dirs:
        stats = restore_selected_protein_dirs(inference_base_dir, protein_names_for_tar)
        logger.info("tar_restore inference: %s", stats)

    logger.info(f"Proteins: {protein_names}")
    
    # Check available GPUs
    try:
        gpu_count = int(subprocess.check_output(['nvidia-smi', '--list-gpus']).decode().count('\n'))
    except:
        gpu_count = 1
    
    if args.num_gpus > gpu_count:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {gpu_count} available, using {gpu_count}")
        args.num_gpus = gpu_count
    
    logger.info(f"Using {args.num_gpus} GPU(s) for parallel processing")
    
    # Process proteins in parallel with proper GPU assignment
    start_time = time.time()
    results = []
    
    # In-process mode loads CUDA-touching modules (deepspeed, torch) in the
    # parent before the worker starts. Default fork-based ProcessPoolExecutor
    # would inherit a partially-initialized CUDA context and the worker's
    # `torch.cuda.is_available()` call fails with
    # "CUDA error: initialization error". Use spawn for a clean child interpreter.
    # Use the SAME context for the Manager so its proxies work with the executor.
    mp_ctx = mp.get_context('spawn') if args.in_process else mp.get_context()
    manager = mp_ctx.Manager()
    worker_counter = manager.Value('i', 0)
    worker_lock = manager.Lock()

    executor = ProcessPoolExecutor(
        max_workers=args.num_gpus,
        initializer=worker_init_proteina,
        initargs=(worker_counter, worker_lock, args.num_gpus),
        mp_context=mp_ctx,
    )
    
    try:
        # Submit all jobs (GPU assignment happens via worker init)
        # 13-tuple: legacy 11 fields + (in_process_mode, dynamic_shapes)
        work_items = [(protein_name, args.csv_file, args.csv_col, args.cif_dir, args.inference_config, args.usalign_path,
                       args.force_compile, args.skip_pt_conversion,
                       args.conditioning_mode, cath_by_name.get(protein_name), args.nsamples_per_protein,
                       args.in_process, args.dynamic_shapes)
                      for protein_name in protein_names]
        future_to_protein = {executor.submit(process_single_protein_wrapper, item): work_items[i][0] 
                            for i, item in enumerate(work_items)}
        
        # Collect results as they complete
        for future in as_completed(future_to_protein):
            protein_name = future_to_protein[future]
            try:
                result = future.result()
                results.append(result)
                
                completed = len(results)
                total = len(protein_names)
                elapsed = time.time() - start_time
                
                if result['status'] == 'success':
                    logger.info(f"✅ Progress: {completed}/{total} proteins completed ({elapsed:.1f}s elapsed)")
                else:
                    logger.error(f"❌ {protein_name} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {protein_name} failed with exception: {e}")
                results.append({'protein': protein_name, 'status': 'failed', 'error': str(e)})
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        logger.info("✓ Executor shut down cleanly")
    
    # Summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'failed'])
    
    # Count error types
    error_types = {}
    for result in results:
        if result['status'] == 'failed':
            error_type = result.get('error_type', 'UNKNOWN')
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    logger.info(f"🎯 Pipeline completed in {total_time:.1f}s")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    
    if error_types:
        logger.info(f"Error breakdown:")
        for error_type, count in error_types.items():
            logger.info(f"  - {error_type}: {count}")
    
    if failed > 0:
        logger.info("Failed proteins:")
        for result in results:
            if result['status'] == 'failed':
                error_type = result.get('error_type', 'UNKNOWN')
                error_preview = str(result.get('error', 'Unknown error'))[:100]
                logger.info(f"  - {result['protein']} [{error_type}]: {error_preview}")

    if args.tar_protein_dirs:
        stats = pack_protein_dirs(inference_base_dir, protein_names_for_tar, delete_after=True)
        logger.info("tar_pack inference finalization: %s", stats)
    
    # Return appropriate exit code
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
