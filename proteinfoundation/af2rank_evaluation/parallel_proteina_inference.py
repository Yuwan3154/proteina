#!/usr/bin/env python3
"""
Parallel Proteina Inference Script

This script handles parallelized protein structure generation using Proteina across multiple GPUs.
It processes all proteins from a CSV file and distributes them across available GPUs.
"""

import argparse
import builtins
import logging
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

from proteinfoundation.af2rank_evaluation.cif_to_pt_converter import convert_from_csv

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
        os.path.join(os.getcwd(), '..')  # Assume we're in af2rank_evaluation
    ]
    
    for dir_path in possible_dirs:
        abs_path = os.path.abspath(dir_path)
        if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, 'proteinfoundation')):
            return abs_path
    
    # Fallback to current directory's parent
    return os.path.abspath(os.path.join(os.getcwd(), '..'))

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


def generate_protein_output_dir(inference_config, protein_name):
    """Generate consistent output directory path for a protein."""
    return os.path.join(PROTEINA_BASE_DIR, 'inference', inference_config, protein_name)

def create_single_protein_csv(csv_file, csv_column, protein_name, output_dir):
    """Create a single-protein CSV file for individual processing."""
    df = pd.read_csv(csv_file)
    
    # Filter for this specific protein
    protein_df = df[df[csv_column] == protein_name]
    
    if protein_df.empty:
        raise ValueError(f"Protein {protein_name} not found in CSV file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write single-protein CSV
    single_csv_path = os.path.join(output_dir, 'single_protein.csv')
    protein_df.to_csv(single_csv_path, index=False)
    
    return single_csv_path

def run_cif_to_pt_conversion(csv_file, csv_column, cif_dir):
    """
    Run CIF to PT conversion step.
    Directly imports and calls the converter to avoid subprocess overhead.
    """
    try:
        convert_from_csv(
            csv_file=csv_file,
            csv_column=csv_column,
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
                           max_nsamples: int = None):
    """
    Run Proteina inference directly.
    Only uses subprocess for calling proteinfoundation/inference.py.

    Note: No timeout - inference can take hours for long proteins.
    Environment activation is handled by the caller (shell script wrapper).
    """
    cmd = [
        'python', 'proteinfoundation/inference.py',
        '--pt', protein_name,
        '--config_name', inference_config,
    ]
    if force_compile:
        cmd.append('--force_compile')
    if max_nsamples is not None:
        cmd.extend(['--max_nsamples', str(max_nsamples)])

    result = subprocess.run(
        cmd,
        cwd=PROTEINA_BASE_DIR,
        capture_output=True,
        text=True
    )
    return result

def process_single_protein(args):
    """
    Process a single protein through the entire Proteina pipeline.
    Includes proper error handling and GPU memory cleanup.
    """
    protein_name, csv_file, csv_column, cif_dir, inference_config, usalign_path, gpu_id, force_compile, skip_pt_conversion = args
    
    try:
        # Set GPU for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        logger.info(f"[GPU {gpu_id}] Starting processing for {protein_name}")
        
        # Generate output directory for this protein
        protein_output_dir = generate_protein_output_dir(inference_config, protein_name)
        
        logger.info(f"[GPU {gpu_id}] {protein_name} -> {protein_output_dir}")
        
        # Create single-protein CSV file
        logger.info(f"[GPU {gpu_id}] Creating single_protein.csv for {protein_name}")
        single_csv_path = create_single_protein_csv(csv_file, csv_column, protein_name, protein_output_dir)
        
        # Step 1: CIF to PT conversion (skip if PT files already exist)
        if skip_pt_conversion:
            logger.info(f"[GPU {gpu_id}] Skipping CIF to PT conversion for {protein_name} (--skip_pt_conversion)")
        else:
            logger.info(f"[GPU {gpu_id}] Step 1: CIF to PT conversion for {protein_name}")
            result = run_cif_to_pt_conversion(single_csv_path, csv_column, cif_dir)

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
        
        # Step 2: Proteina inference with adaptive batch size on OOM
        # Read persisted max_nsamples from prior OOM in this worker (if any)
        current_max_nsamples = getattr(builtins, '_worker_max_nsamples', None)

        while True:
            bs_info = f" (max_nsamples={current_max_nsamples})" if current_max_nsamples else ""
            logger.info(f"[GPU {gpu_id}] Step 2: Running Proteina inference for {protein_name}{bs_info}")
            result = run_proteina_inference(protein_name, inference_config,
                                            force_compile=force_compile,
                                            max_nsamples=current_max_nsamples)

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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Store GPU ID in a process-local variable
    builtins._worker_gpu_id = gpu_id
    logger.info(f"Proteina Worker {worker_id} initialized with GPU {gpu_id}")

# Wrapper function (must be at module level for pickling)
def process_single_protein_wrapper(args_tuple):
    """Wrapper that uses the GPU assigned during worker init."""
    protein_name, csv_file, csv_column, cif_dir, inference_config, usalign_path, force_compile, skip_pt_conversion = args_tuple

    # Get GPU ID from process-local variable set by worker_init
    gpu_id = getattr(builtins, '_worker_gpu_id', 0)

    # Call the actual processing function
    full_args = (protein_name, csv_file, csv_column, cif_dir, inference_config, usalign_path, gpu_id, force_compile, skip_pt_conversion)
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

def get_protein_names(csv_file, csv_column):
    """Extract protein names from CSV file."""
    df = pd.read_csv(csv_file)
    proteins = df[csv_column].dropna().unique().tolist()
    return [p for p in proteins if p.strip()]

def get_protein_lengths(protein_names):
    """Load PT files to get protein lengths for workload balancing."""
    data_dir = os.path.join(
        os.environ.get('DATA_PATH', os.path.join(PROTEINA_BASE_DIR, 'data')),
        'pdb_train', 'processed'
    )
    lengths = {}
    for name in protein_names:
        pt_path = os.path.join(data_dir, f"{name}.pt")
        if os.path.exists(pt_path):
            pt = torch.load(pt_path, weights_only=False, map_location='cpu')
            lengths[name] = len(pt.residue_type)
        else:
            lengths[name] = 0  # unknown -> treated as shortest
    return lengths


def shard_proteins_by_length(protein_names, shard_index, num_shards):
    """
    Sort proteins by length (ascending) and distribute via round-robin
    to balance total workload across shards.
    """
    lengths = get_protein_lengths(protein_names)
    sorted_names = sorted(protein_names, key=lambda p: lengths.get(p, 0))
    # Round-robin (deal cards): shard 0 gets indices 0, num_shards, 2*num_shards, ...
    shard_names = sorted_names[shard_index::num_shards]
    total_residues = sum(lengths.get(p, 0) for p in shard_names)
    logger.info(f"Shard {shard_index}/{num_shards}: {len(shard_names)} proteins, "
                f"~{total_residues} total residues")
    return shard_names


def find_proteins_needing_inference(csv_file, csv_column, inference_config):
    """Find proteins from CSV that need inference (no PDB files generated yet)."""
    # Get proteins from CSV file
    csv_proteins = get_protein_names(csv_file, csv_column)
    
    inference_base_dir = os.path.join(PROTEINA_BASE_DIR, 'inference', inference_config)
    
    proteins_needing_inference = []
    
    for protein_name in csv_proteins:
        protein_dir = Path(inference_base_dir) / protein_name
        
        if protein_dir.exists() and protein_dir.is_dir():
            # Check if there are PDB files (inference completed)
            pdb_files = list(protein_dir.glob(f"{protein_name}_*.pdb"))
            
            if not pdb_files:
                # Directory exists but no PDB files
                proteins_needing_inference.append(protein_name)
            else:
                logger.info(f"Inference already completed for {protein_name} ({len(pdb_files)} PDB files), skipping")
        else:
            # Directory doesn't exist, needs inference
            proteins_needing_inference.append(protein_name)
    
    return proteins_needing_inference

def main():
    parser = argparse.ArgumentParser(description='Parallel Proteina Inference Pipeline')
    parser.add_argument('--csv_file', required=True, help='Path to CSV file with protein data')
    parser.add_argument('--csv_column', required=True, help='Column name in CSV file to use for protein selection')
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
        '--shard_index', type=int, default=None,
        help='Shard index (0-based). Auto-detected from SLURM_ARRAY_TASK_ID if not set.',
    )
    parser.add_argument(
        '--num_shards', type=int, default=None,
        help='Total number of shards. Auto-detected from SLURM_ARRAY_TASK_COUNT if not set.',
    )

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
    
    # Get protein names
    if args.skip_existing:
        protein_names = find_proteins_needing_inference(args.csv_file, args.csv_column, args.inference_config)
        logger.info(f"Found {len(protein_names)} proteins needing inference (from CSV file)")
    else:
        protein_names = get_protein_names(args.csv_file, args.csv_column)
        logger.info(f"Found {len(protein_names)} proteins to process")
    
    # Note: Multi-character chain IDs are now supported in AF2Rank via chain extraction
    # Proteina PT conversion also handles them correctly

    # --- Sharding: distribute proteins across SLURM array tasks ---
    shard_index = args.shard_index
    num_shards = args.num_shards
    # Auto-detect from SLURM array env vars, or MIT SuperCloud LLsub env vars
    if shard_index is None:
        for var in ('SLURM_ARRAY_TASK_ID', 'LLSUB_RANK'):
            if var in os.environ:
                shard_index = int(os.environ[var])
                break
    if num_shards is None:
        for var in ('SLURM_ARRAY_TASK_COUNT', 'LLSUB_SIZE'):
            if var in os.environ:
                num_shards = int(os.environ[var])
                break

    if shard_index is not None and num_shards is not None:
        logger.info(f"Sharding enabled: shard {shard_index} of {num_shards}")
        protein_names = shard_proteins_by_length(protein_names, shard_index, num_shards)
    # ---

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
    
    # Create shared counter for worker initialization
    manager = mp.Manager()
    worker_counter = manager.Value('i', 0)
    worker_lock = manager.Lock()
    
    executor = ProcessPoolExecutor(
        max_workers=args.num_gpus,
        initializer=worker_init_proteina,
        initargs=(worker_counter, worker_lock, args.num_gpus)
    )
    
    try:
        # Submit all jobs (GPU assignment happens via worker init)
        work_items = [(protein_name, args.csv_file, args.csv_column, args.cif_dir, args.inference_config, args.usalign_path, args.force_compile, args.skip_pt_conversion)
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
    
    # Return appropriate exit code
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
