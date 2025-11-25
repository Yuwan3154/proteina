#!/usr/bin/env python3
"""
Parallel AF2Rank Scoring Script

This script handles parallelized AF2Rank scoring across multiple GPUs.
It takes inference results and compares them to ground truth structures.
"""

import os
import sys
import argparse
import pandas as pd
import subprocess
import multiprocessing as mp
import builtins
from pathlib import Path
import time
import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        logger.warning("\n⚠️  Interrupt received! Gracefully shutting down...")
        logger.warning("⚠️  Waiting for current tasks to complete...")
        logger.warning("⚠️  Press Ctrl+C again to force quit (not recommended)")
    else:
        logger.error("\n❌ Force quit requested. Terminating immediately...")
        sys.exit(1)

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

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    env_path = os.path.join(PROTEINA_BASE_DIR, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    load_dotenv = None

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

def find_reference_cif(protein_name, cif_dir):
    """Find the reference CIF file for a protein."""
    pdb_id = protein_name.split('_')[0]
    
    # Search in subdirectories
    cif_path = Path(cif_dir)
    for subdir in cif_path.iterdir():
        if subdir.is_dir():
            potential_cif = subdir / f"{pdb_id}.cif"
            if potential_cif.exists():
                return str(potential_cif)
    
    # Direct search
    potential_cif = cif_path / f"{pdb_id}.cif"
    if potential_cif.exists():
        return str(potential_cif)
    
    raise FileNotFoundError(f"CIF file not found for {pdb_id} in {cif_dir}")

def process_single_protein_af2rank(args):
    """Process AF2Rank scoring for a single protein."""
    protein_name, cif_dir, inference_config, recycles, gpu_id, regenerate_plots = args
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    logger.info(f"[GPU {gpu_id}] Starting AF2Rank scoring for {protein_name}")
    
    # Find reference CIF file
    reference_cif = find_reference_cif(protein_name, cif_dir)
    logger.info(f"[GPU {gpu_id}] Found reference CIF: {reference_cif}")
    
    # Get inference output directory
    inference_output_dir = generate_protein_output_dir(inference_config, protein_name)
    
    if not os.path.exists(inference_output_dir):
        raise FileNotFoundError(f"Inference output directory not found: {inference_output_dir}")
    
    # Check if PDB files exist
    pdb_files = list(Path(inference_output_dir).glob(f"{protein_name}_*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {inference_output_dir}")
    
    logger.info(f"[GPU {gpu_id}] Found {len(pdb_files)} PDB files to score")
    
    # Check if we should do plot-only regeneration
    af2rank_csv = Path(inference_output_dir) / "af2rank_analysis" / f"af2rank_scores_{protein_name}.csv"
    if regenerate_plots and af2rank_csv.exists():
        logger.info(f"[GPU {gpu_id}] Skipping plot regeneration (will be handled by CPU workers)")
        return {'protein': protein_name, 'gpu': gpu_id, 'status': 'skipped', 'reason': 'plot_regeneration_deferred'}
    else:
        # Run full AF2Rank scoring
        logger.info(f"[GPU {gpu_id}] Running full AF2Rank scoring for {protein_name}")
        result = run_af2rank_scoring(protein_name, reference_cif, inference_output_dir, recycles)
    
    if result.returncode != 0:
        raise Exception(f"AF2Rank scoring failed with returncode {result.returncode}")
    
    logger.info(f"[GPU {gpu_id}] ✅ AF2Rank scoring completed for {protein_name}")
    
    # Parse output for summary info
    af2rank_dir = f"{inference_output_dir}/af2rank_analysis"
    summary_file = f"{af2rank_dir}/af2rank_summary_{protein_name}.json"
    
    summary_info = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_info = json.load(f)
    
    return {
        'protein': protein_name, 
        'gpu': gpu_id, 
        'status': 'success', 
        'output_dir': af2rank_dir,
        'summary': summary_info
    }

def run_af2rank_scoring(protein_name, reference_cif, inference_output_dir, recycles=3):
    """Run AF2Rank scoring for a single protein."""
    # Use the colabdesign environment wrapper to ensure correct Python is used
    wrapper_script = os.path.join(PROTEINA_BASE_DIR, 'af2rank_evaluation', 'run_with_colabdesign_env.sh')
    
    # Prepare the Python code to execute
    python_code = f"""
import sys
import os
import jax
sys.path.append('{os.path.join(PROTEINA_BASE_DIR, 'af2rank_evaluation')}')

# Import AF2Rank components
from af2rank_scorer import run_af2rank_analysis

print('Starting AF2Rank scoring for {protein_name} with ColabDesign environment')
print('Using device: ' + str(jax.devices))

protein_id = '{protein_name}'
pdb_id, chain_id = protein_id.split('_')

# Set environment variables for AF2
os.environ['ALPHAFOLD_DATA_DIR'] = os.path.expanduser('~/params')
os.environ['AF_PARAMS_DIR'] = os.path.expanduser('~/params')

# Create AF2Rank output directory
af2rank_dir = '{inference_output_dir}/af2rank_analysis'
os.makedirs(af2rank_dir, exist_ok=True)

# Run complete AF2Rank analysis with new metrics
result = run_af2rank_analysis(
    protein_id=protein_id,
    reference_cif='{reference_cif}',
    inference_output_dir='{inference_output_dir}',
    output_dir=af2rank_dir,
    chain=chain_id,
    recycles={recycles},
    verbose=False,
    regenerate_summary=True
)

if result:
    print(f'Successfully completed AF2Rank analysis for {{protein_id}}')
    print(f'Results saved to: {{af2rank_dir}}')
else:
    print(f'ERROR: AF2Rank analysis failed for {{protein_id}}')
    sys.exit(1)
"""
    
    # Use wrapper script to run in colabdesign environment
    cmd = [wrapper_script, 'python', '-c', python_code]
    # Don't capture output - let it stream to terminal for real-time progress and full error tracebacks
    result = subprocess.run(
        cmd, 
        cwd=os.path.join(PROTEINA_BASE_DIR, 'af2rank_evaluation')
    )
    return result

def run_af2rank_plot_only(protein_name, reference_cif, inference_output_dir, recycles=3):
    """Regenerate plots only for existing AF2Rank scores and update summary."""
    # Use the colabdesign environment wrapper to ensure correct Python is used
    wrapper_script = os.path.join(PROTEINA_BASE_DIR, 'af2rank_evaluation', 'run_with_colabdesign_env.sh')
    
    # Prepare the Python code to execute
    python_code = f"""
import sys
import os
sys.path.append('{os.path.join(PROTEINA_BASE_DIR, 'af2rank_evaluation')}')

# Import AF2Rank components
from af2rank_scorer import run_af2rank_plot_only

print('Regenerating plots and summary for {protein_name}', flush=True)

protein_id = '{protein_name}'
pdb_id, chain_id = protein_id.split('_')

# Run plot regeneration with summary update
af2rank_dir = '{inference_output_dir}/af2rank_analysis'
result = run_af2rank_plot_only(
    protein_id=protein_id,
    reference_cif='{reference_cif}',
    inference_output_dir='{inference_output_dir}',
    output_dir=af2rank_dir,
    chain=chain_id,
    recycles={recycles},
    regenerate_summary=True
)

if result:
    print(f'Successfully regenerated plots and summary for {{protein_id}}', flush=True)
    print(f'Results saved to: {{af2rank_dir}}', flush=True)
else:
    print(f'ERROR: Failed to regenerate plots for {{protein_id}}', flush=True)
    sys.exit(1)
"""
    
    # Use wrapper script to run in colabdesign environment
    cmd = [wrapper_script, 'python', '-u', '-c', python_code]
    result = subprocess.run(
        cmd, 
        cwd=os.path.join(PROTEINA_BASE_DIR, 'af2rank_evaluation'),
        capture_output=False, 
        text=True
    )
    return result


def process_single_protein_plot_regeneration(args):
    """Process plot regeneration for a single protein (CPU-only, no GPU needed)."""
    protein_name, cif_dir, inference_config, recycles = args
    
    logger.info(f"[CPU] Regenerating plots and summary for {protein_name}")
    
    # Find reference CIF file
    reference_cif = find_reference_cif(protein_name, cif_dir)
    
    # Generate output directory
    inference_output_dir = generate_protein_output_dir(inference_config, protein_name)
    
    # Check if CSV file exists
    af2rank_csv = Path(inference_output_dir) / "af2rank_analysis" / f"af2rank_scores_{protein_name}.csv"
    if not af2rank_csv.exists():
        logger.warning(f"[CPU] No AF2Rank scores found for {protein_name}")
        return {'protein': protein_name, 'status': 'skipped', 'reason': 'no_csv_file'}
    
    # Run plot regeneration (CPU-only)
    result = run_af2rank_plot_only(protein_name, reference_cif, inference_output_dir, recycles)
    
    if result.returncode == 0:
        logger.info(f"[CPU] ✅ Plot regeneration completed for {protein_name}")
        return {'protein': protein_name, 'status': 'success'}
    else:
        raise Exception(f"Plot regeneration failed with returncode {result.returncode}")


# Worker initialization function (must be at module level for pickling)
def worker_init_af2rank(counter, lock, num_gpus):
    """Initialize worker with a specific GPU assignment."""
    with lock:
        worker_id = counter.value
        counter.value += 1
    
    gpu_id = worker_id % num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Store GPU ID in a process-local variable
    builtins._worker_gpu_id = gpu_id
    logger.info(f"AF2Rank Worker {worker_id} initialized with GPU {gpu_id}")

# Wrapper function (must be at module level for pickling)
def process_single_protein_af2rank_wrapper(args_tuple):
    """Wrapper that uses the GPU assigned during worker init."""
    protein_name, cif_dir, inference_config, recycles = args_tuple
    
    # Get GPU ID from process-local variable set by worker_init
    gpu_id = getattr(builtins, '_worker_gpu_id', 0)
    
    # Call the actual processing function
    full_args = (protein_name, cif_dir, inference_config, recycles, gpu_id, False)
    return process_single_protein_af2rank(full_args)

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
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    proteins = df[csv_column].dropna().unique().tolist()
    return [p.strip() for p in proteins if p and p.strip()]

def find_proteins_needing_af2rank(csv_file, csv_column, inference_config, regenerate_plots=False):
    """Find proteins from CSV that need AF2Rank scoring or plot regeneration."""
    # Get proteins from CSV file
    csv_proteins = get_protein_names(csv_file, csv_column)
    
    inference_base_dir = os.path.join(PROTEINA_BASE_DIR, 'inference', inference_config)
    
    if not os.path.exists(inference_base_dir):
        return []
    
    proteins_needing_work = []
    
    for protein_name in csv_proteins:
        protein_dir = Path(inference_base_dir) / protein_name
        
        if protein_dir.exists() and protein_dir.is_dir():
            # Check if there are PDB files (inference completed)
            pdb_files = list(protein_dir.glob(f"{protein_name}_*.pdb"))
            
            if pdb_files:
                # Check if AF2Rank scoring is already completed
                af2rank_dir = protein_dir / "af2rank_analysis"
                af2rank_csv = af2rank_dir / f"af2rank_scores_{protein_name}.csv"
                
                if not af2rank_csv.exists():
                    # Has inference but no AF2Rank scoring
                    proteins_needing_work.append(protein_name)
                elif regenerate_plots:
                    # Has scoring but user wants to regenerate plots
                    logger.info(f"AF2Rank scoring exists for {protein_name}, but regenerating plots")
                    proteins_needing_work.append(protein_name)
                else:
                    logger.info(f"AF2Rank scoring already completed for {protein_name}, skipping")
    
    return proteins_needing_work

def main():
    parser = argparse.ArgumentParser(description='Parallel AF2Rank Scoring Pipeline')
    parser.add_argument('--csv_file', required=True, help='Path to CSV file with protein data')
    parser.add_argument('--csv_column', required=True, help='Column name in CSV file to use for protein selection')
    parser.add_argument('--cif_dir', required=True, help='Directory containing reference CIF files')
    parser.add_argument('--inference_config', required=True, help='Inference configuration name')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--recycles', type=int, default=3, help='Number of AF2 recycles')
    parser.add_argument('--filter_existing', action='store_true', 
                       help='Only process proteins that have completed inference')
    parser.add_argument('--regenerate_plots', action='store_true',
                       help='Regenerate plots even if AF2Rank CSV already exists')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    if not os.path.exists(args.cif_dir):
        logger.error(f"CIF directory not found: {args.cif_dir}")
        sys.exit(1)
    
    # Get protein names
    if args.filter_existing:
        protein_names = find_proteins_needing_af2rank(args.csv_file, args.csv_column, args.inference_config, args.regenerate_plots)
        if args.regenerate_plots:
            logger.info(f"Found {len(protein_names)} proteins needing AF2Rank scoring or plot regeneration (from CSV file)")
        else:
            logger.info(f"Found {len(protein_names)} proteins needing AF2Rank scoring (from CSV file)")
    else:
        protein_names = get_protein_names(args.csv_file, args.csv_column)
        logger.info(f"Found {len(protein_names)} proteins in CSV file")
    
    if not protein_names:
        logger.warning("No proteins to process")
        sys.exit(0)
    
    # Note: Multi-character chain IDs are now supported via chain extraction and renaming
    # No need to filter them out anymore!
    
    logger.info(f"Proteins to score: {protein_names}")
    
    # Check available GPUs
    try:
        gpu_count = int(subprocess.check_output(['nvidia-smi', '--list-gpus']).decode().count('\n'))
    except:
        gpu_count = 1
    
    if args.num_gpus > gpu_count:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {gpu_count} available, using {gpu_count}")
        args.num_gpus = gpu_count
    
    logger.info(f"Using {args.num_gpus} GPU(s) for parallel AF2Rank scoring")

    # Separate proteins needing scoring vs plot regeneration
    proteins_needing_scoring = []
    proteins_needing_plots = []

    for protein_name in protein_names:
        protein_dir = Path(os.path.join(PROTEINA_BASE_DIR, 'inference', args.inference_config, protein_name))
        af2rank_csv = protein_dir / "af2rank_analysis" / f"af2rank_scores_{protein_name}.csv"

        if af2rank_csv.exists() and args.regenerate_plots:
            proteins_needing_plots.append(protein_name)
        elif not af2rank_csv.exists():
            proteins_needing_scoring.append(protein_name)

    logger.info(f"Proteins needing AF2Rank scoring: {len(proteins_needing_scoring)}")
    logger.info(f"Proteins needing plot regeneration: {len(proteins_needing_plots)}")

    # Process proteins in parallel with proper GPU assignment
    start_time = time.time()
    all_results = []

    # GPU workers for AF2Rank scoring (fixed GPU assignment)
    if proteins_needing_scoring:
        logger.info(f"Starting GPU-based AF2Rank scoring with {args.num_gpus} workers")

        # Create shared counter for worker initialization
        manager = mp.Manager()
        worker_counter = manager.Value('i', 0)
        worker_lock = manager.Lock()

        executor = ProcessPoolExecutor(
            max_workers=args.num_gpus,
            initializer=worker_init_af2rank,
            initargs=(worker_counter, worker_lock, args.num_gpus)
        )

        try:
            # Submit all jobs (GPU assignment happens via worker init)
            scoring_work_items = [(protein_name, args.cif_dir, args.inference_config, args.recycles)
                                  for protein_name in proteins_needing_scoring]
            future_to_protein = {executor.submit(process_single_protein_af2rank_wrapper, item): scoring_work_items[i][0]
                                for i, item in enumerate(scoring_work_items)}
            
            for future in as_completed(future_to_protein):
                # Check for shutdown request
                if shutdown_requested:
                    logger.warning("⚠️  Shutdown requested, cancelling remaining GPU tasks...")
                    for f in future_to_protein:
                        f.cancel()
                    break
                
                protein_name = future_to_protein[future]
                # No try-except - let exceptions propagate with full tracebacks
                result = future.result()
                all_results.append(result)
                
                completed = len([r for r in all_results if r['status'] == 'success'])
                total = len(scoring_work_items)
                elapsed = time.time() - start_time
                
                if result['status'] == 'success':
                    summary = result.get('summary', {})
                    successful_scores = summary.get('successful_scores', 0)
                    total_structures = summary.get('total_structures', 0)
                    logger.info(f"✅ GPU Progress: {completed}/{total} proteins completed ({successful_scores}/{total_structures} structures scored, {elapsed:.1f}s elapsed)")
                else:
                    logger.error(f"❌ {protein_name} failed: {result.get('error', 'Unknown error')}")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            logger.info("✓ GPU executor shut down cleanly")
    
    # CPU workers for plot regeneration (much faster)
    if proteins_needing_plots and not shutdown_requested:
        logger.info(f"Starting CPU-based plot regeneration with {min(len(proteins_needing_plots), 8)} workers")
        
        plot_work_items = []
        for protein_name in proteins_needing_plots:
            plot_work_items.append((protein_name, args.cif_dir, args.inference_config, args.recycles))
        
        # Use more CPU workers since this is lightweight
        max_cpu_workers = min(len(proteins_needing_plots), 8)
        
        executor = ProcessPoolExecutor(max_workers=max_cpu_workers)
        try:
            future_to_protein = {executor.submit(process_single_protein_plot_regeneration, item): item[0] for item in plot_work_items}
            
            for future in as_completed(future_to_protein):
                # Check for shutdown request
                if shutdown_requested:
                    logger.warning("⚠️  Shutdown requested, cancelling remaining CPU tasks...")
                    for f in future_to_protein:
                        f.cancel()
                    break
                
                protein_name = future_to_protein[future]
                # No try-except - let exceptions propagate with full tracebacks
                result = future.result()
                all_results.append(result)
                
                completed = len([r for r in all_results if r['status'] == 'success'])
                total = len(plot_work_items)
                elapsed = time.time() - start_time
                
                if result['status'] == 'success':
                    logger.info(f"✅ CPU Progress: {completed}/{total} plots regenerated ({elapsed:.1f}s elapsed)")
                else:
                    logger.error(f"❌ {protein_name} plot regeneration failed: {result.get('error', 'Unknown error')}")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            logger.info("✓ CPU executor shut down cleanly")
    
    # Summary
    total_time = time.time() - start_time
    successful = len([r for r in all_results if r['status'] == 'success'])
    failed = len([r for r in all_results if r['status'] == 'failed'])
    skipped = len([r for r in all_results if r['status'] == 'skipped'])
    
    # Calculate total structures scored
    total_structures_scored = 0
    for result in all_results:
        if result['status'] == 'success':
            summary = result.get('summary', {})
            total_structures_scored += summary.get('successful_scores', 0)
    
    logger.info(f"\n📊 AF2Rank Processing Results:")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"⏭️  Skipped: {skipped}")
    logger.info(f"📈 Total structures scored: {total_structures_scored}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s")
    logger.info(f"📊 Average time per protein: {total_time/len(all_results):.1f}s")
    
    if failed:
        logger.error("Failed proteins:")
        for result in all_results:
            if result['status'] == 'failed':
                logger.error(f"  - {result['protein']}: {result.get('error', 'Unknown error')}")
    
    # Check if shutdown was requested
    if shutdown_requested:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        logger.info(f"Processed {successful} proteins before interruption")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    if successful:
        logger.info(f"🎉 Successfully processed {successful} proteins!")
        logger.info(f"📁 Results saved to: {os.path.join(PROTEINA_BASE_DIR, 'inference', args.inference_config)}/")
    
    sys.exit(0 if not failed else 1)

if __name__ == '__main__':
    main()
