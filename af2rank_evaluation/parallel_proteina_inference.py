#!/usr/bin/env python3
"""
Parallel Proteina Inference Script

This script handles parallelized protein structure generation using Proteina across multiple GPUs.
It processes all proteins from a CSV file and distributes them across available GPUs.
"""

import os
import sys
import argparse
import pandas as pd
import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

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

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv('/home/jupyter-chenxi/proteina/.env')
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

def setup_conda_environment():
    """Setup conda environment for proteina."""
    conda_init_script = None
    for script_path in ['/opt/tljh/user/etc/profile.d/conda.sh', '/opt/conda/etc/profile.d/conda.sh']:
        if os.path.exists(script_path):
            conda_init_script = script_path
            break
    
    if conda_init_script:
        return f"source {conda_init_script} && conda activate proteina"
    else:
        return "conda activate proteina"

def generate_protein_output_dir(inference_config, protein_name):
    """Generate consistent output directory path for a protein."""
    return f"/home/jupyter-chenxi/proteina/inference/{inference_config}/{protein_name}"

def create_single_protein_csv(csv_file, protein_name, output_dir):
    """Create a single-protein CSV file for individual processing."""
    df = pd.read_csv(csv_file)
    
    # Filter for this specific protein
    protein_df = df[df['natives_rcsb'] == protein_name]
    
    if protein_df.empty:
        raise ValueError(f"Protein {protein_name} not found in CSV file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write single-protein CSV
    single_csv_path = os.path.join(output_dir, 'single_protein.csv')
    protein_df.to_csv(single_csv_path, index=False)
    
    return single_csv_path

def run_cif_to_pt_conversion(csv_file, cif_dir, output_dir):
    """Run CIF to PT conversion step."""
    conda_cmd = setup_conda_environment()
    
    python_cmd = f"""
{conda_cmd} && cd /home/jupyter-chenxi/proteina/af2rank_evaluation && python -c "
import sys
sys.path.append('/home/jupyter-chenxi/proteina')
from cif_to_pt_converter import convert_from_csv

convert_from_csv(
    csv_file='{csv_file}',
    cif_dir='{cif_dir}', 
    output_dir='{output_dir}/processed'
)
print('CIF to PT conversion completed')
"
"""
    
    result = subprocess.run(python_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    return result

def run_proteina_inference(csv_file, cif_dir, inference_config, output_dir):
    """Run Proteina inference step."""
    conda_cmd = setup_conda_environment()
    
    af2rank_eval_path = "/home/jupyter-chenxi/proteina/af2rank_evaluation"
    
    cmd = f"""
{conda_cmd} && cd {af2rank_eval_path} && python af2rank_evaluation.py \
    --csv_file "{csv_file}" \
    --cif_dir "{cif_dir}" \
    --inference_config "{inference_config}" \
    --output_dir "{output_dir}" \
    --step inference \
    --disable_af2rank
"""
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    return result

def run_usalign_evaluation(csv_file, cif_dir, inference_config, output_dir, usalign_path):
    """Run USalign evaluation step."""
    conda_cmd = setup_conda_environment()
    
    af2rank_eval_path = "/home/jupyter-chenxi/proteina/af2rank_evaluation"
    
    cmd = f"""
{conda_cmd} && cd {af2rank_eval_path} && python af2rank_evaluation.py \
    --csv_file "{csv_file}" \
    --cif_dir "{cif_dir}" \
    --inference_config "{inference_config}" \
    --output_dir "{output_dir}" \
    --step evaluate \
    --usalign_path "{usalign_path}" \
    --disable_af2rank
"""
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    return result

def process_single_protein(args):
    """Process a single protein through the entire Proteina pipeline."""
    protein_name, csv_file, cif_dir, inference_config, usalign_path, gpu_id = args
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    logger.info(f"[GPU {gpu_id}] Starting processing for {protein_name}")
    
    # Generate output directory for this protein
    protein_output_dir = generate_protein_output_dir(inference_config, protein_name)
    
    logger.info(f"[GPU {gpu_id}] {protein_name} -> {protein_output_dir}")
    
    # Create single-protein CSV file
    logger.info(f"[GPU {gpu_id}] Creating single_protein.csv for {protein_name}")
    single_csv_path = create_single_protein_csv(csv_file, protein_name, protein_output_dir)
    
    # Step 1: CIF to PT conversion
    logger.info(f"[GPU {gpu_id}] Step 1: CIF to PT conversion for {protein_name}")
    result = run_cif_to_pt_conversion(single_csv_path, cif_dir, protein_output_dir)
    
    if result.returncode != 0:
        logger.error(f"[GPU {gpu_id}] CIF to PT conversion failed for {protein_name}")
        logger.error(f"[GPU {gpu_id}] STDOUT: {result.stdout}")
        logger.error(f"[GPU {gpu_id}] STDERR: {result.stderr}")
        raise Exception(f"CIF to PT conversion failed: {result.stderr}")
    
    logger.info(f"[GPU {gpu_id}] ✅ CIF to PT conversion completed for {protein_name}")
    
    # Step 2: Proteina inference
    logger.info(f"[GPU {gpu_id}] Step 2: Running Proteina inference for {protein_name}")
    result = run_proteina_inference(single_csv_path, cif_dir, inference_config, protein_output_dir)
    
    if result.returncode != 0:
        logger.error(f"[GPU {gpu_id}] Proteina inference failed for {protein_name}")
        logger.error(f"[GPU {gpu_id}] STDOUT: {result.stdout}")
        logger.error(f"[GPU {gpu_id}] STDERR: {result.stderr}")
        raise Exception(f"Proteina inference failed: {result.stderr}")
    
    logger.info(f"[GPU {gpu_id}] ✅ Proteina inference completed for {protein_name}")
    
    # Step 3: USalign evaluation
    if usalign_path:
        logger.info(f"[GPU {gpu_id}] Step 3: Running USalign evaluation for {protein_name}")
        result = run_usalign_evaluation(single_csv_path, cif_dir, inference_config, protein_output_dir, usalign_path)
        
        if result.returncode != 0:
            logger.warning(f"[GPU {gpu_id}] USalign evaluation failed for {protein_name}: {result.stderr}")
        else:
            logger.info(f"[GPU {gpu_id}] ✅ USalign evaluation completed for {protein_name}")
    
    logger.info(f"[GPU {gpu_id}] ✅ Successfully completed {protein_name}")
    return {'protein': protein_name, 'gpu': gpu_id, 'status': 'success', 'output_dir': protein_output_dir}

def get_protein_names(csv_file):
    """Extract protein names from CSV file."""
    df = pd.read_csv(csv_file)
    proteins = df['natives_rcsb'].dropna().unique().tolist()
    return [p for p in proteins if p.strip()]

def find_proteins_needing_inference(csv_file, inference_config):
    """Find proteins from CSV that need inference (no PDB files generated yet)."""
    # Get proteins from CSV file
    csv_proteins = get_protein_names(csv_file)
    
    inference_base_dir = f"/home/jupyter-chenxi/proteina/inference/{inference_config}"
    
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
    parser.add_argument('--cif_dir', required=True, help='Directory containing CIF files')
    parser.add_argument('--inference_config', required=True, help='Inference configuration name')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--usalign_path', help='Path to USalign executable')
    parser.add_argument('--skip_existing', action='store_true', 
                       help='Skip proteins that already have inference completed')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    if not os.path.exists(args.cif_dir):
        logger.error(f"CIF directory not found: {args.cif_dir}")
        sys.exit(1)
    
    # Get protein names
    if args.skip_existing:
        protein_names = find_proteins_needing_inference(args.csv_file, args.inference_config)
        logger.info(f"Found {len(protein_names)} proteins needing inference (from CSV file)")
    else:
        protein_names = get_protein_names(args.csv_file)
        logger.info(f"Found {len(protein_names)} proteins to process")
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
    
    # Create work items
    work_items = []
    for i, protein_name in enumerate(protein_names):
        gpu_id = i % args.num_gpus
        work_items.append((protein_name, args.csv_file, args.cif_dir, args.inference_config, args.usalign_path, gpu_id))
    
    # Process proteins in parallel
    start_time = time.time()
    results = []
    
    executor = ProcessPoolExecutor(max_workers=args.num_gpus)
    try:
        # Submit all jobs
        future_to_protein = {executor.submit(process_single_protein, item): item[0] for item in work_items}
        
        # Collect results as they complete
        for future in as_completed(future_to_protein):
            # Check for shutdown request
            if shutdown_requested:
                logger.warning("⚠️  Shutdown requested, cancelling remaining tasks...")
                for f in future_to_protein:
                    f.cancel()
                break
            
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
    
    logger.info(f"🎯 Pipeline completed in {total_time:.1f}s")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    
    if failed > 0:
        logger.info("Failed proteins:")
        for result in results:
            if result['status'] == 'failed':
                logger.info(f"  - {result['protein']}: {result.get('error', 'Unknown error')}")
    
    # Check if shutdown was requested
    if shutdown_requested:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        logger.info(f"Processed {successful} proteins before interruption")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    # Return appropriate exit code
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
