#!/usr/bin/env python3
"""
Complete AF2Rank Evaluation Pipeline

This script runs the complete pipeline:
1. Proteina inference (parallel across GPUs)
2. AF2Rank scoring (parallel across GPUs)

Usage:
    python run_full_pipeline.py --csv_file data.csv --cif_dir /path/to/cif --inference_config config_name --num_gpus 4
"""

import os
import sys
import argparse
import subprocess
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_proteina_inference(csv_file, cif_dir, inference_config, num_gpus, usalign_path=None):
    """Run the Proteina inference pipeline."""
    logger.info("🧬 Starting Proteina inference pipeline...")
    
    cmd = [
        'python', 'parallel_proteina_inference.py',
        '--csv_file', csv_file,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        '--skip_existing'  # Always skip existing to avoid re-processing
    ]
    
    if usalign_path:
        cmd.extend(['--usalign_path', usalign_path])
    
    # Run in proteina environment
    conda_cmd = f"source /opt/tljh/user/etc/profile.d/conda.sh && conda activate proteina && {' '.join(cmd)}"
    
    result = subprocess.run(conda_cmd, shell=True, executable='/bin/bash')
    return result.returncode == 0

def run_af2rank_scoring(csv_file, cif_dir, inference_config, num_gpus, recycles=3, regenerate_plots=False):
    """Run the AF2Rank scoring pipeline."""
    logger.info("⚡ Starting AF2Rank scoring pipeline...")
    
    cmd = [
        'python', 'parallel_af2rank_scoring.py',
        '--csv_file', csv_file,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        '--recycles', str(recycles),
        '--filter_existing'  # Always filter to only score proteins needing AF2Rank
    ]
    
    if regenerate_plots:
        cmd.append('--regenerate_plots')
    
    # Run in colabdesign environment  
    conda_cmd = f"source /opt/tljh/user/etc/profile.d/conda.sh && conda activate colabdesign && {' '.join(cmd)}"
    
    result = subprocess.run(conda_cmd, shell=True, executable='/bin/bash')
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Complete AF2Rank Evaluation Pipeline')
    parser.add_argument('--csv_file', required=True, help='Path to CSV file with protein data')
    parser.add_argument('--cif_dir', required=True, help='Directory containing CIF files')
    parser.add_argument('--inference_config', required=True, help='Inference configuration name')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--recycles', type=int, default=3, help='Number of AF2 recycles for scoring')
    parser.add_argument('--usalign_path', help='Path to USalign executable')
    parser.add_argument('--skip_inference', action='store_true', 
                       help='Skip Proteina inference (only run AF2Rank scoring)')
    parser.add_argument('--skip_af2rank', action='store_true',
                       help='Skip AF2Rank scoring (only run Proteina inference)')
    parser.add_argument('--regenerate_plots', action='store_true',
                       help='Regenerate AF2Rank plots even if scoring already completed')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    if not os.path.exists(args.cif_dir):
        logger.error(f"CIF directory not found: {args.cif_dir}")
        sys.exit(1)
    
    # Check available GPUs
    try:
        gpu_count = int(subprocess.check_output(['nvidia-smi', '--list-gpus']).decode().count('\n'))
    except:
        gpu_count = 1
        logger.warning("Could not detect GPUs, assuming 1 GPU available")
    
    if args.num_gpus > gpu_count:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {gpu_count} available, using {gpu_count}")
        args.num_gpus = gpu_count
    
    start_time = time.time()
    
    logger.info(f"🚀 Starting complete AF2Rank evaluation pipeline")
    logger.info(f"📊 CSV file: {args.csv_file}")
    logger.info(f"📂 CIF directory: {args.cif_dir}")
    logger.info(f"⚙️  Inference config: {args.inference_config}")
    logger.info(f"🔥 GPUs: {args.num_gpus}")
    logger.info(f"🔄 AF2Rank recycles: {args.recycles}")
    
    success = True
    
    # Step 1: Proteina Inference
    if not args.skip_inference:
        logger.info("\n" + "="*60)
        logger.info("STEP 1: PROTEINA INFERENCE")
        logger.info("="*60)
        
        inference_success = run_proteina_inference(
            args.csv_file, 
            args.cif_dir, 
            args.inference_config, 
            args.num_gpus,
            args.usalign_path
        )
        
        if inference_success:
            logger.info("✅ Proteina inference completed successfully")
        else:
            logger.error("❌ Proteina inference failed")
            success = False
    else:
        logger.info("⏭️  Skipping Proteina inference")
    
    # Step 2: AF2Rank Scoring
    if not args.skip_af2rank and success:
        logger.info("\n" + "="*60)
        logger.info("STEP 2: AF2RANK SCORING")
        logger.info("="*60)
        
        af2rank_success = run_af2rank_scoring(
            args.csv_file,
            args.cif_dir,
            args.inference_config,
            args.num_gpus,
            args.recycles,
            args.regenerate_plots
        )
        
        if af2rank_success:
            logger.info("✅ AF2Rank scoring completed successfully")
        else:
            logger.error("❌ AF2Rank scoring failed")
            success = False
    elif args.skip_af2rank:
        logger.info("⏭️  Skipping AF2Rank scoring")
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    
    if success:
        logger.info(f"🎉 Pipeline completed successfully in {total_time:.1f}s")
        logger.info(f"📁 Results saved to: /home/jupyter-chenxi/proteina/inference/{args.inference_config}/")
    else:
        logger.error(f"💥 Pipeline failed after {total_time:.1f}s")
    
    logger.info("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
