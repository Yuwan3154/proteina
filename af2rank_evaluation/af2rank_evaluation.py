#!/usr/bin/env python3
"""
AF2Rank Evaluation Pipeline - Individual Step Runner

This script runs individual steps of the evaluation pipeline.
Called by the parallel processing scripts.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append('/home/jupyter-chenxi/proteina')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_cif_to_pt_conversion(csv_file, cif_dir, output_dir):
    """Run CIF to PT conversion."""
    cmd = [
        'python', 'cif_to_pt_converter.py',
        '--csv_file', csv_file,
        '--cif_dir', cif_dir,
        '--output_dir', output_dir
    ]
    
    logger.info(f"Running CIF to PT conversion: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"CIF to PT conversion failed: {result.stderr}")
        return False
    
    logger.info("CIF to PT conversion completed successfully")
    return True

def run_proteina_inference(csv_file, inference_config, output_dir):
    """Run Proteina inference."""
    import pandas as pd
    
    # Read the CSV file to get the actual protein name
    df = pd.read_csv(csv_file)
    if 'natives_rcsb' not in df.columns or df.empty:
        logger.error(f"CSV file {csv_file} does not contain 'natives_rcsb' column or is empty")
        return False
    
    # Get the first protein name from the CSV
    protein_name = df['natives_rcsb'].dropna().iloc[0]
    logger.info(f"Extracted protein name: {protein_name}")
    
    # Change to the proteina directory for inference
    proteina_dir = "/home/jupyter-chenxi/proteina"
    
    cmd = [
        'python', 'proteinfoundation/inference.py',
        '--pt', protein_name,
        '--config_name', inference_config
    ]
    
    logger.info(f"Running Proteina inference in {proteina_dir}: {' '.join(cmd)}")
    
    # Set environment variable for output directory if needed
    env = os.environ.copy()
    env['INFERENCE_OUTPUT_DIR'] = output_dir
    
    # Run inference from the proteina directory
    result = subprocess.run(cmd, cwd=proteina_dir, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        logger.error(f"Proteina inference failed: {result.stderr}")
        logger.error(f"Proteina inference stdout: {result.stdout}")
        return False
    
    logger.info("Proteina inference completed successfully")
    return True

def run_usalign_evaluation(csv_file, cif_dir, inference_config, output_dir, usalign_path):
    """Run USalign evaluation."""
    logger.info("USalign evaluation functionality not implemented in this minimal version")
    logger.info("USalign evaluation will be handled separately if needed")
    return True

def main():
    parser = argparse.ArgumentParser(description='AF2Rank Evaluation Pipeline - Individual Steps')
    parser.add_argument('--csv_file', required=True, help='Path to CSV file')
    parser.add_argument('--cif_dir', required=True, help='Directory containing CIF files')
    parser.add_argument('--inference_config', required=True, help='Inference configuration name')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--step', choices=['convert', 'inference', 'evaluate'], required=True,
                       help='Which step to run')
    parser.add_argument('--usalign_path', help='Path to USalign executable')
    parser.add_argument('--disable_af2rank', action='store_true', help='Disable AF2Rank scoring')
    
    args = parser.parse_args()
    
    logger.info(f"Running step: {args.step}")
    logger.info(f"CSV file: {args.csv_file}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    if args.step == 'convert':
        success = run_cif_to_pt_conversion(args.csv_file, args.cif_dir, args.output_dir)
    
    elif args.step == 'inference':
        # First run CIF to PT conversion
        if run_cif_to_pt_conversion(args.csv_file, args.cif_dir, args.output_dir):
            # Then run Proteina inference
            success = run_proteina_inference(args.csv_file, args.inference_config, args.output_dir)
        else:
            success = False
    
    elif args.step == 'evaluate':
        if args.usalign_path:
            success = run_usalign_evaluation(args.csv_file, args.cif_dir, args.inference_config, 
                                           args.output_dir, args.usalign_path)
        else:
            logger.warning("USalign path not provided, skipping evaluation")
    
    if success:
        logger.info(f"Step '{args.step}' completed successfully")
        sys.exit(0)
    else:
        logger.error(f"Step '{args.step}' failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
