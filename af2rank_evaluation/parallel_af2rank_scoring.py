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
from pathlib import Path
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

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
    """Setup conda environment for colabdesign."""
    conda_init_script = None
    for script_path in ['/opt/tljh/user/etc/profile.d/conda.sh', '/opt/conda/etc/profile.d/conda.sh']:
        if os.path.exists(script_path):
            conda_init_script = script_path
            break
    
    if conda_init_script:
        return f"source {conda_init_script} && conda activate colabdesign"
    else:
        return "conda activate colabdesign"

def generate_protein_output_dir(inference_config, protein_name):
    """Generate consistent output directory path for a protein."""
    return f"/home/jupyter-chenxi/proteina/inference/{inference_config}/{protein_name}"

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

def run_af2rank_scoring(protein_name, reference_cif, inference_output_dir, recycles=3):
    """Run AF2Rank scoring for a single protein."""
    conda_cmd = setup_conda_environment()
    
    python_cmd = f"""
{conda_cmd} && cd /home/jupyter-chenxi/proteina/af2rank_evaluation && python -c "
import sys
import os
import json
from pathlib import Path
sys.path.append('/home/jupyter-chenxi/proteina/af2rank_evaluation')

# Import AF2Rank components
from af2rank_scorer import score_proteina_structures, plot_af2rank_results, save_af2rank_scores

print('Starting AF2Rank scoring for {protein_name} with ColabDesign environment')

protein_id = '{protein_name}'
pdb_id, chain_id = protein_id.split('_')

# Set environment variables for AF2
os.environ['ALPHAFOLD_DATA_DIR'] = os.path.expanduser('~/params')
os.environ['AF_PARAMS_DIR'] = os.path.expanduser('~/params')

# Score structures with AF2Rank  
scores = score_proteina_structures(
    protein_id=protein_id,
    reference_cif='{reference_cif}',
    inference_output_dir='{inference_output_dir}',
    chain=chain_id,
    recycles={recycles},
    verbose=False
)

if not scores:
    print(f'ERROR: No scores generated for {{protein_id}}')
    sys.exit(1)

# Create AF2Rank output directory
af2rank_dir = '{inference_output_dir}/af2rank_analysis'
os.makedirs(af2rank_dir, exist_ok=True)

# Save scores
scores_csv_path = save_af2rank_scores(scores, af2rank_dir, protein_id)

# Generate plots
plot_af2rank_results(scores, af2rank_dir, protein_id)

# Create summary
summary = {{
    'protein_id': protein_id,
    'total_structures': len(scores),
    'successful_scores': len([s for s in scores if 'error' not in s]),
    'failed_scores': len([s for s in scores if 'error' in s]),
    'reference_structure': '{reference_cif}',
    'inference_directory': '{inference_output_dir}',
    'af2rank_directory': af2rank_dir,
    'chain': chain_id,
    'recycles': {recycles},
    'scores_csv': scores_csv_path
}}

summary_file = os.path.join(af2rank_dir, f'af2rank_summary_{{protein_id}}.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

successful_count = len([s for s in scores if 'error' not in s])
total_count = len(scores)
print(f'AF2Rank scoring completed for {{protein_id}} ({{successful_count}}/{{total_count}} structures)')
print(f'Results saved to: {{af2rank_dir}}')
"
"""
    
    result = subprocess.run(python_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    return result

def run_af2rank_plot_only(protein_name, reference_cif, inference_output_dir):
    """Regenerate plots only for existing AF2Rank scores."""
    conda_cmd = setup_conda_environment()
    
    python_cmd = f"""
{conda_cmd} && cd /home/jupyter-chenxi/proteina/af2rank_evaluation && python -c "
import sys
import os
import json
import pandas as pd
from pathlib import Path
sys.path.append('/home/jupyter-chenxi/proteina/af2rank_evaluation')

# Import AF2Rank components
from af2rank_scorer import plot_af2rank_results

print('Regenerating plots for {protein_name}')

protein_id = '{protein_name}'
pdb_id, chain_id = protein_id.split('_')

# Load existing scores
af2rank_dir = '{inference_output_dir}/af2rank_analysis'
scores_csv = os.path.join(af2rank_dir, f'af2rank_scores_{{protein_id}}.csv')

if not os.path.exists(scores_csv):
    print(f'ERROR: No existing scores found at {{scores_csv}}')
    sys.exit(1)

# Load scores from CSV
df = pd.read_csv(scores_csv)
scores = df.to_dict('records')

print(f'Loaded {{len(scores)}} scores from existing CSV')

# Generate plots
correlations = plot_af2rank_results(scores, af2rank_dir, protein_id)

print(f'Regenerated plots for {{protein_id}} with {{len(correlations)}} correlations')
print(f'Plots saved to: {{af2rank_dir}}')
"
"""
    
    result = subprocess.run(python_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    return result

def process_single_protein_af2rank(args):
    """Process AF2Rank scoring for a single protein."""
    protein_name, cif_dir, inference_config, recycles, gpu_id, regenerate_plots = args
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    logger.info(f"[GPU {gpu_id}] Starting AF2Rank scoring for {protein_name}")
    
    try:
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
            logger.info(f"[GPU {gpu_id}] Regenerating plots only for {protein_name}")
            result = run_af2rank_plot_only(protein_name, reference_cif, inference_output_dir)
        else:
            # Run full AF2Rank scoring
            logger.info(f"[GPU {gpu_id}] Running full AF2Rank scoring for {protein_name}")
            result = run_af2rank_scoring(protein_name, reference_cif, inference_output_dir, recycles)
        
        if result.returncode != 0:
            raise Exception(f"AF2Rank scoring failed: {result.stderr}")
        
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
        
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] ❌ Failed AF2Rank scoring for {protein_name}: {e}")
        return {'protein': protein_name, 'gpu': gpu_id, 'status': 'failed', 'error': str(e)}

def get_protein_names(csv_file):
    """Extract protein names from CSV file."""
    df = pd.read_csv(csv_file)
    proteins = df['natives_rcsb'].dropna().unique().tolist()
    return [p for p in proteins if p.strip()]

def find_proteins_needing_af2rank(csv_file, inference_config, regenerate_plots=False):
    """Find proteins from CSV that need AF2Rank scoring or plot regeneration."""
    # Get proteins from CSV file
    csv_proteins = get_protein_names(csv_file)
    
    inference_base_dir = f"/home/jupyter-chenxi/proteina/inference/{inference_config}"
    
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
        protein_names = find_proteins_needing_af2rank(args.csv_file, args.inference_config, args.regenerate_plots)
        if args.regenerate_plots:
            logger.info(f"Found {len(protein_names)} proteins needing AF2Rank scoring or plot regeneration (from CSV file)")
        else:
            logger.info(f"Found {len(protein_names)} proteins needing AF2Rank scoring (from CSV file)")
    else:
        protein_names = get_protein_names(args.csv_file)
        logger.info(f"Found {len(protein_names)} proteins in CSV file")
    
    if not protein_names:
        logger.warning("No proteins to process")
        sys.exit(0)
    
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
    
    # Create work items
    work_items = []
    for i, protein_name in enumerate(protein_names):
        gpu_id = i % args.num_gpus
        work_items.append((protein_name, args.cif_dir, args.inference_config, args.recycles, gpu_id, args.regenerate_plots))
    
    # Process proteins in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
        # Submit all jobs
        future_to_protein = {executor.submit(process_single_protein_af2rank, item): item[0] for item in work_items}
        
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
                    summary = result.get('summary', {})
                    successful_scores = summary.get('successful_scores', 0)
                    total_structures = summary.get('total_structures', 0)
                    logger.info(f"✅ Progress: {completed}/{total} proteins completed ({successful_scores}/{total_structures} structures scored, {elapsed:.1f}s elapsed)")
                else:
                    logger.error(f"❌ {protein_name} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {protein_name} failed with exception: {e}")
                results.append({'protein': protein_name, 'status': 'failed', 'error': str(e)})
    
    # Summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'failed'])
    
    # Calculate total structures scored
    total_structures_scored = 0
    for result in results:
        if result['status'] == 'success':
            summary = result.get('summary', {})
            total_structures_scored += summary.get('successful_scores', 0)
    
    logger.info(f"🎯 AF2Rank scoring completed in {total_time:.1f}s")
    logger.info(f"✅ Successful proteins: {successful}")
    logger.info(f"❌ Failed proteins: {failed}")
    logger.info(f"📊 Total structures scored: {total_structures_scored}")
    
    if failed > 0:
        logger.info("Failed proteins:")
        for result in results:
            if result['status'] == 'failed':
                logger.info(f"  - {result['protein']}: {result.get('error', 'Unknown error')}")
    
    # Save summary report
    summary_report = {
        'total_proteins': len(protein_names),
        'successful_proteins': successful,
        'failed_proteins': failed,
        'total_structures_scored': total_structures_scored,
        'execution_time_seconds': total_time,
        'inference_config': args.inference_config,
        'recycles': args.recycles,
        'results': results
    }
    
    summary_file = f"/home/jupyter-chenxi/proteina/af2rank_evaluation/af2rank_summary_{args.inference_config}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    logger.info(f"📄 Summary report saved to: {summary_file}")
    
    # Return appropriate exit code
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
