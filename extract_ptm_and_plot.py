#!/usr/bin/env python3
"""
Script to extract PTM values from AlphaFold3 scoring results and create a TM vs PTM correlation plot.

This script:
1. Reads the sampled_files_aln.tsv file to get TM scores
2. Extracts PTM values from summary_confidences.json files in score_results subdirectories
3. Creates a high-DPI correlation plot (tm_vs_ptm.png)

Usage:
    python extract_ptm_and_plot.py --input_dir <path_to_directory>
    
Example:
    python extract_ptm_and_plot.py --input_dir ~/proteina/inference/inference_seq_cond_sampling_finetune-all_8-seq_purge-7bny-7kww-7ad5_045-noise/7ad5_A/
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from pathlib import Path
from typing import Dict, List, Tuple

def extract_ptm_from_json(json_path: Path) -> float:
    """Extract PTM value from summary_confidences.json file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('ptm', None)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not extract PTM from {json_path}: {e}")
        return None

def get_tm_scores(aln_file: Path) -> Dict[str, float]:
    """Extract TM scores from alignment TSV file."""
    try:
        aln_df = pd.read_csv(aln_file, sep='\t')
        tm_scores = {}
        
        for _, row in aln_df.iterrows():
            # Extract filename from PDBchain2 column
            pdb_path = row['PDBchain2']
            # Get just the filename without path and chain identifier
            filename = Path(pdb_path).name.split(':')[0]
            # Remove .pdb extension and add _cg2all suffix if not present
            base_name = filename.replace('.pdb', '')
            if not base_name.endswith('_cg2all'):
                base_name += '_cg2all'
            
            tm_scores[base_name] = row['TM1']
        
        return tm_scores
    except Exception as e:
        print(f"Error reading alignment file {aln_file}: {e}")
        return {}

def find_ptm_files(score_results_dir: Path) -> List[Tuple[str, Path]]:
    """Find all summary_confidences.json files in score_results directory."""
    ptm_files = []
    
    if not score_results_dir.exists():
        print(f"Warning: Score results directory {score_results_dir} does not exist")
        return ptm_files
    
    for subdir in score_results_dir.iterdir():
        if subdir.is_dir():
            # Look for summary_confidences.json files
            summary_files = list(subdir.glob("*summary_confidences.json"))
            if summary_files:
                # Use the first one found (should be only one)
                ptm_files.append((subdir.name, summary_files[0]))
            else:
                # Also check in subdirectories (like seed-X_sample-Y)
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir():
                        summary_files = list(subsubdir.glob("summary_confidences.json"))
                        if summary_files:
                            ptm_files.append((subdir.name, summary_files[0]))
                            break
    
    return ptm_files

def create_correlation_plot(tm_scores: List[float], ptm_scores: List[float], 
                          output_path: Path, title: str = "TM vs PTM Correlation") -> None:
    """Create a high-DPI correlation plot."""
    
    # Calculate correlation statistics
    spearman_rho, spearman_p = scipy.stats.spearmanr(tm_scores, ptm_scores)
    pearson_r, pearson_p = scipy.stats.pearsonr(tm_scores, ptm_scores)
    
    # Linear regression
    slope, intercept = np.polyfit(tm_scores, ptm_scores, 1)
    r_squared = pearson_r ** 2
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    plt.scatter(tm_scores, ptm_scores, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    x_line = np.linspace(min(tm_scores), max(tm_scores), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r--', linewidth=2, 
            #  label=f'Linear fit: y = {slope:.3f}x + {intercept:.3f}'
             )
    
    # Labels and title
    plt.xlabel('TM Score', fontsize=24, fontweight='bold')
    plt.ylabel('PTM Score', fontsize=24, fontweight='bold')
    plt.title(title, fontsize=24, fontweight='bold')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0.15, 0.8)
    plt.ylim(0.15, 0.8)
    
    # Add correlation statistics to the plot
    stats_text = f'Spearman ρ = {spearman_rho:.3f} (p = {spearman_p:.2e})\n'
    stats_text += f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})\n'
    stats_text += f'R² = {r_squared:.3f}\n'
    stats_text += f'N = {len(tm_scores)} samples'
    
    # Place the stats text in the bottom right corner
    plt.text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, 
             fontsize=24, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # plt.grid(True, alpha=0.3)
    
    # Set equal aspect ratio if scores are in similar range
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save with high DPI
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    print(f"Correlation statistics:")
    print(f"  Spearman ρ: {spearman_rho:.3f} (p-value: {spearman_p:.2e})")
    print(f"  Pearson r: {pearson_r:.3f} (p-value: {pearson_p:.2e})")
    print(f"  R²: {r_squared:.3f}")
    print(f"  Number of samples: {len(tm_scores)}")

def main():
    parser = argparse.ArgumentParser(description='Extract PTM values and create TM vs PTM correlation plot')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Input directory containing sampled_files_aln.tsv and score_results/')
    parser.add_argument('--title', '-t', type=str, default='TM vs PTM Correlation',
                       help='Title for the plot')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).expanduser().resolve()
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    print(f"Processing directory: {input_dir}")
    
    # Read TM scores from alignment file
    aln_file = input_dir / "sampled_files_aln.tsv"
    if not aln_file.exists():
        print(f"Error: Alignment file {aln_file} not found")
        return 1
    
    print("Reading TM scores from alignment file...")
    tm_scores_dict = get_tm_scores(aln_file)
    print(f"Found TM scores for {len(tm_scores_dict)} structures")
    
    # Find PTM files in score_results directory
    score_results_dir = input_dir / "score_results"
    print("Finding PTM files in score_results directory...")
    ptm_files = find_ptm_files(score_results_dir)
    print(f"Found {len(ptm_files)} PTM files")
    
    # Extract PTM values and match with TM scores
    matched_data = []
    missing_tm = []
    missing_ptm = []
    
    for structure_name, ptm_file_path in ptm_files:
        ptm_value = extract_ptm_from_json(ptm_file_path)
        
        if ptm_value is not None:
            if structure_name in tm_scores_dict:
                tm_value = tm_scores_dict[structure_name]
                matched_data.append((structure_name, tm_value, ptm_value))
            else:
                missing_tm.append(structure_name)
        else:
            missing_ptm.append(structure_name)
    
    print(f"\nMatching results:")
    print(f"  Successfully matched: {len(matched_data)} structures")
    print(f"  Missing TM scores: {len(missing_tm)} structures")
    print(f"  Missing PTM scores: {len(missing_ptm)} structures")
    
    if missing_tm:
        print(f"  Structures missing TM scores: {missing_tm[:5]}{'...' if len(missing_tm) > 5 else ''}")
    if missing_ptm:
        print(f"  Structures missing PTM scores: {missing_ptm[:5]}{'...' if len(missing_ptm) > 5 else ''}")
    
    if len(matched_data) < 3:
        print("Error: Need at least 3 matched data points for correlation analysis")
        return 1
    
    # Prepare data for plotting
    structure_names, tm_scores, ptm_scores = zip(*matched_data)
    tm_scores = list(tm_scores)
    ptm_scores = list(ptm_scores)
    
    # Create output plot
    output_path = score_results_dir / "tm_vs_ptm.png"
    print(f"\nCreating correlation plot...")
    create_correlation_plot(tm_scores, ptm_scores, output_path, args.title)
    
    # Save data to CSV for reference
    data_df = pd.DataFrame({
        'structure_name': structure_names,
        'tm_score': tm_scores,
        'ptm_score': ptm_scores
    })
    csv_output = score_results_dir / "tm_vs_ptm_data.csv"
    data_df.to_csv(csv_output, index=False)
    print(f"Data saved to: {csv_output}")
    
    return 0

if __name__ == "__main__":
    exit(main())