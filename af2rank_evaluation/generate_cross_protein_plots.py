#!/usr/bin/env python3
"""
Cross-Protein AF2Rank Analysis

This script generates summary plots across all protein chains from AF2Rank analysis results.
Each protein chain is represented as a single point showing:
1. Highest template-to-ground-truth TM score vs. Spearman correlation (rho)
2. Template-to-ground-truth TM score of highest composite sample vs. Spearman correlation (rho)

Usage:
    python generate_cross_protein_plots.py --inference_dir <path> --output_dir <path>
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


def find_af2rank_summaries(inference_base_dir: str) -> List[str]:
    """
    Find all AF2Rank summary JSON files in the inference directory structure.
    
    Args:
        inference_base_dir: Base inference directory
        
    Returns:
        List of paths to AF2Rank summary JSON files
    """
    pattern = os.path.join(inference_base_dir, "*", "af2rank_analysis", "af2rank_summary_*.json")
    summary_files = glob.glob(pattern)
    logger.info(f"Found {len(summary_files)} AF2Rank summary files")
    return summary_files


def load_summary_data(summary_files: List[str], dataset_file: str) -> pd.DataFrame:
    """
    Load and compile summary data from all AF2Rank JSON files.
    
    Args:
        summary_files: List of paths to summary JSON files
        dataset_file: Name of the dataset file to analyze
    Returns:
        DataFrame with compiled summary data
    """
    data = []
    
    # Load dataset file
    dataset_df = pd.read_csv(dataset_file)
    
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Extract required metrics
            protein_id = summary.get('protein_id', 'unknown')
            spearman_rho_composite = summary.get('spearman_correlation_rho_composite') or summary.get('spearman_correlation_rho')  # Backward compat
            spearman_rho_ptm = summary.get('spearman_correlation_rho_ptm')
            max_tm_ref_template = summary.get('max_tm_ref_template')
            max_tm_ref_pred = summary.get('max_tm_ref_pred')
            tm_ref_template_at_max_composite = summary.get('tm_ref_template_at_max_composite')
            tm_ref_pred_at_max_composite = summary.get('tm_ref_pred_at_max_composite')
            tm_ref_pred_at_max_ptm = summary.get('tm_ref_pred_at_max_ptm')
            top_1_tm_ref_template = summary.get('top_1_tm_ref_template')
            top_5_tm_ref_template = summary.get('top_5_tm_ref_template')
            top_1_tm_ref_pred = summary.get('top_1_tm_ref_pred')
            top_5_tm_ref_pred = summary.get('top_5_tm_ref_pred')
            
            # Only include if we have the required metrics
            if (spearman_rho_composite is not None and 
                max_tm_ref_template is not None and 
                tm_ref_template_at_max_composite is not None):
                
                data.append({
                    'protein_id': protein_id,
                    'spearman_rho_composite': spearman_rho_composite,
                    'spearman_rho_ptm': spearman_rho_ptm,
                    'max_tm_ref_template': max_tm_ref_template,
                    'max_tm_ref_pred': max_tm_ref_pred,
                    'tm_ref_template_at_max_composite': tm_ref_template_at_max_composite,
                    'tm_ref_pred_at_max_composite': tm_ref_pred_at_max_composite,
                    'tm_ref_pred_at_max_ptm': tm_ref_pred_at_max_ptm,
                    'top_1_tm_ref_template': top_1_tm_ref_template,
                    'top_5_tm_ref_template': top_5_tm_ref_template,
                    'top_1_tm_ref_pred': top_1_tm_ref_pred,
                    'top_5_tm_ref_pred': top_5_tm_ref_pred,
                })
                
        except Exception as e:
            logger.warning(f"Failed to load {summary_file}: {e}")
            continue
    
    df = pd.DataFrame(data)
    df = df.merge(dataset_df[['natives_rcsb', 'tms_single', 'in_train']], left_on='protein_id', right_on='natives_rcsb', how='left')
    df.drop(columns=['natives_rcsb'], inplace=True)
    logger.info(f"Loaded data for {len(df)} proteins with complete metrics")
    return df


def create_summary_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create cross-protein summary plots.
    
    Args:
        df: DataFrame with compiled summary data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create first figure with four subplots (2x2 grid) for correlation analysis
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 18))
    
    # Plot 1: Max TM-score vs Spearman correlation (composite vs template quality)
    valid_data_1 = df.dropna(subset=['max_tm_ref_template', 'spearman_rho_composite'])
    
    if len(valid_data_1) > 0:
        scatter1 = ax1.scatter(
            valid_data_1['max_tm_ref_template'], 
            valid_data_1['spearman_rho_composite'],
            alpha=0.7, 
            s=60
        )
        
        ax1.set_xlabel('Highest Template-to-Ground-Truth TM Score', fontsize=12)
        ax1.set_ylabel('Spearman ρ (Composite vs Template Quality)', fontsize=12)
        ax1.set_title('Max TM Score vs. Template-Composite Correlation\n(How well does AF2 rank template quality?)', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Add correlation line if there's enough data
        if len(valid_data_1) > 2:
            z = np.polyfit(valid_data_1['max_tm_ref_template'], valid_data_1['spearman_rho_composite'], 1)
            p = np.poly1d(z)
            ax1.plot(valid_data_1['max_tm_ref_template'], p(valid_data_1['max_tm_ref_template']), 
                    "r--", alpha=0.8, linewidth=1)
            
            # Calculate and display correlation
            corr = np.corrcoef(valid_data_1['max_tm_ref_template'], valid_data_1['spearman_rho_composite'])[0,1]
            ax1.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: Template quality at max composite vs Spearman correlation (composite)
    valid_data_2 = df.dropna(subset=['tm_ref_template_at_max_composite', 'spearman_rho_composite'])
    
    if len(valid_data_2) > 0:
        scatter2 = ax2.scatter(
            valid_data_2['tm_ref_template_at_max_composite'], 
            valid_data_2['spearman_rho_composite'],
            alpha=0.7, 
            s=60
        )
        
        ax2.set_xlabel('Template TM Score at Highest Composite', fontsize=12)
        ax2.set_ylabel('Spearman ρ (Composite vs Template Quality)', fontsize=12)
        ax2.set_title('Template Quality of Best AF2 Score vs. Correlation\n(Does AF2\'s best match real best?)', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        # Add correlation line if there's enough data
        if len(valid_data_2) > 2:
            z = np.polyfit(valid_data_2['tm_ref_template_at_max_composite'], valid_data_2['spearman_rho_composite'], 1)
            p = np.poly1d(z)
            ax2.plot(valid_data_2['tm_ref_template_at_max_composite'], p(valid_data_2['tm_ref_template_at_max_composite']), 
                    "r--", alpha=0.8, linewidth=1)
            
            # Calculate and display correlation
            corr = np.corrcoef(valid_data_2['tm_ref_template_at_max_composite'], valid_data_2['spearman_rho_composite'])[0,1]
            ax2.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax2.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Prediction quality at max pTM vs Spearman correlation (pTM)
    valid_data_3 = df.dropna(subset=['tm_ref_pred_at_max_ptm', 'spearman_rho_ptm'])
    
    if len(valid_data_3) > 0:
        scatter3 = ax3.scatter(
            valid_data_3['tm_ref_pred_at_max_ptm'], 
            valid_data_3['spearman_rho_ptm'],
            alpha=0.7, 
            s=60
        )
        
        ax3.set_xlabel('Prediction TM Score at Highest pTM', fontsize=12)
        ax3.set_ylabel('Spearman ρ (pTM vs Prediction Quality)', fontsize=12)
        ax3.set_title('Prediction Quality of Best pTM vs. Correlation\n(Does pTM correlate with prediction quality?)', fontsize=13)
        ax3.grid(True, alpha=0.3)
        
        # Add correlation line if there's enough data
        if len(valid_data_3) > 2:
            z = np.polyfit(valid_data_3['tm_ref_pred_at_max_ptm'], valid_data_3['spearman_rho_ptm'], 1)
            p = np.poly1d(z)
            ax3.plot(valid_data_3['tm_ref_pred_at_max_ptm'], p(valid_data_3['tm_ref_pred_at_max_ptm']), 
                    "r--", alpha=0.8, linewidth=1)
            
            # Calculate and display correlation
            corr = np.corrcoef(valid_data_3['tm_ref_pred_at_max_ptm'], valid_data_3['spearman_rho_ptm'])[0,1]
            ax3.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax3.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Comparison of two correlation types
    valid_data_4 = df.dropna(subset=['spearman_rho_composite', 'spearman_rho_ptm'])
    
    if len(valid_data_4) > 0:
        scatter4 = ax4.scatter(
            valid_data_4['spearman_rho_composite'], 
            valid_data_4['spearman_rho_ptm'],
            alpha=0.7, 
            s=60
        )
        
        ax4.set_xlabel('Spearman ρ (Composite vs Template Quality)', fontsize=12)
        ax4.set_ylabel('Spearman ρ (pTM vs Prediction Quality)', fontsize=12)
        ax4.set_title('Comparison of AF2Rank Correlations\n(Template ranking vs Prediction quality)', fontsize=13)
        ax4.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        ax4.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Add correlation if there's enough data
        if len(valid_data_4) > 2:
            corr = np.corrcoef(valid_data_4['spearman_rho_composite'], valid_data_4['spearman_rho_ptm'])[0,1]
            ax4.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax4.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the first plot
    plot_path_1 = os.path.join(output_dir, "cross_protein_af2rank_correlation_analysis.png")
    plt.savefig(plot_path_1, dpi=900, bbox_inches='tight')
    logger.info(f"Saved cross-protein correlation plot: {plot_path_1}")
    
    plt.close()
    
    # Create second figure with eight subplots (2x4 grid) for top analysis
    fig2, ((ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(2, 4, figsize=(32, 18))

    # Plot 5: Max TM-score vs Top 1 Template TM-score By Composite
    valid_data_5 = df.dropna(subset=['max_tm_ref_template','top_1_tm_ref_template'])
    
    if len(valid_data_5) > 0:
        scatter5 = ax5.scatter(
            valid_data_5['max_tm_ref_template'], 
            valid_data_5['top_1_tm_ref_template'],
            alpha=0.7,
            c=valid_data_5['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line5 = ax5.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax5.set_xlabel('Max Template TM Score', fontsize=12)
        ax5.set_ylabel('Top 1 Template TM Score By Composite', fontsize=12)
        ax5.set_title('Max Template TM Score vs. Top 1 Template TM Score By Composite', fontsize=13)
        ax5.grid(True, alpha=0.3)
        ax5.legend(['In Train', 'Not In Train'], fontsize=24, loc='upper left')

    # Plot 6: Max TM-score vs Top 5 Template TM-score By Composite
    valid_data_6 = df.dropna(subset=['max_tm_ref_template','top_5_tm_ref_template'])
    
    if len(valid_data_6) > 0:
        scatter6 = ax6.scatter(
            valid_data_6['max_tm_ref_template'], 
            valid_data_6['top_5_tm_ref_template'],
            alpha=0.7,
            c=valid_data_6['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line6 = ax6.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax6.set_xlabel('Max Template TM Score', fontsize=12)
        ax6.set_ylabel('Top 5 Template TM Score By Composite', fontsize=12)
        ax6.set_title('Max Template TM Score vs. Top 5 Template TM Score By Composite', fontsize=13)
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Max TM-score vs Top 1 Template TM-score By pTM
    valid_data_7 = df.dropna(subset=['max_tm_ref_pred','top_1_tm_ref_pred'])
    
    if len(valid_data_7) > 0:
        scatter7 = ax7.scatter(
            valid_data_7['max_tm_ref_pred'], 
            valid_data_7['top_1_tm_ref_pred'],
            alpha=0.7,
            c=valid_data_7['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line7 = ax7.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax7.set_xlabel('Max Prediction TM Score', fontsize=12)
        ax7.set_ylabel('Top 1 Prediction TM Score By pTM', fontsize=12)
        ax7.set_title('Max Prediction TM Score vs. Top 1 Prediction TM Score By pTM', fontsize=13)
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: Max TM-score vs Top 5 Template TM-score By pTM
    valid_data_8 = df.dropna(subset=['max_tm_ref_pred','top_5_tm_ref_pred'])
    
    if len(valid_data_8) > 0:
        scatter8 = ax8.scatter(
            valid_data_8['max_tm_ref_pred'], 
            valid_data_8['top_5_tm_ref_pred'],
            alpha=0.7,
            c=valid_data_8['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line8 = ax8.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax8.set_xlabel('Max Prediction TM Score', fontsize=12)
        ax8.set_ylabel('Top 5 Prediction TM Score By pTM', fontsize=12)
        ax8.set_title('Max Prediction TM Score vs. Top 5 Prediction TM Score By pTM', fontsize=13)
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: Single-sequence TM score vs. Top 1 Template TM-score By Composite
    valid_data_9 = df.dropna(subset=['tms_single', 'top_1_tm_ref_template'])
    
    if len(valid_data_9) > 0:
        scatter9 = ax9.scatter(
            valid_data_9['tms_single'], 
            valid_data_9['top_1_tm_ref_template'],
            alpha=0.7,
            c=valid_data_9['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line9 = ax9.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax9.set_xlabel('Single-sequence TM Score', fontsize=12)
        ax9.set_ylabel('Top 1 Template TM Score By Composite', fontsize=12)
        ax9.set_title('Single-sequence TM Score vs. Top 1 Template TM Score By Composite', fontsize=13)
        ax9.grid(True, alpha=0.3)
    
    # Plot 10: Single-sequence TM score vs. Top 5 Template TM-score By Composite
    valid_data_10 = df.dropna(subset=['tms_single', 'top_5_tm_ref_template'])
    
    if len(valid_data_10) > 0:
        scatter10 = ax10.scatter(
            valid_data_10['tms_single'], 
            valid_data_10['top_5_tm_ref_template'],
            alpha=0.7,
            c=valid_data_10['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line10 = ax10.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax10.set_xlabel('Single-sequence TM Score', fontsize=12)
        ax10.set_ylabel('Top 5 Template TM Score By Composite', fontsize=12)
        ax10.set_title('Single-sequence TM Score vs. Top 5 Template TM Score By Composite', fontsize=13)
        ax10.grid(True, alpha=0.3)
    
    # Plot 11: Single-sequence TM score vs. Top 1 Template TM-score By pTM
    valid_data_11 = df.dropna(subset=['tms_single', 'top_1_tm_ref_pred'])
    
    if len(valid_data_11) > 0:
        scatter11 = ax11.scatter(
            valid_data_11['tms_single'], 
            valid_data_11['top_1_tm_ref_pred'],
            alpha=0.7,
            c=valid_data_11['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line11 = ax11.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax11.set_xlabel('Single-sequence TM Score', fontsize=12)
        ax11.set_ylabel('Top 1 Prediction TM Score By pTM', fontsize=12)
        ax11.set_title('Single-sequence TM Score vs. Top 1 Prediction TM Score By pTM', fontsize=13)
        ax11.grid(True, alpha=0.3)
    
    # Plot 12: Single-sequence TM score vs. Top 5 Template TM-score By pTM
    valid_data_12 = df.dropna(subset=['tms_single', 'top_5_tm_ref_pred'])
    
    if len(valid_data_12) > 0:
        scatter12 = ax12.scatter(
            valid_data_12['tms_single'], 
            valid_data_12['top_5_tm_ref_pred'],
            alpha=0.7,
            c=valid_data_12['in_train'].map({True: 'blue', False: 'red'}),
            s=60
        )
        line12 = ax12.plot([0, 1], [0, 1], 'r--', alpha=0.75, linewidth=3)
        ax12.set_xlabel('Single-sequence TM Score', fontsize=12)
        ax12.set_ylabel('Top 5 Prediction TM Score By pTM', fontsize=12)
        ax12.set_title('Single-sequence TM Score vs. Top 5 Prediction TM Score By pTM', fontsize=13)
        ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the second plot
    plot_path_2 = os.path.join(output_dir, "cross_protein_af2rank_top_analysis.png")
    plt.savefig(plot_path_2, dpi=900, bbox_inches='tight')
    logger.info(f"Saved cross-protein top analysis plot: {plot_path_2}")
    
    plt.close()


def save_summary_statistics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Save summary statistics to CSV and JSON files.
    
    Args:
        df: DataFrame with compiled summary data
        output_dir: Directory to save files
    """
    # Save the full dataset
    csv_path = os.path.join(output_dir, "cross_protein_summary_data.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary data: {csv_path}")
    
    # Calculate and save summary statistics
    stats = {
        "total_proteins": len(df),
        "statistics": {
            "spearman_correlation_rho_composite": {
                "mean": float(df['spearman_rho_composite'].mean()),
                "median": float(df['spearman_rho_composite'].median()),
                "std": float(df['spearman_rho_composite'].std()),
                "min": float(df['spearman_rho_composite'].min()),
                "max": float(df['spearman_rho_composite'].max())
            },
            "spearman_correlation_rho_ptm": {
                "mean": float(df['spearman_rho_ptm'].mean()),
                "median": float(df['spearman_rho_ptm'].median()),
                "std": float(df['spearman_rho_ptm'].std()),
                "min": float(df['spearman_rho_ptm'].min()),
                "max": float(df['spearman_rho_ptm'].max())
            },
            "max_tm_ref_template": {
                "mean": float(df['max_tm_ref_template'].mean()),
                "median": float(df['max_tm_ref_template'].median()),
                "std": float(df['max_tm_ref_template'].std()),
                "min": float(df['max_tm_ref_template'].min()),
                "max": float(df['max_tm_ref_template'].max())
            },
            "tm_ref_template_at_max_composite": {
                "mean": float(df['tm_ref_template_at_max_composite'].mean()),
                "median": float(df['tm_ref_template_at_max_composite'].median()),
                "std": float(df['tm_ref_template_at_max_composite'].std()),
                "min": float(df['tm_ref_template_at_max_composite'].min()),
                "max": float(df['tm_ref_template_at_max_composite'].max())
            },
            "tm_ref_pred_at_max_ptm": {
                "mean": float(df['tm_ref_pred_at_max_ptm'].mean()),
                "median": float(df['tm_ref_pred_at_max_ptm'].median()),
                "std": float(df['tm_ref_pred_at_max_ptm'].std()),
                "min": float(df['tm_ref_pred_at_max_ptm'].min()),
                "max": float(df['tm_ref_pred_at_max_ptm'].max())
            }
        },
        "correlations": {
            "max_tm_vs_spearman_rho_composite": float(np.corrcoef(df['max_tm_ref_template'], df['spearman_rho_composite'])[0,1]),
            "tm_at_max_composite_vs_spearman_rho_composite": float(np.corrcoef(df['tm_ref_template_at_max_composite'], df['spearman_rho_composite'])[0,1]),
            "tm_at_max_ptm_vs_spearman_rho_ptm": float(np.corrcoef(df['tm_ref_pred_at_max_ptm'].dropna(), df['spearman_rho_ptm'].dropna())[0,1]),
            "spearman_rho_composite_vs_spearman_rho_ptm": float(np.corrcoef(df['spearman_rho_composite'].dropna(), df['spearman_rho_ptm'].dropna())[0,1])
        }
    }
    
    stats_path = os.path.join(output_dir, "cross_protein_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved summary statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate cross-protein AF2Rank analysis plots')
    parser.add_argument('--inference_dir', required=True, 
                       help='Base inference directory containing protein chain results')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for plots and summary files')
    parser.add_argument('--dataset_file', required=True,
                       help='Name of the dataset file to analyze')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    logger.info("🔬 Starting cross-protein AF2Rank analysis")
    logger.info(f"📂 Inference directory: {args.inference_dir}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"📄 Dataset file: {args.dataset_file}")

    # Validate input directory
    if not os.path.exists(args.inference_dir):
        logger.error(f"Inference directory not found: {args.inference_dir}")
        sys.exit(1)
    
    # Validate dataset file
    if not os.path.exists(args.dataset_file):
        logger.error(f"Dataset file not found: {args.dataset_file}")
        sys.exit(1)
    
    # Find all AF2Rank summary files
    summary_files = find_af2rank_summaries(args.inference_dir)
    
    if not summary_files:
        logger.error("No AF2Rank summary files found")
        sys.exit(1)
    
    # Load and compile data
    df = load_summary_data(summary_files, args.dataset_file)
    
    if len(df) == 0:
        logger.error("No valid data found in summary files")
        sys.exit(1)
    
    logger.info(f"📊 Analyzing {len(df)} proteins")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    create_summary_plots(df, args.output_dir)
    
    # Save summary statistics
    save_summary_statistics(df, args.output_dir)
    
    logger.info("✅ Cross-protein analysis completed successfully!")
    logger.info(f"📈 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
