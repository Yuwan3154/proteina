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


def load_summary_data(summary_files: List[str]) -> pd.DataFrame:
    """
    Load and compile summary data from all AF2Rank JSON files.
    
    Args:
        summary_files: List of paths to summary JSON files
        
    Returns:
        DataFrame with compiled summary data
    """
    data = []
    
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Extract required metrics
            protein_id = summary.get('protein_id', 'unknown')
            spearman_rho_composite = summary.get('spearman_correlation_rho_composite') or summary.get('spearman_correlation_rho')  # Backward compat
            spearman_rho_ptm = summary.get('spearman_correlation_rho_ptm')
            max_tm_ref_template = summary.get('max_tm_ref_template')
            tm_ref_template_at_max_composite = summary.get('tm_ref_template_at_max_composite')
            tm_ref_pred_at_max_composite = summary.get('tm_ref_pred_at_max_composite')
            tm_ref_pred_at_max_ptm = summary.get('tm_ref_pred_at_max_ptm')
            successful_scores = summary.get('successful_scores', 0)
            total_structures = summary.get('total_structures', 0)
            
            # Only include if we have the required metrics
            if (spearman_rho_composite is not None and 
                max_tm_ref_template is not None and 
                tm_ref_template_at_max_composite is not None):
                
                data.append({
                    'protein_id': protein_id,
                    'spearman_rho_composite': spearman_rho_composite,
                    'spearman_rho_ptm': spearman_rho_ptm,
                    'max_tm_ref_template': max_tm_ref_template,
                    'tm_ref_template_at_max_composite': tm_ref_template_at_max_composite,
                    'tm_ref_pred_at_max_composite': tm_ref_pred_at_max_composite,
                    'tm_ref_pred_at_max_ptm': tm_ref_pred_at_max_ptm,
                    'successful_scores': successful_scores,
                    'total_structures': total_structures,
                    'success_rate': successful_scores / total_structures if total_structures > 0 else 0
                })
                
        except Exception as e:
            logger.warning(f"Failed to load {summary_file}: {e}")
            continue
    
    df = pd.DataFrame(data)
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
    
    # Create figure with four subplots (2x2 grid)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Max TM-score vs Spearman correlation (composite vs template quality)
    valid_data_1 = df.dropna(subset=['max_tm_ref_template', 'spearman_rho_composite'])
    
    if len(valid_data_1) > 0:
        scatter1 = ax1.scatter(
            valid_data_1['max_tm_ref_template'], 
            valid_data_1['spearman_rho_composite'],
            c=valid_data_1['successful_scores'], 
            cmap='viridis', 
            alpha=0.7, 
            s=60
        )
        
        ax1.set_xlabel('Highest Template-to-Ground-Truth TM Score', fontsize=12)
        ax1.set_ylabel('Spearman ρ (Composite vs Template Quality)', fontsize=12)
        ax1.set_title('Max TM Score vs. Template-Composite Correlation\n(How well does AF2 rank template quality?)', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for successful scores
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Successful Scores', fontsize=10)
        
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
            c=valid_data_2['successful_scores'], 
            cmap='viridis', 
            alpha=0.7, 
            s=60
        )
        
        ax2.set_xlabel('Template TM Score at Highest Composite', fontsize=12)
        ax2.set_ylabel('Spearman ρ (Composite vs Template Quality)', fontsize=12)
        ax2.set_title('Template Quality of Best AF2 Score vs. Correlation\n(Does AF2\'s best match real best?)', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for successful scores
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Successful Scores', fontsize=10)
        
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
            c=valid_data_3['successful_scores'], 
            cmap='plasma', 
            alpha=0.7, 
            s=60
        )
        
        ax3.set_xlabel('Prediction TM Score at Highest pTM', fontsize=12)
        ax3.set_ylabel('Spearman ρ (pTM vs Prediction Quality)', fontsize=12)
        ax3.set_title('Prediction Quality of Best pTM vs. Correlation\n(Does pTM correlate with prediction quality?)', fontsize=13)
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for successful scores
        cbar3 = plt.colorbar(scatter3, ax=ax3)
        cbar3.set_label('Successful Scores', fontsize=10)
        
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
            c=valid_data_4['successful_scores'], 
            cmap='coolwarm', 
            alpha=0.7, 
            s=60
        )
        
        ax4.set_xlabel('Spearman ρ (Composite vs Template Quality)', fontsize=12)
        ax4.set_ylabel('Spearman ρ (pTM vs Prediction Quality)', fontsize=12)
        ax4.set_title('Comparison of AF2Rank Correlations\n(Template ranking vs Prediction quality)', fontsize=13)
        ax4.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        ax4.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Add colorbar for successful scores
        cbar4 = plt.colorbar(scatter4, ax=ax4)
        cbar4.set_label('Successful Scores', fontsize=10)
        
        # Add correlation if there's enough data
        if len(valid_data_4) > 2:
            corr = np.corrcoef(valid_data_4['spearman_rho_composite'], valid_data_4['spearman_rho_ptm'])[0,1]
            ax4.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax4.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "cross_protein_af2rank_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cross-protein plot: {plot_path}")
    
    plt.close()
    
    # Create individual detailed plots
    create_detailed_plots(df, output_dir)


def create_detailed_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create additional detailed analysis plots.
    
    Args:
        df: DataFrame with compiled summary data
        output_dir: Directory to save plots
    """
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Spearman correlation distribution (composite)
    df['spearman_rho_composite'].hist(bins=20, ax=axes[0, 0], alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Spearman ρ (Composite vs Template)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Template-Composite Correlations')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spearman correlation distribution (pTM)
    df['spearman_rho_ptm'].dropna().hist(bins=20, ax=axes[0, 1], alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Spearman ρ (pTM vs Prediction)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of pTM-Prediction Correlations')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Max TM score distribution
    df['max_tm_ref_template'].hist(bins=20, ax=axes[0, 2], alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_xlabel('Maximum Template-to-Ground-Truth TM Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Maximum TM Scores')
    axes[0, 2].grid(True, alpha=0.3)
    
    # TM score at max composite distribution
    df['tm_ref_template_at_max_composite'].hist(bins=20, ax=axes[1, 0], alpha=0.7, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('TM Score at Highest Composite')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Template TM at Max Composite Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # TM score at max pTM distribution
    df['tm_ref_pred_at_max_ptm'].dropna().hist(bins=20, ax=axes[1, 1], alpha=0.7, color='magenta', edgecolor='black')
    axes[1, 1].set_xlabel('Prediction TM Score at Highest pTM')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction TM at Max pTM')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Success rate vs correlation (composite)
    axes[1, 2].scatter(df['success_rate'], df['spearman_rho_composite'], alpha=0.7, color='purple')
    axes[1, 2].set_xlabel('Success Rate')
    axes[1, 2].set_ylabel('Spearman ρ (Composite vs Template)')
    axes[1, 2].set_title('Success Rate vs. Correlation')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the detailed plots
    detailed_plot_path = os.path.join(output_dir, "cross_protein_detailed_analysis.png")
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved detailed analysis plot: {detailed_plot_path}")
    
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
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    logger.info("🔬 Starting cross-protein AF2Rank analysis")
    logger.info(f"📂 Inference directory: {args.inference_dir}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    
    # Validate input directory
    if not os.path.exists(args.inference_dir):
        logger.error(f"Inference directory not found: {args.inference_dir}")
        sys.exit(1)
    
    # Find all AF2Rank summary files
    summary_files = find_af2rank_summaries(args.inference_dir)
    
    if not summary_files:
        logger.error("No AF2Rank summary files found")
        sys.exit(1)
    
    # Load and compile data
    df = load_summary_data(summary_files)
    
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
