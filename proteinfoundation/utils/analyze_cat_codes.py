# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import torch
import pandas as pd
import hydra
from loguru import logger
from omegaconf import OmegaConf

# Add proteina to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.utils.cirpin_utils import CirpinEmbeddingLoader


def extract_cat_code_from_cath(cath_code: str) -> str:
    """Extract CAT code (first 3 parts) from CATH code."""
    # Use the centralized method from CirpinEmbeddingLoader
    result = CirpinEmbeddingLoader._extract_cat_code_from_cath(cath_code)
    return result if result is not None else "invalid"


def analyze_dataset_cat_codes(dataset_config_path: str, 
                              cirpin_emb_path: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Analyze CAT codes in the training dataset.
    
    Args:
        dataset_config_path (str): Path to dataset config
        cirpin_emb_path (str): Path to CIRPIN embeddings file
        
    Returns:
        Tuple of (cat_code_counts, cat_code_to_proteins) mappings
    """
    logger.info("Analyzing CAT codes in training dataset...")
    
    # Load dataset config
    with hydra.initialize(config_path=os.path.dirname(dataset_config_path), version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=os.path.basename(dataset_config_path).replace('.yaml', ''))
    
    # Create datamodule
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    # Get training dataset
    train_dataset = datamodule.train_dataset()
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    # Load CIRPIN embeddings to check availability
    cirpin_loader = CirpinEmbeddingLoader(cirpin_emb_path)
    available_protein_ids = set(cirpin_loader.get_all_protein_ids())
    
    # Collect CAT codes and protein IDs
    cat_code_counts = Counter()
    cat_code_to_proteins = defaultdict(list)
    protein_to_cat = {}
    proteins_with_cirpin = 0
    total_proteins = 0
    
    logger.info("Scanning training data for CAT codes...")
    
    for i, data in enumerate(train_dataset):
        if i % 1000 == 0:
            logger.info(f"Processed {i}/{len(train_dataset)} samples")
        
        # Extract protein ID
        protein_id = None
        if hasattr(data, 'id'):
            protein_id = data.id
        elif hasattr(data, 'protein_id'):
            protein_id = data.protein_id
        elif hasattr(data, 'pdb_id'):
            protein_id = data.pdb_id
        elif hasattr(data, 'name'):
            protein_id = data.name
        
        # Extract CATH code
        cath_code = None
        if hasattr(data, 'cath_code'):
            cath_code = data.cath_code
        elif hasattr(data, 'fold_label'):
            cath_code = data.fold_label
        
        if protein_id is not None and cath_code is not None:
            # Handle list format for CATH codes (most common case)
            if isinstance(cath_code, list):
                if len(cath_code) > 0:
                    cath_code = cath_code[0]  # Take first CATH code
                else:
                    cath_code = None  # Empty list means no CATH code
            
            # Convert to strings
            if protein_id is not None:
                if isinstance(protein_id, torch.Tensor):
                    protein_id = protein_id.item() if protein_id.numel() == 1 else str(protein_id)
                protein_id = str(protein_id)
            
            if cath_code is not None:
                if isinstance(cath_code, torch.Tensor):
                    cath_code = cath_code.item() if cath_code.numel() == 1 else str(cath_code)
                cath_code = str(cath_code)
            
            if cath_code is not None:
                # Clean protein ID
                clean_protein_id = protein_id.replace('.pt', '') if protein_id.endswith('.pt') else protein_id
                
                # Extract CAT code
                cat_code = extract_cat_code_from_cath(cath_code)
                
                if cat_code != "invalid":
                    total_proteins += 1
                    cat_code_counts[cat_code] += 1
                    cat_code_to_proteins[cat_code].append(clean_protein_id)
                    protein_to_cat[clean_protein_id] = cat_code
                    
                    # Check if CIRPIN embedding is available
                    if clean_protein_id in available_protein_ids:
                        proteins_with_cirpin += 1
    
    logger.info(f"Analysis complete!")
    logger.info(f"Total proteins with valid CAT codes: {total_proteins}")
    logger.info(f"Proteins with CIRPIN embeddings: {proteins_with_cirpin}")
    if total_proteins > 0:
        logger.info(f"CIRPIN coverage: {proteins_with_cirpin/total_proteins*100:.2f}%")
    else:
        logger.warning("No proteins with valid CAT codes found - CIRPIN sampling mode will not work")
    logger.info(f"Unique CAT codes found: {len(cat_code_counts)}")
    
    return dict(cat_code_counts), dict(cat_code_to_proteins), protein_to_cat


def print_cat_code_statistics(cat_code_counts: Dict[str, int], 
                              cat_code_to_proteins: Dict[str, List[str]],
                              cirpin_loader: CirpinEmbeddingLoader,
                              top_n: int = 20):
    """Print detailed CAT code statistics."""
    
    logger.info(f"\n{'='*60}")
    logger.info("CAT CODE STATISTICS FOR TRAINING DATASET")
    logger.info(f"{'='*60}")
    
    # Sort by count
    sorted_cats = sorted(cat_code_counts.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"\nTop {top_n} CAT codes by protein count:")
    logger.info(f"{'CAT Code':<15} {'Count':<8} {'CIRPIN Available':<15} {'Coverage %':<12}")
    logger.info("-" * 60)
    
    total_proteins = sum(cat_code_counts.values())
    total_with_cirpin = 0
    available_protein_ids = set(cirpin_loader.get_all_protein_ids())
    
    for cat_code, count in sorted_cats[:top_n]:
        proteins = cat_code_to_proteins[cat_code]
        proteins_with_cirpin = sum(1 for p in proteins if p in available_protein_ids)
        coverage = proteins_with_cirpin / count * 100 if count > 0 else 0
        total_with_cirpin += proteins_with_cirpin
        
        logger.info(f"{cat_code:<15} {count:<8} {proteins_with_cirpin:<15} {coverage:<12.1f}")
    
    if len(sorted_cats) > top_n:
        remaining_count = sum(count for _, count in sorted_cats[top_n:])
        logger.info(f"... and {len(sorted_cats) - top_n} more CAT codes with {remaining_count} proteins")
    
    overall_coverage = total_with_cirpin / total_proteins * 100 if total_proteins > 0 else 0
    logger.info(f"\nOverall CIRPIN coverage: {total_with_cirpin}/{total_proteins} ({overall_coverage:.2f}%)")
    
    # Distribution statistics
    counts = list(cat_code_counts.values())
    logger.info(f"\nCAT code distribution:")
    logger.info(f"  Mean proteins per CAT: {sum(counts)/len(counts):.2f}")
    logger.info(f"  Median proteins per CAT: {sorted(counts)[len(counts)//2]:.2f}")
    logger.info(f"  Max proteins per CAT: {max(counts)}")
    logger.info(f"  Min proteins per CAT: {min(counts)}")
    
    # CAT codes with good CIRPIN coverage for sampling
    good_coverage_cats = []
    for cat_code, count in sorted_cats:
        if count >= 5:  # At least 5 proteins
            proteins = cat_code_to_proteins[cat_code]
            proteins_with_cirpin = sum(1 for p in proteins if p in available_protein_ids)
            coverage = proteins_with_cirpin / count * 100
            if coverage >= 50:  # At least 50% CIRPIN coverage
                good_coverage_cats.append((cat_code, count, proteins_with_cirpin, coverage))
    
    logger.info(f"\nCAT codes suitable for sampling (≥5 proteins, ≥50% CIRPIN coverage): {len(good_coverage_cats)}")
    if good_coverage_cats:
        logger.info("Top candidates for sampling:")
        for cat_code, count, cirpin_count, coverage in good_coverage_cats[:10]:
            logger.info(f"  {cat_code}: {cirpin_count}/{count} proteins ({coverage:.1f}% coverage)")


def main():
    parser = argparse.ArgumentParser(description="Analyze CAT codes in training dataset")
    parser.add_argument("--dataset_config", type=str, 
                       default="../configs/datasets_config/pdb/pdb_train_S25_max320_purge-test.yaml",
                       help="Path to dataset config file")
    parser.add_argument("--cirpin_emb_path", type=str,
                       default="/home/jupyter-chenxi/progres/progres/databases/v_0_2_1/pdb_max384.pt",
                       help="Path to CIRPIN embeddings file")
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Optional: Save results to CSV file")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top CAT codes to display")
    
    args = parser.parse_args()
    
    # Analyze CAT codes
    cat_code_counts, cat_code_to_proteins, protein_to_cat = analyze_dataset_cat_codes(
        args.dataset_config, args.cirpin_emb_path
    )
    
    # Load CIRPIN loader for coverage analysis
    cirpin_loader = CirpinEmbeddingLoader(args.cirpin_emb_path)
    
    # Print statistics
    print_cat_code_statistics(cat_code_counts, cat_code_to_proteins, cirpin_loader, args.top_n)
    
    # Save to CSV if requested
    if args.output_csv:
        results = []
        available_protein_ids = set(cirpin_loader.get_all_protein_ids())
        
        for cat_code, count in cat_code_counts.items():
            proteins = cat_code_to_proteins[cat_code]
            proteins_with_cirpin = sum(1 for p in proteins if p in available_protein_ids)
            coverage = proteins_with_cirpin / count * 100 if count > 0 else 0
            
            results.append({
                'cat_code': cat_code,
                'protein_count': count,
                'cirpin_available': proteins_with_cirpin,
                'cirpin_coverage_percent': coverage,
                'protein_ids': ','.join(proteins[:10])  # First 10 protein IDs as sample
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('protein_count', ascending=False)
        df.to_csv(args.output_csv, index=False)
        logger.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
