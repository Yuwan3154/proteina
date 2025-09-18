#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Utility to convert CIF files to PT format for Proteina inference.
This script extracts sequence information from CIF files and creates 
the torch geometric Data objects needed for inference.

The amino acid encoding uses Proteina's exact constants from 
openfold.np.residue_constants to ensure 100% compatibility.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch_geometric.data import Data
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from loguru import logger
from dotenv import load_dotenv

# Add proteina to path if needed
sys.path.append('/home/jupyter-chenxi/proteina')

# Import Proteina's exact amino acid encoding to ensure consistency
from openfold.np.residue_constants import (
    restypes, 
    restype_order, 
    restype_3to1, 
    resname_to_idx,
    unk_restype_index
)

# Load environment variables
load_dotenv('/home/jupyter-chenxi/proteina/.env')


def extract_sequence_from_cif(cif_file: str, chain_id: str) -> Optional[List[str]]:
    """
    Extract amino acid sequence from a CIF file for a specific chain.
    
    Args:
        cif_file: Path to the CIF file
        chain_id: Chain identifier to extract
        
    Returns:
        List of 1-letter amino acid codes, or None if chain not found
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein', cif_file)
        
        # Collect all available chains for debugging
        available_chains = []
        
        for model in structure:
            for chain in model:
                available_chains.append(chain.id)
                if chain.id == chain_id:
                    sequence = []
                    for residue in chain:
                        if residue.id[0] == ' ':  # Standard residue (not heteroatom)
                            resname = residue.get_resname()
                            if resname in restype_3to1:
                                sequence.append(restype_3to1[resname])
                            else:
                                logger.warning(f"Unknown residue {resname} in {cif_file}, chain {chain_id}")
                                sequence.append('X')  # Unknown residue
                    return sequence
        
        # If exact chain not found, try common mappings
        # Many CIF files use different chain naming conventions
        chain_mappings = {
            'A': ['X', '1', 'a'],  # Common mappings for chain A
            'B': ['Y', '2', 'b'],  # Common mappings for chain B  
            'C': ['Z', '3', 'c'],  # Common mappings for chain C
        }
        
        if chain_id in chain_mappings:
            for alt_chain in chain_mappings[chain_id]:
                for model in structure:
                    for chain in model:
                        if chain.id == alt_chain:
                            logger.info(f"Found chain {chain_id} as {alt_chain} in {cif_file}")
                            sequence = []
                            for residue in chain:
                                if residue.id[0] == ' ':  # Standard residue (not heteroatom)
                                    resname = residue.get_resname()
                                    if resname in restype_3to1:
                                        sequence.append(restype_3to1[resname])
                                    else:
                                        logger.warning(f"Unknown residue {resname} in {cif_file}, chain {alt_chain}")
                                        sequence.append('X')  # Unknown residue
                            return sequence
        
        logger.error(f"Chain {chain_id} not found in {cif_file}. Available chains: {available_chains}")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing {cif_file}: {e}")
        return None


def sequence_to_pt_data(sequence: List[str], protein_id: str) -> Data:
    """
    Convert amino acid sequence to PyTorch Geometric Data object.
    
    Args:
        sequence: List of 1-letter amino acid codes
        protein_id: Identifier for the protein
        
    Returns:
        PyTorch Geometric Data object
    """
    # Convert amino acids to integer types using Proteina's encoding
    residue_type = []
    for aa in sequence:
        if aa in restype_order:
            residue_type.append(restype_order[aa])
        else:
            # Unknown residue - use Proteina's unknown residue index
            residue_type.append(unk_restype_index)
    
    residue_type_tensor = torch.tensor(residue_type, dtype=torch.long)
    
    # Create a minimal Data object with the required fields for inference
    data = Data()
    data.residue_type = residue_type_tensor
    data.id = protein_id
    
    # Add sequence position information
    seq_pos = torch.arange(len(sequence)).unsqueeze(1)
    data.seq_pos = seq_pos
    
    return data


def convert_cif_to_pt(cif_file: str, chain_id: str, output_file: str) -> bool:
    """
    Convert a CIF file to PT format for a specific chain.
    
    Args:
        cif_file: Path to input CIF file
        chain_id: Chain identifier to extract
        output_file: Path to output PT file
        
    Returns:
        True if successful, False otherwise
    """
    # Extract sequence from CIF
    sequence = extract_sequence_from_cif(cif_file, chain_id)
    if sequence is None:
        return False
    
    # Create protein ID from filename and chain
    protein_id = f"{Path(cif_file).stem}_{chain_id}"
    
    # Convert to PT data
    pt_data = sequence_to_pt_data(sequence, protein_id)
    
    # Save to file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(pt_data, output_file)
        logger.info(f"Saved {protein_id} with {len(sequence)} residues to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving {output_file}: {e}")
        return False


def convert_from_csv(csv_file: str, cif_dir: str, output_dir: str, processed_subdir: str = "processed") -> None:
    """
    Convert multiple CIF files to PT format based on a CSV file.
    Saves PT files to the shared DATA_PATH/processed directory.
    
    Args:
        csv_file: Path to CSV file with protein information
        cif_dir: Directory containing CIF files
        output_dir: Output directory for PT files (ignored, uses DATA_PATH)
        processed_subdir: Subdirectory name for processed files
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Use shared DATA_PATH from environment, matching inference script expectations
    data_path = os.environ.get('DATA_PATH', '/home/jupyter-chenxi/proteina/data')
    # Inference script expects files in DATA_PATH/pdb_train/processed/
    processed_dir = os.path.join(data_path, 'pdb_train', processed_subdir)
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"Saving PT files to shared data path: {processed_dir}")
    
    successful_conversions = 0
    failed_conversions = 0
    
    for _, row in df.iterrows():
        # Parse the natives_rcsb field (e.g., "1a2y_C")
        natives_rcsb = row['natives_rcsb']
        if pd.isna(natives_rcsb) or natives_rcsb == '':
            continue
            
        pdb_id, chain_id = natives_rcsb.split('_')
        
        # Find CIF file (check subdirectories)
        cif_file = None
        for subdir in os.listdir(cif_dir):
            subdir_path = os.path.join(cif_dir, subdir)
            if os.path.isdir(subdir_path):
                potential_cif = os.path.join(subdir_path, f"{pdb_id}.cif")
                if os.path.exists(potential_cif):
                    cif_file = potential_cif
                    break
        
        if cif_file is None:
            logger.warning(f"CIF file not found for {pdb_id}")
            failed_conversions += 1
            continue
        
        # Output PT file to shared data path
        output_file = os.path.join(processed_dir, f"{natives_rcsb}.pt")
        
        # Convert
        if convert_cif_to_pt(cif_file, chain_id, output_file):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    logger.info(f"Conversion complete: {successful_conversions} successful, {failed_conversions} failed")


def main():
    parser = argparse.ArgumentParser(description="Convert CIF files to PT format for Proteina inference")
    parser.add_argument("--cif_file", type=str, help="Input CIF file")
    parser.add_argument("--chain_id", type=str, help="Chain identifier to extract")
    parser.add_argument("--output_file", type=str, help="Output PT file")
    parser.add_argument("--csv_file", type=str, help="CSV file with protein information")
    parser.add_argument("--cif_dir", type=str, help="Directory containing CIF files")
    parser.add_argument("--output_dir", type=str, help="Output directory for PT files")
    
    args = parser.parse_args()
    
    if args.csv_file and args.cif_dir and args.output_dir:
        # Batch conversion from CSV
        convert_from_csv(args.csv_file, args.cif_dir, args.output_dir)
    elif args.cif_file and args.chain_id and args.output_file:
        # Single file conversion
        success = convert_cif_to_pt(args.cif_file, args.chain_id, args.output_file)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
