#!/usr/bin/env python3
"""
Generate tms_single values for Bad AFDB dataset

This script:
1. Downloads AlphaFold predictions for each uniprot ID
2. Extracts residues at specified indices
3. Runs USalign to compute TM-scores
4. Adds tms_single column to the dataset

Usage:
    python generate_tms_single.py \
        --dataset_csv /path/to/dataset.csv \
        --indices_csv /path/to/indices.csv \
        --cif_dir /path/to/cif/files \
        --output_csv /path/to/output.csv \
        --afdb_cache_dir /path/to/afdb/cache \
        --usalign_path /path/to/USalign \
        --num_workers 4
"""

import os
import sys
import argparse
import pandas as pd
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm

# Add BioPython imports
from Bio.PDB import MMCIFParser, PDBIO, Select
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)


class ResidueSelect(Select):
    """Select specific residues by index."""
    def __init__(self, residue_indices: List[int]):
        self.residue_indices = set(residue_indices)
    
    def accept_residue(self, residue):
        # Get residue index (0-based)
        res_idx = residue.get_parent().get_list().index(residue)
        return res_idx in self.residue_indices


def parse_indices_csv(indices_csv: str) -> Dict[str, List[int]]:
    """
    Parse the indices CSV file.
    
    Format: Lines alternate between protein info and indices.
    Example:
        7M4X_G,B7IA24,1.0,93.27620689655173,0.9830508474576272,174
        1,2,3,4,5,6,7,8,...
    
    Returns:
        Dictionary mapping pdb_chain to list of residue indices (0-based)
    """
    indices_map = {}
    
    with open(indices_csv, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Process pairs of lines
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        
        # First line: protein info
        info_parts = lines[i].split(',')
        pdb_chain = info_parts[0]
        
        # Second line: indices (0-based)
        indices_str = lines[i + 1]
        indices = [int(idx) for idx in indices_str.split(',')]
        
        indices_map[pdb_chain] = indices
        logger.debug(f"Loaded {len(indices)} indices for {pdb_chain}")
    
    logger.info(f"Parsed indices for {len(indices_map)} proteins")
    return indices_map


def download_afdb_prediction(uniprot_id: str, cache_dir: str) -> Optional[str]:
    """
    Download AlphaFold prediction CIF file.
    
    Args:
        uniprot_id: UniProt accession
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Path to downloaded CIF file, or None if download failed
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cif_path = os.path.join(cache_dir, f"AF-{uniprot_id}-F1-model_v6.cif")
    
    # Check if already downloaded
    if os.path.exists(cif_path):
        logger.debug(f"Using cached AFDB prediction: {cif_path}")
        return cif_path
    
    # Download
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.cif"
    logger.debug(f"Downloading {url}")
    
    result = subprocess.run(
        ['wget', '-nv', '-O', cif_path, url],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0 and os.path.exists(cif_path):
        # Check if file is not empty
        if os.path.getsize(cif_path) > 0:
            logger.debug(f"Downloaded AFDB prediction: {cif_path}")
            return cif_path
        else:
            logger.warning(f"Downloaded empty file for {uniprot_id}")
            os.remove(cif_path)
            return None
    else:
        logger.warning(f"Failed to download {url}: {result.stderr}")
        if os.path.exists(cif_path):
            os.remove(cif_path)
        return None


def extract_residues_from_cif(cif_path: str, residue_indices: List[int], output_path: str) -> bool:
    """
    Extract specific residues from CIF file.
    
    Args:
        cif_path: Path to input CIF file
        residue_indices: List of 0-based residue indices to extract
        output_path: Path to save extracted structure
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(cif_path):
        logger.warning(f"CIF file does not exist: {cif_path}")
        return False
    
    if os.path.getsize(cif_path) == 0:
        logger.warning(f"CIF file is empty: {cif_path}")
        return False
    
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', cif_path)
    
    # Save with residue selection
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ResidueSelect(residue_indices))
    
    logger.debug(f"Extracted {len(residue_indices)} residues to {output_path}")
    return True


def run_usalign(pdb_path: str, pdb_chain: str, afdb_path: str, usalign_path: str) -> Optional[float]:
    """
    Run USalign to compute TM-score.
    
    Args:
        pdb_path: Path to PDB/CIF file
        pdb_chain: Chain ID in PDB
        afdb_path: Path to AFDB prediction (segmented)
        usalign_path: Path to USalign executable
        
    Returns:
        TM1 score (normalized by chain 1), or None if failed
    """
    cmd = [
        usalign_path,
        pdb_path,
        '-chain1', pdb_chain,
        afdb_path,
        '-chain2', 'A',
        '-outfmt', '2'
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        logger.warning(f"USalign failed with code {result.returncode}")
        return None
    
    # Parse TSV output
    # Expected format: header line, then data line with TM1 and TM2
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        logger.warning(f"USalign output too short: {result.stdout}")
        return None
    
    # Parse data line (skip header)
    data_line = lines[1].strip().split('\t')
    
    # Find TM1 (normalized by chain 1)
    # Typical format: name1 name2 TM1 TM2 ...
    if len(data_line) < 4:
        logger.warning(f"USalign output format unexpected: {data_line}")
        return None
    
    tm1 = float(data_line[2])
    logger.debug(f"USalign TM1: {tm1}")
    return tm1


def process_single_protein(args_tuple) -> Tuple[str, Optional[float], str]:
    """
    Process a single protein: download, extract, align.
    
    Args:
        args_tuple: (pdb_chain, uniprot_id, indices, cif_dir, afdb_cache_dir, usalign_path)
        
    Returns:
        (pdb_chain, tms_single, status_message)
    """
    pdb_chain, uniprot_id, indices, cif_dir, afdb_cache_dir, usalign_path = args_tuple
    
    # Extract PDB and chain from pdb_chain (e.g., "6guw_A" -> "6guw", "A")
    pdb_id, chain_id = pdb_chain.rsplit('_', 1)
    
    # Find PDB/CIF file
    pdb_file = None
    for subdir in [pdb_id[1:3].upper(), pdb_id[1:3].lower()]:
        for ext in ['.cif', '.pdb']:
            potential_path = os.path.join(cif_dir, subdir, f"{pdb_id.upper()}{ext}")
            if os.path.exists(potential_path):
                pdb_file = potential_path
                break
        if pdb_file:
            break
    
    if not pdb_file:
        return (pdb_chain, None, f"PDB file not found: {pdb_id}")
    
    # Download AFDB prediction
    afdb_cif = download_afdb_prediction(uniprot_id, afdb_cache_dir)
    if not afdb_cif:
        logger.warning(f"{pdb_chain}: Failed to download AFDB for {uniprot_id}, setting TM=0.0")
        return (pdb_chain, 0.0, f"Failed to download AFDB: {uniprot_id}")
    
    # Extract residues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        segmented_afdb = tmp.name
    
    try:
        if not extract_residues_from_cif(afdb_cif, indices, segmented_afdb):
            logger.warning(f"{pdb_chain}: Failed to extract residues, setting TM=0.0")
            return (pdb_chain, 0.0, "Failed to extract residues")
        
        # Run USalign
        tm_score = run_usalign(pdb_file, chain_id, segmented_afdb, usalign_path)
        
        if tm_score is None:
            logger.warning(f"{pdb_chain}: USalign failed, setting TM=0.0")
            return (pdb_chain, 0.0, "USalign failed")
        
        return (pdb_chain, tm_score, "Success")
    
    finally:
        # Clean up temp file
        if os.path.exists(segmented_afdb):
            os.remove(segmented_afdb)


def main():
    parser = argparse.ArgumentParser(description='Generate tms_single values for Bad AFDB dataset')
    parser.add_argument('--dataset_csv', required=True, help='Input dataset CSV file')
    parser.add_argument('--indices_csv', required=True, help='Residue indices CSV file')
    parser.add_argument('--cif_dir', required=True, help='Directory containing PDB/CIF files')
    parser.add_argument('--output_csv', required=True, help='Output CSV file with tms_single column')
    parser.add_argument('--afdb_cache_dir', default='./afdb_cache', help='Directory to cache AFDB predictions')
    parser.add_argument('--usalign_path', default='USalign', help='Path to USalign executable')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    
    logger.info("🔬 Starting tms_single generation")
    logger.info(f"📄 Dataset CSV: {args.dataset_csv}")
    logger.info(f"📄 Indices CSV: {args.indices_csv}")
    logger.info(f"📂 CIF directory: {args.cif_dir}")
    logger.info(f"💾 Output CSV: {args.output_csv}")
    logger.info(f"🌐 AFDB cache: {args.afdb_cache_dir}")
    logger.info(f"⚙️  USalign: {args.usalign_path}")
    logger.info(f"👥 Workers: {args.num_workers}")
    
    # Validate inputs
    if not os.path.exists(args.dataset_csv):
        logger.error(f"Dataset CSV not found: {args.dataset_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.indices_csv):
        logger.error(f"Indices CSV not found: {args.indices_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.cif_dir):
        logger.error(f"CIF directory not found: {args.cif_dir}")
        sys.exit(1)
    
    # Check USalign
    result = subprocess.run([args.usalign_path, '-h'], capture_output=True, timeout=5)
    if result.returncode != 0:
        logger.warning(f"USalign check returned non-zero: {result.returncode}")

    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_csv(args.dataset_csv)
    logger.info(f"Loaded {len(df)} proteins")
    
    # Load indices
    logger.info("Loading residue indices...")
    indices_map = parse_indices_csv(args.indices_csv)
    
    # Check coverage
    pdb_chains = df['pdb'].tolist()
    missing_indices = [p for p in pdb_chains if p not in indices_map]
    if missing_indices:
        logger.warning(f"Missing indices for {len(missing_indices)} proteins")
        logger.warning(f"First 10: {missing_indices[:10]}")
    
    # Prepare tasks
    tasks = []
    for _, row in df.iterrows():
        pdb_chain = row['pdb']
        uniprot_id = row['uniprot']
        
        if pdb_chain not in indices_map:
            logger.warning(f"No indices found for {pdb_chain}, skipping")
            continue
        
        indices = indices_map[pdb_chain]
        
        tasks.append((
            pdb_chain,
            uniprot_id,
            indices,
            args.cif_dir,
            args.afdb_cache_dir,
            args.usalign_path
        ))
    
    logger.info(f"Processing {len(tasks)} proteins with {args.num_workers} workers...")
    
    # Process in parallel
    results = {}
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_protein, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing proteins"):
            pdb_chain = futures[future]
            result_pdb_chain, tm_score, status = future.result()
            results[result_pdb_chain] = tm_score

    # Add tms_single column to dataframe
    logger.info("Adding tms_single column to dataset...")
    df['tms_single'] = df['pdb'].map(results)
    
    # Save output
    logger.info(f"Saving output to {args.output_csv}...")
    df.to_csv(args.output_csv, index=False)
    
    # Summary
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total proteins: {len(df)}")
    logger.info(f"Mean TM-score: {df['tms_single'].mean():.4f}")
    logger.info(f"Median TM-score: {df['tms_single'].median():.4f}")
    logger.info("="*60)
    logger.info(f"✅ Output saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
