#!/usr/bin/env python3
"""
Input parsing utilities for the prediction pipeline.

Handles CSV and FASTA input files, creates PT files from sequences,
and writes working CSVs for downstream pipeline scripts.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import torch

from proteinfoundation.af2rank_evaluation.cif_to_pt_converter import sequence_to_pt_data
from proteinfoundation.utils.cluster_utils import fasta_to_df

logger = logging.getLogger(__name__)


def parse_input(input_file: str, id_column: str = "id", sequence_column: str = "sequence") -> pd.DataFrame:
    """
    Parse input file (CSV or FASTA) and return DataFrame with 'id' and 'sequence' columns.

    Auto-detects format by file extension:
      - .csv / .tsv -> CSV/TSV
      - .fasta / .fa / .faa -> FASTA

    Args:
        input_file: Path to input CSV or FASTA file.
        id_column: Column name for protein ID in CSV (ignored for FASTA).
        sequence_column: Column name for sequence in CSV (ignored for FASTA).

    Returns:
        DataFrame with columns ['id', 'sequence'].
    """
    ext = Path(input_file).suffix.lower()

    if ext in (".fasta", ".fa", ".faa"):
        df = fasta_to_df(input_file)
        # fasta_to_df returns columns ['id', 'sequence']
    elif ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(input_file, sep=sep)
        if id_column not in df.columns:
            raise KeyError(f"Column '{id_column}' not found in {input_file}. Available: {list(df.columns)}")
        if sequence_column not in df.columns:
            raise KeyError(f"Column '{sequence_column}' not found in {input_file}. Available: {list(df.columns)}")
        df = df[[id_column, sequence_column]].rename(columns={id_column: "id", sequence_column: "sequence"})
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .csv, .tsv, .fasta, .fa, or .faa")

    # Validation
    df = df.dropna(subset=["id", "sequence"])
    df["id"] = df["id"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    if df.empty:
        raise ValueError(f"No valid entries found in {input_file}")

    dupes = df[df["id"].duplicated(keep=False)]
    if not dupes.empty:
        raise ValueError(f"Duplicate IDs found: {dupes['id'].unique().tolist()}")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    for _, row in df.iterrows():
        invalid = set(row["sequence"]) - valid_aa
        if invalid:
            logger.warning(f"Protein '{row['id']}' has non-standard characters: {invalid}. They will be treated as unknown (X).")

    logger.info(f"Parsed {len(df)} proteins from {input_file}")
    return df.reset_index(drop=True)


def create_pt_files(df: pd.DataFrame) -> str:
    """
    Create PT files from sequences for Proteina inference.

    Saves to {DATA_PATH}/pdb_train/processed/{id}.pt.
    Skips if PT file already exists.

    Args:
        df: DataFrame with 'id' and 'sequence' columns.

    Returns:
        Path to the processed directory.
    """
    data_path = os.environ.get("DATA_PATH", os.path.join(os.getcwd(), "data"))
    processed_dir = os.path.join(data_path, "pdb_train", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    created = 0
    skipped = 0
    for _, row in df.iterrows():
        protein_id = row["id"]
        sequence = row["sequence"]
        output_path = os.path.join(processed_dir, f"{protein_id}.pt")

        if os.path.exists(output_path):
            skipped += 1
            continue

        pt_data = sequence_to_pt_data(list(sequence), protein_id)
        torch.save(pt_data, output_path)
        created += 1

    logger.info(f"PT files: {created} created, {skipped} already existed in {processed_dir}")
    return processed_dir


def create_working_csv(df: pd.DataFrame, output_path: str) -> str:
    """
    Write a working CSV with 'id' column for downstream pipeline scripts.

    Args:
        df: DataFrame with 'id' column.
        output_path: Path to write the CSV.

    Returns:
        Path to the written CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[["id"]].to_csv(output_path, index=False)
    logger.info(f"Working CSV written to {output_path} ({len(df)} proteins)")
    return output_path
