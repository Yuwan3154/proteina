#!/usr/bin/env python3
"""
Compatibility wrapper for the old diversity-only API.

The implementation now lives in `proteina_analysis.py`.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

from proteinfoundation.af2rank_evaluation.proteina_analysis import (
    compute_diversity_for_proteins,
    compute_pairwise_tm,
    find_diversity_summaries,
    load_diversity_data,
    plot_pairwise_tm_histogram,
    resolve_num_workers,
    run_template_template_dir,
)


def _pairwise_tm_via_usalign_dir(
    protein_dir: str,
    basenames: List[str],
    env: Optional[Dict[str, str]],
) -> List[float]:
    del env
    return [float(row["tms"]) for row in run_template_template_dir(protein_dir, basenames, env=None)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Proteina sample diversity (all-to-all TMscore)")
    parser.add_argument("--inference_dir", required=True, help="Base inference directory")
    parser.add_argument("--protein_ids", nargs="*", help="Protein IDs to process")
    parser.add_argument("--csv_file", help="CSV file with protein IDs (alternative to --protein_ids)")
    parser.add_argument("--csv_col", default="id", help="Column name for protein ID in CSV (default: id)")
    parser.add_argument("--output_subdir", default="proteina_diversity", help="Per-protein subdirectory for outputs")
    parser.add_argument("--rerun", action="store_true", help="Recompute even if results exist")
    parser.add_argument("--num_workers", type=int, default=None, help="Max parallel worker count")
    parser.add_argument("--no_usalign_dir", action="store_true", help="Disable USalign -dir all-against-all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.protein_ids:
        protein_ids = args.protein_ids
    elif args.csv_file:
        df = pd.read_csv(args.csv_file)
        protein_ids = df[args.csv_col].dropna().astype(str).str.strip().unique().tolist()
    else:
        protein_ids = sorted(
            [
                name
                for name in os.listdir(args.inference_dir)
                if os.path.isdir(os.path.join(args.inference_dir, name))
            ]
        )

    if not protein_ids:
        raise ValueError("No protein IDs found")

    results = compute_diversity_for_proteins(
        inference_dir=args.inference_dir,
        protein_ids=protein_ids,
        output_subdir=args.output_subdir,
        skip_existing=not args.rerun,
        num_workers=args.num_workers,
        use_usalign_dir=not args.no_usalign_dir,
    )
    logging.getLogger(__name__).info(f"Done. {len(results)} proteins with diversity metrics.")


if __name__ == "__main__":
    main()
