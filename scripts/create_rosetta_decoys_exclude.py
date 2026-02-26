#!/usr/bin/env python3
"""
Create rosetta_decoys.txt from af2rank_single_set.csv natives_rcsb column,
and split into val/test files according to train_val_test ratios.

Usage:
  python scripts/create_rosetta_decoys_exclude.py \\
    --csv /path/to/af2rank_single_set.csv \\
    --output-dir ${DATA_PATH}/pdb_train \\
    --train-val-test 0.98 0.019 0.001

Output:
  - {output_dir}/rosetta_decoys.txt (all chain IDs, one per line)
  - {output_dir}/rosetta_decoys_val.txt (val portion)
  - {output_dir}/rosetta_decoys_test.txt (test portion)
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to af2rank_single_set.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory (e.g. ${DATA_PATH}/pdb_train)",
    )
    parser.add_argument(
        "--train-val-test",
        nargs=3,
        type=float,
        default=[0.98, 0.019, 0.001],
        help="Train/val/test ratios (default: 0.98 0.019 0.001)",
    )
    parser.add_argument(
        "--base-name",
        default="rosetta_decoys",
        help="Base name for output files (default: rosetta_decoys)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for val/test split (default: 42)",
    )
    args = parser.parse_args()

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        if "natives_rcsb" not in reader.fieldnames:
            raise ValueError(f"CSV must have 'natives_rcsb' column, got: {reader.fieldnames}")
        chains = []
        seen = set()
        for row in reader:
            val = row.get("natives_rcsb", "").strip()
            if val and val not in seen:
                seen.add(val)
                chains.append(val)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write full list
    full_path = output_dir / f"{args.base_name}.txt"
    with open(full_path, "w") as f:
        for c in chains:
            f.write(f"{c}\n")
    print(f"Wrote {len(chains)} chain IDs to {full_path}")

    # Split into val/test according to train_val_test
    train_ratio, val_ratio, test_ratio = args.train_val_test
    val_frac = val_ratio / (val_ratio + test_ratio)  # fraction of non-train that goes to val
    test_frac = 1.0 - val_frac

    n_val = int(len(chains) * val_frac)
    n_test = len(chains) - n_val

    # Deterministic shuffle
    import random
    random.seed(args.seed)
    shuffled = chains.copy()
    random.shuffle(shuffled)
    val_chains = shuffled[:n_val]
    test_chains = shuffled[n_val:]

    # Write val and test
    val_path = output_dir / f"{args.base_name}_val.txt"
    test_path = output_dir / f"{args.base_name}_test.txt"
    with open(val_path, "w") as f:
        for c in val_chains:
            f.write(f"{c}\n")
    with open(test_path, "w") as f:
        for c in test_chains:
            f.write(f"{c}\n")
    print(f"Wrote {len(val_chains)} to {val_path}, {len(test_chains)} to {test_path}")


if __name__ == "__main__":
    main()
