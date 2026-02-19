#!/usr/bin/env python3
"""Find and delete .pt files that lack coords, residue_type, or coord_mask.

Processed .pt files are expected to have graph.coords, graph.residue_type, and
graph.coord_mask. Files missing any of these (or that fail to load) are invalid
and can cause KeyError during training/validation. This script identifies and
optionally removes them.

Usage:
    python -m proteinfoundation.utils.purge_invalid_pt_files
    python -m proteinfoundation.utils.purge_invalid_pt_files --processed-dir /path/to/processed --dry-run
    python -m proteinfoundation.utils.purge_invalid_pt_files --workers 16
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[2] / "data"))
PROCESSED_DIR_DEFAULT = DATA_PATH / "pdb_train" / "processed"


def _check_one(path: Path) -> Tuple[str, bool, str]:
    """Check a single .pt file. Returns (fname, valid, reason). Used by workers."""
    try:
        graph = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as e:
        return path.name, False, f"load_failed: {e}"

    if not hasattr(graph, "coords") or graph.coords is None:
        return path.name, False, "missing_coords"
    if not hasattr(graph, "coord_mask") or graph.coord_mask is None:
        return path.name, False, "missing_coord_mask"
    if not hasattr(graph, "residue_type") or graph.residue_type is None:
        return path.name, False, "missing_residue_type"

    try:
        n = graph.coords.shape[0]
        if n == 0:
            return path.name, False, "empty_coords"
        if hasattr(graph.residue_type, "shape") and graph.residue_type.shape[0] != n:
            return path.name, False, f"length_mismatch coords={n} residue_type={graph.residue_type.shape[0]}"
    except Exception as e:
        return path.name, False, f"shape_check: {e}"

    return path.name, True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Find and delete .pt files without coords or residue_type"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR_DEFAULT,
        help="Processed .pt directory (default: $DATA_PATH/pdb_train/processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report invalid files, do not delete",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: half of CPU cores)",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist")
        sys.exit(1)

    pt_files = sorted(processed_dir.glob("*.pt"))
    workers = max(1, args.workers or (os.cpu_count() or 1) // 2)
    print(f"Checking {len(pt_files)} .pt files in {processed_dir} with {workers} workers")

    invalid = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(_check_one, p): p for p in pt_files}
        for future in as_completed(future_to_path):
            try:
                fname, valid, reason = future.result()
                if not valid:
                    invalid.append((fname, reason))
            except Exception as e:
                path = future_to_path[future]
                invalid.append((path.name, f"exception: {e}"))

    invalid.sort(key=lambda x: x[0])
    if not invalid:
        print("All files are valid.")
        return

    print(f"\nFound {len(invalid)} invalid file(s):")
    for fname, reason in invalid:
        print(f"  {fname}: {reason}")

    if args.dry_run:
        print("\n[DRY RUN] No files deleted. Run without --dry-run to delete.")
        return

    deleted = 0
    for fname, _ in invalid:
        path = processed_dir / fname
        try:
            path.unlink()
            deleted += 1
            print(f"Deleted: {fname}")
        except Exception as e:
            print(f"Failed to delete {fname}: {e}")

    print(f"\nDeleted {deleted}/{len(invalid)} invalid file(s).")


if __name__ == "__main__":
    main()
