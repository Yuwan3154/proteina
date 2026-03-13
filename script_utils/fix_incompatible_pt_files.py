#!/usr/bin/env python
"""Find and fix .pt files that lack the coords attribute (incompatible format).

Such files cause errors like 'GlobalStorage' object has no attribute 'coords'.
The fix is to delete them and regenerate from raw using process_training_data.

Usage:
    # Dry run: list incompatible files only
    python script_utils/fix_incompatible_pt_files.py \
        --processed_dir /home/ubuntu/proteina/data/pdb_train/processed \
        --dry_run

    # Delete incompatible files, then print the regenerate command
    python script_utils/fix_incompatible_pt_files.py \
        --processed_dir /home/ubuntu/proteina/data/pdb_train/processed \
        --config configs/datasets_config/pdb/pdb_train_S25_max512_purge-test_cutoff-190828.yaml

    # Delete and immediately run regeneration (requires DATA_PATH)
    python script_utils/fix_incompatible_pt_files.py \
        --processed_dir /home/ubuntu/proteina/data/pdb_train/processed \
        --config configs/datasets_config/pdb/pdb_train_S25_max512_purge-test_cutoff-190828.yaml \
        --run_regenerate \
        --num_workers 32
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)


def _is_incompatible(pt_path: str) -> bool:
    """Return True if the .pt file lacks coords (incompatible format)."""
    try:
        graph = torch.load(pt_path, weights_only=False)
        coords = getattr(graph, "coords", None)
        if coords is None and hasattr(graph, "get"):
            coords = graph.get("coords")
        return coords is None
    except Exception:
        return True  # Treat load errors as incompatible


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processed_dir",
        required=True,
        help="Directory containing .pt files",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config YAML for process_training_data (required for --run_regenerate)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list incompatible files, do not delete",
    )
    parser.add_argument(
        "--run_regenerate",
        action="store_true",
        help="After deleting, run process_training_data to regenerate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Workers for process_training_data (default: 32)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation before deleting",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"ERROR: {processed_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    pt_files = sorted(processed_dir.glob("*.pt"))
    print(f"Scanning {len(pt_files)} .pt files in {processed_dir}")

    incompatible = []
    for pt_path in tqdm(pt_files, desc="Checking"):
        if _is_incompatible(str(pt_path)):
            incompatible.append(pt_path)

    if not incompatible:
        print("No incompatible files found.")
        return

    print(f"\nFound {len(incompatible)} incompatible files (no coords):")
    for p in incompatible[:20]:
        print(f"  {p.name}")
    if len(incompatible) > 20:
        print(f"  ... and {len(incompatible) - 20} more")

    if args.dry_run:
        print("\nDry run: no files deleted. Run without --dry_run to delete and regenerate.")
        return

    if not args.yes:
        resp = input(f"\nDelete these {len(incompatible)} files? [y/N] ")
        if resp.lower() != "y":
            print("Aborted.")
            return

    for pt_path in tqdm(incompatible, desc="Deleting"):
        pt_path.unlink()

    print(f"Deleted {len(incompatible)} incompatible files.")

    if args.run_regenerate:
        if not args.config:
            print("ERROR: --config required for --run_regenerate", file=sys.stderr)
            sys.exit(1)
        if not os.environ.get("DATA_PATH"):
            print(
                "ERROR: DATA_PATH must be set for process_training_data. "
                "E.g. export DATA_PATH=/home/ubuntu/proteina/data",
                file=sys.stderr,
            )
            sys.exit(1)

        script_path = Path(root) / "proteinfoundation" / "scripts" / "process_training_data.py"
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path(root) / config_path
        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--num_workers",
            str(args.num_workers),
            "--regenerate_missing",
            "true",
        ]
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=root)
    else:
        config_hint = f" --config {args.config}" if args.config else ""
        script = "proteinfoundation/scripts/process_training_data.py"
        print(
            f"\nTo regenerate the deleted files, run:\n"
            f"  DATA_PATH=/path/to/data python {script}{config_hint} \\\n"
            f"    --num_workers {args.num_workers} --regenerate_missing true"
        )


if __name__ == "__main__":
    main()
