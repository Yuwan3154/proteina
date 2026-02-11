#!/usr/bin/env python3
"""Run confind locally on failed examples from a confind precompute CSV.

Usage:
    python -m proteinfoundation.utils.run_confind_on_failed /path/to/confind_precompute_shard8_of_10.csv
    python -m proteinfoundation.utils.run_confind_on_failed /path/to/confind_precompute_shard8_of_10.csv --processed-dir /path/to/pdb_train/processed --rotlib /path/to/rotlibs --timeout 120
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[2] / "data"))
PROCESSED_DIR_DEFAULT = DATA_PATH / "pdb_train" / "processed"
ROTLIB_DEFAULT = DATA_PATH / "rotlibs"


def main():
    parser = argparse.ArgumentParser(description="Run confind on failed CSV rows")
    parser.add_argument("csv_path", type=Path, help="Path to confind_precompute_shard*_of_*.csv")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR_DEFAULT,
        help="Processed .pt directory (default: $DATA_PATH/pdb_train/processed)",
    )
    parser.add_argument(
        "--rotlib",
        type=str,
        default=str(ROTLIB_DEFAULT),
        help="Confind rotamer library path",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per confind run in seconds (default: 120)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max number of failed examples to try (0 = all)",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    # Collect failed rows
    failed_files = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "failed":
                fname = row.get("file", "").strip()
                if fname.endswith(".pt"):
                    failed_files.append(fname)

    print(f"Found {len(failed_files)} failed rows in {csv_path}")
    if not failed_files:
        print("Nothing to run.")
        return

    if args.max > 0:
        failed_files = failed_files[: args.max]
        print(f"Running confind on first {len(failed_files)} failed examples (--max {args.max})")
    else:
        print(f"Running confind on all {len(failed_files)} failed examples")

    processed_dir = args.processed_dir
    rotlib = args.rotlib
    timeout = args.timeout

    if not processed_dir.is_dir():
        print(f"ERROR: Processed dir not found: {processed_dir}", file=sys.stderr)
        sys.exit(2)
    if not Path(rotlib).is_dir():
        print(f"WARNING: Rotlib not found: {rotlib}", file=sys.stderr)

    from proteinfoundation.utils.confind_utils import write_graph_pdb, run_confind, parse_confind_contacts

    ok = 0
    missing = 0
    fail_local = 0
    for fname in failed_files:
        pt_path = processed_dir / fname
        if not pt_path.exists():
            print(f"  SKIP (missing): {fname}")
            missing += 1
            continue
        try:
            graph = torch.load(pt_path, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"  SKIP (load error): {fname} -> {e}")
            fail_local += 1
            continue
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = Path(tmp) / "input.pdb"
            contact_path = Path(tmp) / "contacts.txt"
            try:
                write_graph_pdb(graph, pdb_path)
            except Exception as e:
                print(f"  FAIL (write_pdb): {fname} -> {e}")
                fail_local += 1
                continue
            try:
                result = subprocess.run(
                    [
                        "confind",
                        "--p", str(pdb_path),
                        "--o", str(contact_path),
                        "--rLib", rotlib,
                        "--ren",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                )
            except subprocess.TimeoutExpired:
                print(f"  FAIL (timeout {timeout}s): {fname}")
                fail_local += 1
                continue
            except FileNotFoundError:
                print(f"  FAIL (confind not in PATH): {fname}")
                fail_local += 1
                continue
            except Exception as e:
                print(f"  FAIL (run): {fname} -> {e}")
                fail_local += 1
                continue
            if result.returncode != 0:
                stderr = (result.stderr or "")[:200]
                print(f"  FAIL (confind exit {result.returncode}): {fname} stderr={stderr}")
                fail_local += 1
                continue
            try:
                cm = parse_confind_contacts(
                    contact_path, expected_len=graph.coords.shape[0], renumbered=True
                )
            except Exception as e:
                print(f"  FAIL (parse): {fname} -> {e}")
                fail_local += 1
                continue
            print(f"  OK: {fname} contact_map {cm.shape}")
            ok += 1

    print()
    print(f"Summary: ok={ok} missing={missing} fail_local={fail_local} total={len(failed_files)}")
    if fail_local > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
