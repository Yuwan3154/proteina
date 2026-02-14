#!/usr/bin/env python3
"""Collect confind contact maps for failed examples in a precompute CSV and save to .pt files.

ONLY processes rows with status=="failed" from the CSV. Loads each .pt, runs confind,
adds contact_map_confind, and atomically saves back to the same file.

Usage:
    python -m proteinfoundation.utils.collect_confind_failed /path/to/confind_precompute_shard8_of_10.csv
    python -m proteinfoundation.utils.collect_confind_failed /path/to/confind_precompute_shard8_of_10.csv --processed-dir /path/to/processed --rotlib /path/to/rotlibs --timeout 120 --max 5 --workers 4
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[2] / "data"))
PROCESSED_DIR_DEFAULT = DATA_PATH / "pdb_train" / "processed"
ROTLIB_DEFAULT = DATA_PATH / "rotlibs"


def _process_one(
    path: Path,
    processed_dir: Path,
    rotlib: str,
    timeout: int,
    confind_bin: str = "confind",
) -> Dict:
    """Process a single failed file. Returns status dict."""
    try:
        graph = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as e:
        return {"file": path.name, "status": "skipped", "reason": f"load: {e}"}

    if hasattr(graph, "contact_map_confind") and graph.contact_map_confind is not None:
        cm = graph.contact_map_confind
        n_res = graph.coords.shape[0]
        if isinstance(cm, torch.Tensor) and cm.shape == (n_res, n_res):
            return {"file": path.name, "status": "already_done"}

    from proteinfoundation.utils.confind_utils import confind_raw_contact_map

    try:
        raw_map = confind_raw_contact_map(
            graph,
            rotlib_path=rotlib,
            confind_bin=confind_bin,
            omp_threads=1,
            renumber=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"file": path.name, "status": "skipped", "reason": f"timeout {timeout}s"}
    except Exception as e:
        return {"file": path.name, "status": "skipped", "reason": str(e)}

    graph.contact_map_confind = torch.as_tensor(raw_map, dtype=torch.float16)
    tmp_path = path.with_suffix(".pt.tmp")
    try:
        torch.save(graph, tmp_path)
        tmp_path.rename(path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return {"file": path.name, "status": "skipped", "reason": f"save: {e}"}

    return {"file": path.name, "status": "saved", "shape": tuple(graph.contact_map_confind.shape)}


def main():
    parser = argparse.ArgumentParser(
        description="Process failed confind examples and save contact_map_confind to .pt files"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to confind_precompute_shard*_of_*.csv (only status=='failed' rows are processed)",
    )
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
        help="Max number of failed examples to process (0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: half of CPU cores)",
    )
    parser.add_argument(
        "--confind-bin",
        type=str,
        default="confind",
        help="Path to confind binary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List failed files that would be processed, do not run confind or save",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    # Collect ONLY failed rows
    failed_files = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "failed":
                fname = row.get("file", "").strip()
                if fname.endswith(".pt"):
                    failed_files.append(fname)

    print(f"Found {len(failed_files)} failed rows (status=='failed') in {csv_path}")
    if not failed_files:
        print("Nothing to process.")
        return

    if args.max > 0:
        failed_files = failed_files[: args.max]
        print(f"Processing first {len(failed_files)} failed examples (--max {args.max})")
    else:
        print(f"Processing all {len(failed_files)} failed examples")

    processed_dir = args.processed_dir
    rotlib = args.rotlib
    timeout = args.timeout
    confind_bin = args.confind_bin

    if args.workers is None:
        workers = max(1, (os.cpu_count() or 1) // 2)
    else:
        workers = max(1, args.workers)

    if not processed_dir.is_dir():
        print(f"ERROR: Processed dir not found: {processed_dir}", file=sys.stderr)
        sys.exit(2)
    if not Path(rotlib).is_dir():
        print(f"WARNING: Rotlib not found: {rotlib}", file=sys.stderr)

    # Build list of paths (exclude missing)
    paths: List[Path] = []
    missing = 0
    for fname in failed_files:
        pt_path = processed_dir / fname
        if not pt_path.exists():
            missing += 1
            continue
        paths.append(pt_path)

    if args.dry_run:
        for pt_path in paths:
            print(f"  {pt_path.name} (exists)")
        print(f"\nDry run: would process {len(paths)} files ({missing} missing). Remove --dry-run to run.")
        return

    print(f"Processing {len(paths)} files with {workers} workers (missing={missing})")

    saved = 0
    skipped = 0
    already_done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(
                _process_one,
                path,
                processed_dir,
                rotlib,
                timeout,
                confind_bin,
            ): path
            for path in paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  SKIP (exception): {path.name if path else '?'} -> {e}")
                skipped += 1
                continue
            status = result.get("status", "")
            fname = result.get("file", "?")
            if status == "saved":
                print(f"  SAVED: {fname} contact_map {result.get('shape', '')}")
                saved += 1
            elif status == "already_done":
                print(f"  SKIP (already has confind): {fname}")
                already_done += 1
            else:
                reason = result.get("reason", "?")
                print(f"  SKIP ({reason}): {fname}")
                skipped += 1

    print()
    print(
        f"Summary: saved={saved} already_done={already_done} missing={missing} "
        f"skipped={skipped} total={len(failed_files)}"
    )


if __name__ == "__main__":
    main()
