#!/usr/bin/env python
"""Backfill ext_lig labels into existing .pt files without full reprocessing.

Usage:
    # Backfill PDB .pt files (compute from raw deposits):
    python script_utils/backfill_ext_lig.py \
        --processed_dir data/pdb_train/processed \
        --raw_dir data/pdb_train/raw \
        --format cif \
        --database pdb \
        --workers 16

    # Backfill AFDB .pt files (set all unknown):
    python script_utils/backfill_ext_lig.py \
        --processed_dir data/d_FS/processed \
        --database afdb

    # Dry run to check how many files need backfill:
    python script_utils/backfill_ext_lig.py \
        --processed_dir data/pdb_train/processed \
        --dry_run
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from proteinfoundation.datasets.ext_lig_utils import (
    compute_ext_lig_from_df,
    make_unknown_ext_lig,
)
from proteinfoundation.graphein_utils.graphein_utils import read_pdb_to_dataframe


def _backfill_single(
    pt_path: str,
    raw_dir: str,
    format_type: str,
    database: str,
    overwrite: bool,
) -> dict:
    """Backfill ext_lig for a single .pt file. Returns stats dict."""
    stats = {"path": pt_path, "status": "skipped", "time_s": 0.0}
    t0 = time.time()
    try:
        graph = torch.load(pt_path, weights_only=False)
        if hasattr(graph, "ext_lig") and not overwrite:
            stats["status"] = "already_has_ext_lig"
            return stats

        L = graph.coords.shape[0]

        if database == "afdb":
            graph.ext_lig = make_unknown_ext_lig(L)
        else:
            graph_id = graph.id
            parts = graph_id.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) <= 2:
                pdb_code, chain = parts
                chains = chain
            else:
                pdb_code = graph_id
                chains = "all"

            raw_path = _find_raw_file(raw_dir, pdb_code, format_type)
            if raw_path is None:
                stats["status"] = "raw_not_found"
                stats["time_s"] = time.time() - t0
                return stats

            full_df = read_pdb_to_dataframe(path=raw_path)
            graph.ext_lig = compute_ext_lig_from_df(
                full_df=full_df,
                self_chains=chains,
                graph_residue_ids=list(graph.residue_id),
            )

        torch.save(graph, pt_path)
        stats["status"] = "backfilled"
        present = int((graph.ext_lig == 1).sum())
        absent = int((graph.ext_lig == 0).sum())
        unknown = int((graph.ext_lig == 2).sum())
        stats["present"] = present
        stats["absent"] = absent
        stats["unknown"] = unknown
    except Exception as e:
        stats["status"] = f"error: {e}"

    stats["time_s"] = time.time() - t0
    return stats


def _find_raw_file(raw_dir, pdb_code, format_type):
    for name in [
        f"{pdb_code.lower()}.{format_type}",
        f"{pdb_code}.{format_type}",
        f"{pdb_code.lower()}.{format_type}.gz",
        f"{pdb_code}.{format_type}.gz",
    ]:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed_dir", required=True, help="Directory containing .pt files")
    parser.add_argument("--raw_dir", default=None, help="Directory containing raw structure files (required for PDB)")
    parser.add_argument("--format", default="cif", help="Raw file format (cif, pdb, mmtf)")
    parser.add_argument("--database", default="pdb", choices=["pdb", "afdb"], help="Database type")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ext_lig")
    parser.add_argument("--dry_run", action="store_true", help="Only count files needing backfill")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    pt_files = sorted(processed_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {processed_dir}")

    if args.dry_run:
        need_backfill = 0
        for pt_path in tqdm(pt_files, desc="Checking"):
            graph = torch.load(str(pt_path), weights_only=False)
            if not hasattr(graph, "ext_lig"):
                need_backfill += 1
        print(f"{need_backfill} / {len(pt_files)} files need ext_lig backfill")
        return

    if args.database == "pdb" and args.raw_dir is None:
        parser.error("--raw_dir is required for PDB backfill")

    from functools import partial
    fn = partial(
        _backfill_single,
        raw_dir=args.raw_dir or "",
        format_type=args.format,
        database=args.database,
        overwrite=args.overwrite,
    )

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fn, str(p)): p for p in pt_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Backfilling"):
                results.append(future.result())
    else:
        for p in tqdm(pt_files, desc="Backfilling"):
            results.append(fn(str(p)))

    statuses = {}
    total_time = 0.0
    for r in results:
        s = r["status"]
        statuses[s] = statuses.get(s, 0) + 1
        total_time += r.get("time_s", 0.0)

    print(f"\nResults ({len(results)} files, {total_time:.1f}s total):")
    for s, count in sorted(statuses.items()):
        print(f"  {s}: {count}")


if __name__ == "__main__":
    main()
