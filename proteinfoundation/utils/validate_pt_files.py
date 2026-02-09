#!/usr/bin/env python3
"""Validate processed .pt files for corruption and attribute consistency.

Usage:
    python validate_pt_files.py /path/to/data/pdb_train/processed/ [--sample N] [--workers W]
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch


def _validate_one(pt_path_str):
    pt_path = Path(pt_path_str)
    result = {
        "file": pt_path.name, "loadable": False, "has_coords": False,
        "has_coord_mask": False, "has_contact_map_confind": False,
        "has_protein_id": False, "coords_shape": "", "confind_shape": "",
        "file_size": 0, "error": "",
    }
    try:
        result["file_size"] = pt_path.stat().st_size
    except Exception:
        pass
    try:
        graph = torch.load(pt_path, weights_only=False, map_location="cpu")
        result["loadable"] = True
    except Exception as e:
        result["error"] = repr(e)[:200]
        return result
    result["has_coords"] = hasattr(graph, "coords") and graph.coords is not None
    result["has_coord_mask"] = hasattr(graph, "coord_mask") and graph.coord_mask is not None
    result["has_contact_map_confind"] = hasattr(graph, "contact_map_confind") and graph.contact_map_confind is not None
    result["has_protein_id"] = hasattr(graph, "protein_id") and graph.protein_id is not None
    if result["has_coords"]:
        result["coords_shape"] = str(list(graph.coords.shape))
    if result["has_contact_map_confind"]:
        try:
            result["confind_shape"] = str(list(graph.contact_map_confind.shape))
        except Exception:
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate .pt files")
    parser.add_argument("processed_dir", type=str)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    all_files = sorted(processed_dir.glob("*.pt"))
    print(f"Found {len(all_files)} .pt files")

    if args.sample > 0 and args.sample < len(all_files):
        files = random.sample(all_files, args.sample)
        print(f"Sampling {args.sample} files")
    else:
        files = all_files

    output_path = Path(args.output) if args.output else processed_dir / "validation_results.csv"
    results = []
    start = time.perf_counter()
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_validate_one, str(f)): f for f in files}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
            except Exception as e:
                result = {"file": futures[future].name, "error": repr(e)[:200]}
            results.append(result)
            if done % 5000 == 0 or done == len(files):
                elapsed = time.perf_counter() - start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  Validated {done}/{len(files)} ({rate:.0f}/s)")

    n_total = len(results)
    n_loadable = sum(1 for r in results if r.get("loadable", False))
    n_corrupted = n_total - n_loadable
    n_no_coords = sum(1 for r in results if r.get("loadable") and not r.get("has_coords"))
    n_has_confind = sum(1 for r in results if r.get("has_contact_map_confind", False))
    n_has_pid = sum(1 for r in results if r.get("has_protein_id", False))
    n_zero = sum(1 for r in results if r.get("file_size", 0) == 0)

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY ({n_total} files)")
    print(f"{'='*60}")
    print(f"  Loadable:                {n_loadable:>7} / {n_total}")
    print(f"  CORRUPTED:               {n_corrupted:>7} / {n_total}")
    print(f"  Zero-size:               {n_zero:>7}")
    print(f"  Missing coords:          {n_no_coords:>7}")
    print(f"  Has contact_map_confind: {n_has_confind:>7} / {n_loadable}")
    print(f"  Has protein_id:          {n_has_pid:>7} / {n_loadable}")

    corrupted = [r for r in results if not r.get("loadable", False)]
    if corrupted:
        print("\nCORRUPTED FILES:")
        for r in corrupted[:50]:
            print(f"  {r['file']} (size={r.get('file_size','?')}): {r.get('error','?')}")

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults: {output_path}")
    if n_corrupted > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
