#!/usr/bin/env python3
"""Validate .pt files from the last stuck batch (all four ranks) for corruption.

Last collate entries before NCCL timeout (slurm-9027755.err ~21:06:23):
  R0: 4eyc_B, 2z5b_B, 6kfn_A, 4zjm_A
  R1: 1fl7_B, 5j0f_A, 4d50_B, 3ch4_B
  R2: 3g7n_B, 1vkf_C, 6j0k_B, 6cfz_J
  R3: 4noo_B, 6hd8_B, 2zhx_L, 5ywr_B
"""
import os
import sys
from pathlib import Path

import torch

STUCK_BATCH_IDS = [
    "4eyc_B", "2z5b_B", "6kfn_A", "4zjm_A",  # R0
    "1fl7_B", "5j0f_A", "4d50_B", "3ch4_B",  # R1
    "3g7n_B", "1vkf_C", "6j0k_B", "6cfz_J",  # R2
    "4noo_B", "6hd8_B", "2zhx_L", "5ywr_B",  # R3
]


def validate_one(processed_dir: Path, pid: str) -> dict:
    path = processed_dir / f"{pid}.pt"
    out = {"id": pid, "path": str(path), "exists": path.exists(), "loadable": False, "error": ""}
    if not path.exists():
        out["error"] = "file not found"
        return out
    try:
        out["size"] = path.stat().st_size
    except Exception as e:
        out["error"] = f"stat: {e}"
        return out
    try:
        graph = torch.load(path, weights_only=False, map_location="cpu")
        out["loadable"] = True
        out["has_coords"] = hasattr(graph, "coords") and graph.coords is not None
        if out["has_coords"]:
            out["coords_shape"] = list(graph.coords.shape)
        out["has_contact_map"] = hasattr(graph, "contact_map") and graph.contact_map is not None
        if hasattr(graph, "contact_map_confind"):
            out["has_contact_map_confind"] = graph.contact_map_confind is not None
        return out
    except Exception as e:
        out["error"] = repr(e)[:300]
        return out


def main():
    data_path = os.environ.get("DATA_PATH", "")
    if not data_path:
        data_path = os.environ.get("HOME", "/tmp")
        processed_dir = Path(data_path) / "pdb_train" / "processed"
    else:
        processed_dir = Path(data_path) / "pdb_train" / "processed"

    if len(sys.argv) > 1:
        processed_dir = Path(sys.argv[1])

    if not processed_dir.is_dir():
        print(f"ERROR: processed dir not found: {processed_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Validating {len(STUCK_BATCH_IDS)} .pt files from last stuck batch")
    print(f"Dir: {processed_dir}\n")

    results = [validate_one(processed_dir, pid) for pid in STUCK_BATCH_IDS]
    corrupted = [r for r in results if not r.get("loadable", True)]
    missing = [r for r in results if not r.get("exists", False)]

    for r in results:
        status = "OK" if r.get("loadable") else ("MISSING" if not r.get("exists") else "CORRUPT")
        extra = ""
        if r.get("coords_shape"):
            extra = f" coords={r['coords_shape']}"
        if r.get("error"):
            extra = f" error={r['error'][:80]}"
        print(f"  {r['id']}: {status}{extra}")

    print()
    if missing:
        print(f"MISSING ({len(missing)}): {[r['id'] for r in missing]}")
    if corrupted and not missing:
        print(f"CORRUPTED ({len(corrupted)}):")
        for r in corrupted:
            print(f"  {r['id']}: {r.get('error', '')}")
    if not corrupted and not missing:
        print("All 16 files load successfully. No corruption detected in these .pt files.")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
