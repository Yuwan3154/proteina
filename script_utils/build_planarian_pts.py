#!/usr/bin/env python3
"""Pre-build the per-sequence .pt files the seq_cath prediction run needs.

The orchestrator only builds PT files in --input mode or when --cif_dir is set;
a no-ground-truth --dataset_file run builds none. This script fills that gap by
calling the pipeline's own PT builder, so no PT-construction logic is duplicated.

Refuses to run if an id already has a .pt in processed/ whose stored sequence
differs from ours (create_pt_files silently skips existing files, which would
otherwise let a stale/foreign file be used as our input).

Usage (CPU node):
    python script_utils/build_planarian_pts.py --manifest ~/planarian_lig/planarian_extracellular.csv
"""

import argparse
import os

import pandas as pd
import torch

from proteinfoundation.prediction_pipeline.input_parser import create_pt_files
from proteinfoundation.openfold_stub.np.residue_constants import restype_order, unk_restype_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with id,sequence columns")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)[["id", "sequence"]]
    data_path = os.environ.get("DATA_PATH", os.path.join(os.getcwd(), "data"))
    processed = os.path.join(data_path, "pdb_train", "processed")

    conflicts = []
    for pid, seq in zip(df["id"], df["sequence"]):
        path = os.path.join(processed, f"{pid}.pt")
        if not os.path.exists(path):
            continue
        pt = torch.load(path, weights_only=False, map_location="cpu")
        want = [restype_order.get(a, unk_restype_index) for a in seq]
        if list(pt.residue_type.tolist()) != want:
            conflicts.append(pid)
    if conflicts:
        raise SystemExit(
            f"{len(conflicts)} id(s) already have a .pt with a DIFFERENT sequence in "
            f"{processed}; refusing to proceed. First few: {conflicts[:10]}"
        )

    create_pt_files(df)

    missing = [pid for pid in df["id"] if not os.path.exists(os.path.join(processed, f"{pid}.pt"))]
    if missing:
        raise SystemExit(f"{len(missing)} PT files missing after build: {missing[:10]}")
    print(f"OK: {len(df)} PT files present in {processed}")


if __name__ == "__main__":
    main()
