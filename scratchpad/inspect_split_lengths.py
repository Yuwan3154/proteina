"""Read-only: which length columns the train split dataframe carries for --config_name, and how they compare with the
served graph (residue axis before PaddingTransform, and residues with coordinates) for a sample of train chains.
Usage (PYTHONPATH=repo): python scratchpad/inspect_split_lengths.py --config_name C [--n 300]
"""

import argparse
import random

import pandas as pd

from scratchpad.test_plan_datamodule_integration import compose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    _, dm = compose(args.config_name, plan_on=False, num_workers=0)
    dm.setup("fit")  # no prepare_data: it may rewrite the shared CSV
    print("train split columns:", list(dm.dfs_splits["train"].columns), "rows", len(dm.dfs_splits["train"]))
    df = dm.df_data
    print("df_data columns:", list(df.columns), "rows", len(df))
    assert df["id"].is_unique
    ds = dm.train_ds
    ds.transform = None
    by_id = df.set_index(df["id"].astype(str))
    rng = random.Random(0)
    idx = rng.sample(range(len(ds)), min(args.n, len(ds)))
    stats = {"n": 0, "n_len": 0, "n_proc": 0, "axis_eq_length": 0, "coords_eq_length": 0, "axis_eq_coords": 0,
             "axis_eq_processed": 0, "coords_eq_processed": 0}
    has_proc = "processed_length" in df.columns
    has_len = "length" in df.columns
    examples = []
    for i in idx:
        g = ds[i]
        stem = str(g.id)
        axis = int(g.coord_mask.shape[0])
        coords = int(g.coord_mask.any(-1).sum())
        row = by_id.loc[stem]
        length = int(row["length"]) if has_len else None
        proc = int(row["processed_length"]) if has_proc and not pd.isna(row["processed_length"]) else None
        stats["n"] += 1
        stats["axis_eq_coords"] += axis == coords
        if has_len:
            stats["n_len"] += 1
            stats["axis_eq_length"] += axis == length
            stats["coords_eq_length"] += coords == length
        if proc is not None:
            stats["n_proc"] += 1
            stats["axis_eq_processed"] += axis == proc
            stats["coords_eq_processed"] += coords == proc
        if len(examples) < 10 and (axis != coords or (has_len and axis != length)):
            examples.append((stem, axis, coords, length, proc))
    print("of", stats["n"], "sampled train chains:", {k: v for k, v in stats.items() if k != "n"})
    print("examples (stem, residue axis, residues with coords, df length, df processed_length):")
    for e in examples:
        print("  ", e)


if __name__ == "__main__":
    main()
