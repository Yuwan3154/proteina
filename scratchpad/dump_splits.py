"""Dump the train / val / test chain-ID lists the datamodule actually builds.

Needed because no split list exists on disk: `all_stems.txt` mixes entry and chain IDs and is not
the split, and PDBDataSplitter derives it at runtime from the 25% sequence-identity cluster TSV
plus the train_val_test ratios. The foldseek val-vs-train search needs the training list either to
restrict the target DB or to filter hits, so it is required either way -- and the val list it
produces is authoritative rather than reconstructed.

Runs the REAL datamodule setup rather than reimplementing the split, so the lists cannot drift
from what training and validation actually saw.
"""

import argparse
import os
import sys

import hydra

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset config name, without .yaml")
    ap.add_argument("--subdir", default="pdb")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with hydra.initialize(f"../configs/datasets_config/{args.subdir}", version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=args.dataset)

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    dm.setup()

    # PDBDataSplitter stores the finished splits in `dfs_splits` (pdb_data.py:377, 645-650), each
    # a DataFrame with an `id` column of chain IDs. Read it off the splitter the datamodule
    # actually used rather than re-deriving it.
    splitter = dm.datasplitter
    print(f"[splits] {[(k, len(v)) for k, v in splitter.dfs_splits.items()]}", flush=True)
    for split in ("train", "val", "test"):
        df = splitter.dfs_splits[split]
        ids = list(df["id"])
        path = os.path.join(args.out_dir, f"{split}_chain_ids.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(str(i) for i in ids) + "\n")
        print(f"[{split}] n={len(ids)} -> {path}   first: {ids[:3]}", flush=True)


if __name__ == "__main__":
    main()
