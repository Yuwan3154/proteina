"""Gate for PDBDataSplitter(split_dir=...) (T8 frozen max512 split). Toy data dir; run: python scratchpad/test_frozen_split.py

1. The frozen lists come back exactly, with per-split cluster mappings restricted to each split's chains.
2. A listed chain missing from the csv, a chain in no cluster, and overlapping split files each fail loudly.
3. Without split_dir the splitter still draws (the old path is untouched).
"""

import contextlib
import os
import tempfile

import pandas as pd

from proteinfoundation.datasets.pdb_data import PDBDataSplitter
from proteinfoundation.utils.cluster_utils import setup_clustering_file_paths

IDENT = "toy_ident"


def raises(exc, fn):
    done = False
    with contextlib.suppress(exc):
        fn()
        done = True
    return not done


def write(d, rows, clus, splits):
    df = pd.DataFrame([[c, c[:-2], "A", 50, "A" * 50] for c in rows], columns=["id", "pdb", "chain", "length", "sequence"])
    inp, fasta, tsv = setup_clustering_file_paths(d, IDENT, 0.25)
    with open(tsv, "w") as f:
        f.write("".join(f"{r}\t{m}\n" for m, r in clus.items()))
    open(fasta, "w").write("".join(f">{r}\n{'A' * 50}\n" for r in sorted(set(clus.values()))))
    open(inp, "w").write("")
    sd = os.path.join(d, "split")
    os.makedirs(sd, exist_ok=True)
    for s, ids in splits.items():
        open(os.path.join(sd, f"{s}_chain_ids.txt"), "w").write("".join(f"{c}\n" for c in ids))
    return df, sd


def splitter(d, sd):
    return PDBDataSplitter(data_dir=d, train_val_test=[0.98, 0.019, 0.001], split_type="sequence_similarity",
                           split_sequence_similarity=0.25, split_dir=sd)


def main():
    with tempfile.TemporaryDirectory() as d:
        rows = ["a_A", "a_B", "b_A", "c_A", "d_A", "e_A"]
        clus = {"a_A": "a_A", "a_B": "a_A", "b_A": "b_A", "c_A": "c_A", "d_A": "c_A", "e_A": "e_A"}
        df, sd = write(d, rows, clus, {"train": ["a_A", "a_B", "b_A"], "val": ["c_A", "d_A"], "test": ["e_A"]})
        dfs, maps = splitter(d, sd).split_data(df, IDENT)
        assert sorted(dfs["train"]["id"]) == ["a_A", "a_B", "b_A"] and sorted(dfs["val"]["id"]) == ["c_A", "d_A"]
        assert maps["train"] == {"a_A": ["a_A", "a_B"], "b_A": ["b_A"]} and maps["val"] == {"c_A": ["c_A", "d_A"]}
        assert maps["test"] == {"e_A": ["e_A"]}
        print("PASS frozen lists returned exactly; cluster mappings restricted per split")

        df2, sd2 = write(d, rows, clus, {"train": ["a_A", "zz_A"], "val": ["c_A"], "test": ["e_A"]})
        assert raises(AssertionError, lambda: splitter(d, sd2).split_data(df2, IDENT)), "a chain missing from the csv must fail"
        df3, sd3 = write(d, rows + ["f_A"], clus, {"train": ["a_A", "f_A"], "val": ["c_A"], "test": ["e_A"]})
        assert raises(AssertionError, lambda: splitter(d, sd3).split_data(df3, IDENT)), "a chain in no cluster must fail"
        df4, sd4 = write(d, rows, clus, {"train": ["a_A", "c_A"], "val": ["c_A"], "test": ["e_A"]})
        assert raises(AssertionError, lambda: splitter(d, sd4).split_data(df4, IDENT)), "overlapping splits must fail"
        print("PASS missing chain / unclustered chain / overlap each fail loudly")

        sp = PDBDataSplitter(data_dir=d, train_val_test=[0.5, 0.25, 0.25], split_type="sequence_similarity",
                             split_sequence_similarity=0.25)
        dfs, _ = sp.split_data(df, IDENT)
        assert sum(len(v) for v in dfs.values()) == len(rows)
        print("PASS without split_dir the splitter still draws a split over every chain")
    print("ALL PASS")


if __name__ == "__main__":
    main()
