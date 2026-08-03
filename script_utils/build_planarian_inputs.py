#!/usr/bin/env python3
"""Build the planarian ligand-search dataset CSVs for the seq_cath prediction pipeline.

Reads the two DeepLoc FASTAs, sorts sequences ascending by length (so the
in-distribution short proteins are sampled first), splits them into
length-ordered batches, and emits one dataset CSV per (CATH code x batch).

Batch size is capped so every downstream per-run directory
(inference/<cfg>/seq_cath_cond/, best_templates/, best_predictions/) stays
under the 1024-files-per-directory limit.

Usage:
    python script_utils/build_planarian_inputs.py \
        --full_fasta extracellular.fa \
        --subset_fasta extracellular_no_besthit.fa \
        --out_dir ~/planarian_lig \
        --n_batches 4
"""

import argparse
import os

import pandas as pd

CATH_TAGS = {"2.60.40": "c2p60p40", "2.80.10": "c2p80p10"}


def read_fasta(path):
    ids, seqs, buf, cur = [], [], [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur is not None:
                    seqs.append("".join(buf))
                cur = line[1:].split()[0]
                ids.append(cur)
                buf = []
            else:
                buf.append(line)
    if cur is not None:
        seqs.append("".join(buf))
    return ids, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_fasta", required=True)
    ap.add_argument("--subset_fasta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_batches", type=int, default=4)
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ids, seqs = read_fasta(args.full_fasta)
    sub_ids, _ = read_fasta(args.subset_fasta)
    sub = set(sub_ids)

    assert len(ids) == len(set(ids)), "duplicate IDs in full fasta"
    assert sub <= set(ids), "no_besthit fasta is not a subset of the full fasta"

    df = pd.DataFrame({"id": ids, "sequence": seqs})
    df["length"] = df["sequence"].str.len()
    df["in_no_besthit_subset"] = df["id"].isin(sub)
    df = df.sort_values(["length", "id"], kind="mergesort").reset_index(drop=True)
    df["length_rank"] = range(len(df))

    manifest_path = os.path.join(out_dir, "planarian_extracellular.csv")
    df.to_csv(manifest_path, index=False)
    print(f"manifest: {manifest_path}  n={len(df)}  "
          f"subset={int(df.in_no_besthit_subset.sum())}  "
          f"len {df.length.min()}-{df.length.max()}")

    n = len(df)
    per = -(-n // args.n_batches)  # ceil
    assert per <= 1000, f"batch size {per} would break the 1024-files-per-dir limit"

    rows = []
    for b in range(args.n_batches):
        chunk = df.iloc[b * per:(b + 1) * per]
        if chunk.empty:
            continue
        for code, tag in CATH_TAGS.items():
            out = chunk[["id", "sequence", "length"]].copy()
            out["cath_code"] = code
            path = os.path.join(out_dir, f"planarian_{tag}_b{b}.csv")
            out.to_csv(path, index=False)
            rows.append({"batch": b, "cath_code": code, "tag": tag, "csv": path,
                         "n": len(out), "len_min": int(out.length.min()),
                         "len_max": int(out.length.max())})
            print(f"  {os.path.basename(path)}: n={len(out)} "
                  f"L {out.length.min()}-{out.length.max()}")

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "batch_manifest.csv"), index=False)


if __name__ == "__main__":
    main()
