#!/usr/bin/env python3
"""Augment an evaluation-data-list CSV with a `cath_code` column derived from a
foldseek_cath_batch summary TSV.

Inputs:
  --eval-csv         the per-chain eval CSV (column with chain IDs like "6UF2_A")
  --eval-col         column name in eval-csv that carries the chain ID (default: pdb)
  --foldseek-summary cath50_summary.tsv produced by scripts/foldseek_cath_batch.py
  --out              destination CSV path

Output:
  Same rows as --eval-csv. Appends columns:
    cath_code           4-level CATH code "C.A.T.x" (padded H). "x.x.x.x" when no foldseek hit.
    cath_top1_tm        the foldseek top-1 TM-score (string; empty when no hit)
    cath_top1_target    the foldseek top-1 target ID (e.g. "4fprA00")
"""

import argparse
import pathlib
import sys

import pandas as pd


NULL_CATH = "x.x.x.x"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--eval-col", default="pdb")
    p.add_argument("--foldseek-summary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--null", default=NULL_CATH, help="value for chains without a CATH hit")
    return p.parse_args()


def normalize_cath(raw):
    if raw is None:
        return NULL_CATH
    code = str(raw).strip()
    if not code or code.lower() == "nan":
        return NULL_CATH
    parts = code.split(".")
    if len(parts) == 3:
        parts.append("x")
    elif len(parts) == 4:
        pass
    else:
        # malformed; treat as null
        return NULL_CATH
    return ".".join(parts)


def main():
    args = parse_args()
    eval_path = pathlib.Path(args.eval_csv)
    fs_path = pathlib.Path(args.foldseek_summary)
    out_path = pathlib.Path(args.out)

    if not eval_path.is_file():
        sys.exit(f"--eval-csv not found: {eval_path}")
    if not fs_path.is_file():
        sys.exit(f"--foldseek-summary not found: {fs_path}")

    eval_df = pd.read_csv(eval_path)
    if args.eval_col not in eval_df.columns:
        sys.exit(f"Column {args.eval_col!r} not in eval CSV. Have: {list(eval_df.columns)}")
    fs_df = pd.read_csv(fs_path, sep="\t")
    for need in ("chain_id", "top1_cat", "top1_tm", "top1_target"):
        if need not in fs_df.columns:
            sys.exit(f"Column {need!r} not in foldseek summary. Have: {list(fs_df.columns)}")

    fs_slim = fs_df[["chain_id", "top1_cat", "top1_tm", "top1_target"]].copy()
    fs_slim = fs_slim.rename(columns={"chain_id": args.eval_col})

    merged = eval_df.merge(fs_slim, how="left", on=args.eval_col)
    merged["cath_code"] = merged["top1_cat"].apply(normalize_cath)
    merged["cath_top1_tm"] = merged["top1_tm"].fillna("").astype(str)
    merged["cath_top1_target"] = merged["top1_target"].fillna("").astype(str)
    merged = merged.drop(columns=["top1_cat", "top1_tm", "top1_target"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    null_n = int((merged["cath_code"] == NULL_CATH).sum())
    print(f"Wrote {out_path}  rows={len(merged)}  cath_code_null={null_n}")
    print(f"Preview:")
    print(merged[[args.eval_col, "cath_code", "cath_top1_tm", "cath_top1_target"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
