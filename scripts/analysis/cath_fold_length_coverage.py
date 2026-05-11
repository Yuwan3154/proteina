"""Compute per-CAT-fold length coverage statistics.

For each CAT-level fold (the C.A.T parent of a CATH C.A.T.H code), aggregate the
lengths of every chain that carries that fold. Outputs a CSV with min/max/range/
median/IQR/count per fold and a histogram of (max_len - min_len) over folds.

This answers the user's specific question: "if I plot a histogram of (max length
of CAT fold X - min length of CAT fold X) for all folds, what does it look like?"

Inputs:
    --df_csv     The df_pdb_<...>.csv used by PDBDataSelector (one row per chain;
                 columns include `id`, `processed_length` or `length`, `split`).
    --cath_root  Directory containing pdb_chain_cath_uniprot.tsv.gz and
                 cath-b-newest-all.gz (defaults to data/cathdata/). Loaded via
                 CATHLabelTransform.
    --split      Optional split filter ("train"/"val"/"test"); if blank, use all rows.
    --out_dir    Where to write CSV + PNGs.

Usage:
    python scripts/analysis/cath_fold_length_coverage.py \
        --df_csv data/pdb_train/df_pdb_f1_minl50_maxl256_<...>.csv \
        --cath_root data/cathdata \
        --out_dir analysis_out/cath_length_coverage
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

_THIS = pathlib.Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

from proteinfoundation.utils.ff_utils.pdb_utils import extract_cath_code_by_level  # noqa: E402


def _load_pdbchain_to_cathids(sifts_gz: pathlib.Path) -> Dict[str, List[str]]:
    """Parallel of CATHLabelTransform._parse_cath_id, no Lightning deps."""
    out: Dict[str, List[str]] = defaultdict(list)
    with gzip.open(sifts_gz, "rt") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pdb, chain, _uniprot, cath_id = parts[:4]
            out[f"{pdb}_{chain}"].append(cath_id)
    return out


def _load_cathid_to_cathcode(cath_b_gz: pathlib.Path) -> Dict[str, str]:
    """Parallel of CATHLabelTransform._parse_cath_code (codes only — segments unused)."""
    out: Dict[str, str] = {}
    with gzip.open(cath_b_gz, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cath_id, _ver, cath_code = parts[0], parts[1], parts[2]
            out[cath_id] = cath_code
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_csv", type=pathlib.Path, required=True)
    ap.add_argument(
        "--cath_root", type=pathlib.Path,
        default=_PROTEINA_ROOT / "data" / "cathdata",
    )
    ap.add_argument("--split", type=str, default="")
    ap.add_argument(
        "--out_dir", type=pathlib.Path,
        default=pathlib.Path("analysis_out/cath_length_coverage"),
    )
    ap.add_argument(
        "--length_col", type=str, default="processed_length",
        help="Length column to use (falls back to 'length' if missing).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cath_length_coverage] loading {args.df_csv}")
    df = pd.read_csv(args.df_csv)
    if args.split:
        df = df[df["split"].astype(str) == args.split]
    length_col = args.length_col if args.length_col in df.columns else "length"
    df = df[["id", length_col]].dropna()
    df[length_col] = df[length_col].astype(int)
    print(f"[cath_length_coverage] {len(df)} chains (split={args.split or 'ALL'})")

    sifts_path = args.cath_root / "pdb_chain_cath_uniprot.tsv.gz"
    cathb_path = args.cath_root / "cath-b-newest-all.gz"
    if not sifts_path.exists() or not cathb_path.exists():
        print(f"ERROR: expected {sifts_path} and {cathb_path}", file=sys.stderr)
        sys.exit(1)
    print("[cath_length_coverage] loading SIFTS map ...")
    pdbchain_to_cathids = _load_pdbchain_to_cathids(sifts_path)
    print("[cath_length_coverage] loading CATH-B codes ...")
    cathid_to_cathcode = _load_cathid_to_cathcode(cathb_path)

    fold_to_lengths: Dict[str, List[int]] = defaultdict(list)
    n_no_cath = 0
    n_with_cath = 0
    for _, row in df.iterrows():
        cath_ids = pdbchain_to_cathids.get(row["id"])
        if not cath_ids:
            n_no_cath += 1
            continue
        seen_folds = set()  # one chain contributes once per fold even if multi-domain duplicates
        for cid in cath_ids:
            code = cathid_to_cathcode.get(cid)
            if not code:
                continue
            cat_fold = extract_cath_code_by_level(code, "T")
            seen_folds.add(cat_fold)
        if seen_folds:
            n_with_cath += 1
            for fold in seen_folds:
                fold_to_lengths[fold].append(int(row[length_col]))

    print(
        f"[cath_length_coverage] chains_with_cath={n_with_cath}, no_cath={n_no_cath}, "
        f"unique_CAT_folds={len(fold_to_lengths)}"
    )

    rows = []
    for fold, lens in fold_to_lengths.items():
        arr = np.asarray(lens)
        rows.append({
            "cat_fold": fold,
            "count": int(arr.size),
            "min_len": int(arr.min()),
            "max_len": int(arr.max()),
            "range": int(arr.max() - arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        })
    stats_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    csv_path = args.out_dir / "cath_cat_length_stats.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"[cath_length_coverage] wrote {csv_path}")

    # Plots — defer matplotlib import so the script is usable headless without it.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Histogram of (max - min) per fold.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(stats_df["range"].values, bins=60, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("max_len - min_len within a CAT fold (residues)")
    ax.set_ylabel("Number of CAT folds")
    ax.set_title(f"Length-range per CAT fold (n_folds={len(stats_df)})")
    fig.tight_layout()
    fig.savefig(args.out_dir / "hist_length_range_per_fold.png", dpi=130)
    plt.close(fig)

    # 2) Same, restricted to folds with at least 5 chains (so range isn't trivially small from undersampling).
    well_sampled = stats_df[stats_df["count"] >= 5]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(well_sampled["range"].values, bins=60, color="darkorange", edgecolor="black", alpha=0.85)
    ax.set_xlabel("max_len - min_len within a CAT fold (residues), folds with >=5 chains")
    ax.set_ylabel("Number of CAT folds")
    ax.set_title(f"Length-range per CAT fold, count>=5 (n_folds={len(well_sampled)})")
    fig.tight_layout()
    fig.savefig(args.out_dir / "hist_length_range_per_fold_count5plus.png", dpi=130)
    plt.close(fig)

    # 3) Scatter of count vs range (does range grow with sample size?).
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(stats_df["count"], stats_df["range"], s=8, alpha=0.4, color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("count (chains in fold, log scale)")
    ax.set_ylabel("max_len - min_len (residues)")
    ax.set_title("Per-CAT-fold length-range vs sample size")
    fig.tight_layout()
    fig.savefig(args.out_dir / "scatter_count_vs_range.png", dpi=130)
    plt.close(fig)

    # 4) Top-20 by count: min/median/max bars.
    top = stats_df.head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(top))
    ax.hlines(y, top["min_len"], top["max_len"], color="steelblue", lw=4)
    ax.scatter(top["median"], y, color="firebrick", zorder=5, label="median")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(top["cat_fold"], top["count"])])
    ax.invert_yaxis()
    ax.set_xlabel("Length (residues)")
    ax.set_title("Top-20 CAT folds by sample count: length spread")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "top20_folds_length_spread.png", dpi=130)
    plt.close(fig)

    summary = stats_df["range"].describe()
    print("\n[cath_length_coverage] summary of per-fold length-range:")
    print(summary.to_string())
    summary.to_csv(args.out_dir / "range_summary.csv")
    print(f"\n[cath_length_coverage] all outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
