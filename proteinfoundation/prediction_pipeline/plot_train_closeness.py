#!/usr/bin/env python
"""Plot top-1 template quality against each native's closeness to the training set.

Joins a run's ``cross_protein_summary_data.csv`` (which now carries
``top_1_tm_ref_template`` and ``top_1_rmsd_ref_template`` — see Part 0) with the
``train_closeness.csv`` produced by ``train_closeness_search.py`` and makes two
scatter plots:

  * TM   : x = train_closest_tm   (native vs closest training chain)
           y = top_1_tm_ref_template (top-1 AF2Rank template vs native)
  * RMSD : x = train_closest_rmsd
           y = top_1_rmsd_ref_template

Points are coloured by ``in_train`` (blue=train, red=not) and sized by ``length``,
matching the existing cross-protein scatter style. A target sitting at high
train-closeness AND high template quality is a memorization candidate; genuine
generalization shows good template quality at LOW train-closeness.

Light deps only (pandas / numpy / matplotlib); runs in any analysis env.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_THIS = Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

from scripts.analysis.mark_in_train import normalize_pdb_chain  # noqa: E402

log = logging.getLogger("plot_train_closeness")


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x[m], y[m]).correlation)
    except Exception:  # noqa: BLE001 - scipy optional; fall back to rank-Pearson
        xr = pd.Series(x[m]).rank().to_numpy()
        yr = pd.Series(y[m]).rank().to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])


def _scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path,
             xlabel: str, ylabel: str, title: str, diag: bool, unit_box: bool) -> None:
    valid = df.dropna(subset=[x_col, y_col]).copy()
    if len(valid) == 0:
        log.warning("No finite (%s, %s) pairs; skipping %s", x_col, y_col, out_path.name)
        return
    colors = _as_bool(valid["in_train"]).map({True: "tab:blue", False: "tab:red"}) \
        if "in_train" in valid.columns else "tab:gray"
    sizes = (valid["length"] / 1.5) if "length" in valid.columns else 30.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(valid[x_col], valid[y_col], c=colors, s=sizes, alpha=0.4, edgecolors="none")

    if diag:
        if unit_box:
            lo, hi = 0.0, 1.0
        else:
            lo = float(min(valid[x_col].min(), valid[y_col].min()))
            hi = float(max(valid[x_col].max(), valid[y_col].max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.5, label="y = x")
    if unit_box:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    rho = _spearman(valid[x_col].to_numpy(float), valid[y_col].to_numpy(float))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\nSpearman rho = {rho:.3f}  (n = {len(valid)})", fontsize=12)
    ax.grid(True, alpha=0.3)

    handles = []
    if "in_train" in valid.columns:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="in train", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", label="not in train", markersize=8),
        ]
    if diag:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="k", linestyle="--", label="y = x"))
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=900, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s (n=%d, rho=%.3f)", out_path, len(valid), rho)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary_csv", required=True, help="cross_protein_summary_data.csv for the run.")
    ap.add_argument("--train_closeness_csv", required=True, help="train_closeness.csv from train_closeness_search.py.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--label", default="", help="Title prefix (e.g. dataset name).")
    ap.add_argument("--tm_y_col", default="top_1_tm_ref_template",
                    help="TM y-axis column (e.g. top_1_tm_ref_template / min_top_1_tm_ref_template).")
    ap.add_argument("--rmsd_y_col", default="top_1_rmsd_ref_template",
                    help="RMSD y-axis column (e.g. top_1_rmsd_ref_template / min_top_1_rmsd_ref_template).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s", stream=sys.stderr)

    summ = pd.read_csv(args.summary_csv)
    close = pd.read_csv(args.train_closeness_csv)
    summ["_key"] = summ["protein_id"].astype(str).map(normalize_pdb_chain)
    close["_key"] = close["protein_id"].astype(str).map(normalize_pdb_chain)
    merged = summ.merge(close.drop(columns=["protein_id"]), on="_key", how="left").drop(columns=["_key"])

    n_close = int(merged["train_closest_tm"].notna().sum())
    log.info("Merged %d rows; %d have train-closeness", len(merged), n_close)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "train_closeness_merged.csv", index=False)

    prefix = f"{args.label}: " if args.label else ""
    if args.tm_y_col in merged.columns:
        _scatter(
            merged, "train_closest_tm", args.tm_y_col, out_dir / "train_closeness_vs_top1_tm.png",
            xlabel="Native: TM-score to closest training chain",
            ylabel=f"Top-1 template TM-score vs native ({args.tm_y_col})",
            title=f"{prefix}Template quality vs training-set proximity (TM)",
            diag=True, unit_box=True,
        )
    else:
        log.warning("Column %s absent in summary; skipping TM plot", args.tm_y_col)

    if args.rmsd_y_col in merged.columns:
        _scatter(
            merged, "train_closest_rmsd", args.rmsd_y_col, out_dir / "train_closeness_vs_top1_rmsd.png",
            xlabel="Native: RMSD (A) to closest training chain",
            ylabel=f"Top-1 template RMSD (A) vs native ({args.rmsd_y_col})",
            title=f"{prefix}Template quality vs training-set proximity (RMSD)",
            diag=False, unit_box=False,
        )
    else:
        log.warning("Column %s absent in summary (re-run with Part 0); skipping RMSD plot", args.rmsd_y_col)


if __name__ == "__main__":
    main()
