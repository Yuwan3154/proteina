"""
Compare top-1 template (by min pTM across AF2Rank m1/m2) and the corresponding
AF2Rank refined prediction between two Proteina model replicas.

For each target present in both replica inference directories:
- Read per-target AF2Rank m1 / m2 score CSVs, inner-join on structure_file.
- Parse Proteina sample index from structure_file stem (last "_"-delimited token).
- Filter to sample_index < max_sample_index (default 512) to keep the comparison
  apples-to-apples when one replica has more samples / different top_k.
- Pick row with the highest min(ptm_m1, ptm_m2).
- "Corresponding prediction" = the AF2Rank variant (m1 or m2) with the higher
  pTM for that template, matching run_prediction_pipeline.step_collect_results.
- Run USalign -TMscore 5 -outfmt 2 on (a_template, b_template) and
  (a_prediction, b_prediction); record TM1/TM2/RMSD/Lali/L1/L2.
- Emit per-target CSV and an aggregate JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proteinfoundation.af2rank_evaluation.usalign_tabular import iter_usalign_outfmt2_rows

logger = logging.getLogger(__name__)

AF2RANK_TOPK_SUBDIR = "af2rank_on_proteinebm_top_k"
M1_SUBDIR = "af2rank_analysis"
M2_SUBDIR = "af2rank_analysis_model_2_ptm"
PREDICTED_SUBDIR = "predicted_structures"


def _scores_csv_path(inference_dir: Path, protein_id: str, model_subdir: str) -> Path:
    return inference_dir / protein_id / AF2RANK_TOPK_SUBDIR / model_subdir / f"af2rank_scores_{protein_id}.csv"


def _prediction_pdb_path(inference_dir: Path, protein_id: str, model_subdir: str, predicted_file: str) -> Path:
    return inference_dir / protein_id / AF2RANK_TOPK_SUBDIR / model_subdir / PREDICTED_SUBDIR / predicted_file


def _parse_sample_index(structure_file: str) -> Optional[int]:
    stem = Path(str(structure_file)).stem
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1]
    if not tail.lstrip("-").isdigit():
        return None
    return int(tail)


def _load_merged_m1_m2(inference_dir: Path, protein_id: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return inner-joined per-template DataFrame or (None, error_status)."""
    m1_csv = _scores_csv_path(inference_dir, protein_id, M1_SUBDIR)
    m2_csv = _scores_csv_path(inference_dir, protein_id, M2_SUBDIR)
    if not m1_csv.exists():
        return None, "missing_csv_m1"
    if not m2_csv.exists():
        return None, "missing_csv_m2"
    df_m1 = pd.read_csv(m1_csv)
    df_m2 = pd.read_csv(m2_csv)
    required = {"structure_file", "ptm"}
    if not required.issubset(df_m1.columns) or not required.issubset(df_m2.columns):
        return None, "missing_columns"
    cols_m1 = ["structure_file", "ptm"]
    if "structure_path" in df_m1.columns:
        cols_m1.append("structure_path")
    if "predicted_structure_file" in df_m1.columns:
        cols_m1.append("predicted_structure_file")
    cols_m2 = ["structure_file", "ptm"]
    if "predicted_structure_file" in df_m2.columns:
        cols_m2.append("predicted_structure_file")
    df_m1 = df_m1[cols_m1].rename(columns={"ptm": "ptm_m1", "predicted_structure_file": "predicted_file_m1"})
    df_m2 = df_m2[cols_m2].rename(columns={"ptm": "ptm_m2", "predicted_structure_file": "predicted_file_m2"})
    merged = df_m1.merge(df_m2, on="structure_file", how="inner")
    if merged.empty:
        return None, "empty_merge"
    merged["sample_index"] = merged["structure_file"].apply(_parse_sample_index)
    merged = merged.dropna(subset=["sample_index", "ptm_m1", "ptm_m2"]).copy()
    if merged.empty:
        return None, "empty_after_index_parse"
    merged["sample_index"] = merged["sample_index"].astype(int)
    merged["min_ptm"] = merged[["ptm_m1", "ptm_m2"]].min(axis=1)
    merged["max_ptm"] = merged[["ptm_m1", "ptm_m2"]].max(axis=1)
    return merged, None


def _select_top1(
    merged: pd.DataFrame,
    inference_dir: Path,
    protein_id: str,
    max_sample_index: int,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    filtered = merged[merged["sample_index"] < max_sample_index]
    if filtered.empty:
        return None, "empty_after_filter"
    top = filtered.sort_values("min_ptm", ascending=False).iloc[0]
    structure_file = str(top["structure_file"])
    ptm_m1 = float(top["ptm_m1"])
    ptm_m2 = float(top["ptm_m2"])
    best_model = "m1" if ptm_m1 >= ptm_m2 else "m2"
    template_path = inference_dir / protein_id / structure_file
    pred_file_m1 = top.get("predicted_file_m1", structure_file) if isinstance(top.get("predicted_file_m1", None), str) else structure_file
    pred_file_m2 = top.get("predicted_file_m2", structure_file) if isinstance(top.get("predicted_file_m2", None), str) else structure_file
    pred_m1 = _prediction_pdb_path(inference_dir, protein_id, M1_SUBDIR, pred_file_m1)
    pred_m2 = _prediction_pdb_path(inference_dir, protein_id, M2_SUBDIR, pred_file_m2)

    if best_model == "m1":
        if pred_m1.exists():
            chosen_pred = pred_m1
        elif pred_m2.exists():
            chosen_pred = pred_m2
            best_model = "m2"
        else:
            return None, "missing_prediction_pdb"
    else:
        if pred_m2.exists():
            chosen_pred = pred_m2
        elif pred_m1.exists():
            chosen_pred = pred_m1
            best_model = "m1"
        else:
            return None, "missing_prediction_pdb"

    if not template_path.exists():
        return None, "missing_template_pdb"

    return (
        {
            "sample_index": int(top["sample_index"]),
            "structure_file": structure_file,
            "template_path": str(template_path),
            "ptm_m1": ptm_m1,
            "ptm_m2": ptm_m2,
            "min_ptm": float(top["min_ptm"]),
            "max_ptm": float(top["max_ptm"]),
            "best_model": best_model,
            "prediction_path": str(chosen_pred),
        },
        None,
    )


def _run_usalign_pair(pdb_a: str, pdb_b: str, tmscore_mode: int) -> Dict[str, float]:
    """Run USalign on a pair and return metrics dict with NaN on failure."""
    nan = {"TM1": float("nan"), "TM2": float("nan"), "RMSD": float("nan"), "L1": float("nan"), "L2": float("nan"), "Lali": float("nan")}
    cmd = ["USalign", pdb_a, pdb_b, "-TMscore", str(int(tmscore_mode)), "-outfmt", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("USalign failed (rc=%d) for %s vs %s: %s", proc.returncode, pdb_a, pdb_b, proc.stderr.strip()[:200])
        return nan
    rows = iter_usalign_outfmt2_rows(proc.stdout)
    if not rows:
        logger.warning("USalign produced no data rows for %s vs %s", pdb_a, pdb_b)
        return nan
    parts = rows[0]
    def _get(idx: int) -> float:
        if len(parts) <= idx:
            return float("nan")
        token = parts[idx].strip()
        if not token:
            return float("nan")
        return float(token)
    return {"TM1": _get(2), "TM2": _get(3), "RMSD": _get(4), "L1": _get(8), "L2": _get(9), "Lali": _get(10)}


def _empty_row(protein_id: str, status: str) -> Dict[str, object]:
    nan = float("nan")
    row = {
        "protein_id": protein_id,
        "a_sample_index": nan, "b_sample_index": nan, "same_sample_index": False,
        "a_template_file": "", "a_template_path": "", "b_template_file": "", "b_template_path": "",
        "a_m1_ptm": nan, "a_m2_ptm": nan, "a_min_ptm": nan, "a_max_ptm": nan, "a_best_model": "",
        "b_m1_ptm": nan, "b_m2_ptm": nan, "b_min_ptm": nan, "b_max_ptm": nan, "b_best_model": "",
        "a_prediction_path": "", "b_prediction_path": "",
        "tm_templates_TM1": nan, "tm_templates_TM2": nan, "rmsd_templates": nan,
        "lali_templates": nan, "l1_templates": nan, "l2_templates": nan,
        "tm_predictions_TM1": nan, "tm_predictions_TM2": nan, "rmsd_predictions": nan,
        "lali_predictions": nan, "l1_predictions": nan, "l2_predictions": nan,
        "passes_cutoff_a": False, "passes_cutoff_b": False, "passes_cutoff_either": False,
        "status": status,
    }
    return row


def compare_protein(
    protein_id: str,
    replica_a_dir: str,
    replica_b_dir: str,
    max_sample_index: int,
    tmscore_mode: int,
    ptm_cutoff: float,
) -> Dict[str, object]:
    a_dir = Path(replica_a_dir)
    b_dir = Path(replica_b_dir)

    merged_a, err_a = _load_merged_m1_m2(a_dir, protein_id)
    if merged_a is None:
        return _empty_row(protein_id, f"a_{err_a}")
    merged_b, err_b = _load_merged_m1_m2(b_dir, protein_id)
    if merged_b is None:
        return _empty_row(protein_id, f"b_{err_b}")

    top_a, err_a = _select_top1(merged_a, a_dir, protein_id, max_sample_index)
    if top_a is None:
        return _empty_row(protein_id, f"a_{err_a}")
    top_b, err_b = _select_top1(merged_b, b_dir, protein_id, max_sample_index)
    if top_b is None:
        return _empty_row(protein_id, f"b_{err_b}")

    tm_templates = _run_usalign_pair(top_a["template_path"], top_b["template_path"], tmscore_mode)
    tm_predictions = _run_usalign_pair(top_a["prediction_path"], top_b["prediction_path"], tmscore_mode)

    passes_a = bool(top_a["min_ptm"] >= ptm_cutoff)
    passes_b = bool(top_b["min_ptm"] >= ptm_cutoff)

    row = {
        "protein_id": protein_id,
        "a_sample_index": top_a["sample_index"],
        "b_sample_index": top_b["sample_index"],
        "same_sample_index": bool(top_a["sample_index"] == top_b["sample_index"]),
        "a_template_file": top_a["structure_file"],
        "a_template_path": top_a["template_path"],
        "b_template_file": top_b["structure_file"],
        "b_template_path": top_b["template_path"],
        "a_m1_ptm": top_a["ptm_m1"], "a_m2_ptm": top_a["ptm_m2"],
        "a_min_ptm": top_a["min_ptm"], "a_max_ptm": top_a["max_ptm"], "a_best_model": top_a["best_model"],
        "b_m1_ptm": top_b["ptm_m1"], "b_m2_ptm": top_b["ptm_m2"],
        "b_min_ptm": top_b["min_ptm"], "b_max_ptm": top_b["max_ptm"], "b_best_model": top_b["best_model"],
        "a_prediction_path": top_a["prediction_path"],
        "b_prediction_path": top_b["prediction_path"],
        "tm_templates_TM1": tm_templates["TM1"], "tm_templates_TM2": tm_templates["TM2"],
        "rmsd_templates": tm_templates["RMSD"], "lali_templates": tm_templates["Lali"],
        "l1_templates": tm_templates["L1"], "l2_templates": tm_templates["L2"],
        "tm_predictions_TM1": tm_predictions["TM1"], "tm_predictions_TM2": tm_predictions["TM2"],
        "rmsd_predictions": tm_predictions["RMSD"], "lali_predictions": tm_predictions["Lali"],
        "l1_predictions": tm_predictions["L1"], "l2_predictions": tm_predictions["L2"],
        "passes_cutoff_a": passes_a,
        "passes_cutoff_b": passes_b,
        "passes_cutoff_either": bool(passes_a or passes_b),
        "status": "ok",
    }
    return row


def _load_protein_ids_from_input(input_path: str, id_col: str) -> List[str]:
    df = pd.read_csv(input_path)
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not in {input_path}; columns present: {list(df.columns)}")
    ids = [str(x).strip() for x in df[id_col].dropna().tolist() if str(x).strip()]
    return sorted(set(ids))


def _finite_series(values: List[float]) -> np.ndarray:
    arr = np.array([v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    return arr


def _aggregate_stats(df: pd.DataFrame, args: argparse.Namespace, passing_only: bool) -> Dict[str, object]:
    ok_all = df[df["status"] == "ok"].copy()
    skipped_counts = df[df["status"] != "ok"]["status"].value_counts().to_dict()
    num_pass_a = int(ok_all["passes_cutoff_a"].sum()) if len(ok_all) else 0
    num_pass_b = int(ok_all["passes_cutoff_b"].sum()) if len(ok_all) else 0
    num_pass_either = int(ok_all["passes_cutoff_either"].sum()) if len(ok_all) else 0
    num_pass_both = int((ok_all["passes_cutoff_a"] & ok_all["passes_cutoff_b"]).sum()) if len(ok_all) else 0

    ok = ok_all[ok_all["passes_cutoff_either"]].copy() if passing_only else ok_all

    summary: Dict[str, object] = {
        "replica_a_dir": args.replica_a_dir,
        "replica_b_dir": args.replica_b_dir,
        "replica_a_label": args.replica_a_label,
        "replica_b_label": args.replica_b_label,
        "max_sample_index": args.max_sample_index,
        "tmscore_mode": args.tmscore_mode,
        "ptm_cutoff": args.ptm_cutoff,
        "filter_passing_either": bool(passing_only),
        "num_targets_total": int(len(df)),
        "num_targets_ok": int(len(ok_all)),
        "num_targets_compared": int(len(ok)),
        "num_skipped": int(len(df) - len(ok_all)),
        "num_passes_cutoff_a": num_pass_a,
        "num_passes_cutoff_b": num_pass_b,
        "num_passes_cutoff_either": num_pass_either,
        "num_passes_cutoff_both": num_pass_both,
        "skipped_status_counts": {k: int(v) for k, v in skipped_counts.items()},
    }
    if ok.empty:
        return summary

    def _stats(col: str) -> Dict[str, float]:
        arr = _finite_series(ok[col].tolist())
        if arr.size == 0:
            return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }

    for col in ["tm_templates_TM1", "tm_predictions_TM1", "a_min_ptm", "b_min_ptm", "rmsd_templates", "rmsd_predictions"]:
        summary[f"stats_{col}"] = _stats(col)

    summary["fraction_same_sample_index"] = float(ok["same_sample_index"].mean()) if len(ok) else None
    for thresh in (0.5, 0.6, 0.7, 0.8, 0.9):
        summary[f"fraction_tm_templates_ge_{thresh}"] = float((ok["tm_templates_TM1"] >= thresh).mean())
        summary[f"fraction_tm_predictions_ge_{thresh}"] = float((ok["tm_predictions_TM1"] >= thresh).mean())
    return summary


def _plot_tm_distribution(
    tm_templates: np.ndarray,
    tm_predictions: np.ndarray,
    output_path: Path,
    replica_a_label: str,
    replica_b_label: str,
    extra_title: str = "",
) -> None:
    """Histogram + CDF for TM-score of templates and predictions, side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bins = np.linspace(0.0, 1.0, 41)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    title_suffix = f"{replica_a_label} vs {replica_b_label}"
    if extra_title:
        title_suffix += f" {extra_title}"

    datasets = [
        ("Templates (top-1)", tm_templates, "tab:blue"),
        ("Predictions (top-1)", tm_predictions, "tab:orange"),
    ]
    for col, (label, arr, color) in enumerate(datasets):
        ax_hist = axes[0, col]
        ax_cdf = axes[1, col]
        if arr.size == 0:
            for ax in (ax_hist, ax_cdf):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, 1)
            ax_hist.set_title(f"{label}: TM1 histogram ({title_suffix})")
            ax_cdf.set_title(f"{label}: TM1 CDF ({title_suffix})")
            continue

        ax_hist.hist(arr, bins=bins, color=color, edgecolor="black", alpha=0.85)
        ax_hist.axvline(float(np.mean(arr)), linestyle="--", color="black", linewidth=1, label=f"mean={np.mean(arr):.3f}")
        ax_hist.axvline(float(np.median(arr)), linestyle=":", color="black", linewidth=1, label=f"median={np.median(arr):.3f}")
        ax_hist.set_title(f"{label}: TM1 histogram ({title_suffix})")
        ax_hist.set_xlabel("TM-score (TM1)")
        ax_hist.set_ylabel("count")
        ax_hist.set_xlim(0, 1)
        ax_hist.legend(loc="upper right", fontsize=8)

        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax_cdf.plot(sorted_arr, cdf, color=color, linewidth=2)
        for t in thresholds:
            frac_ge = float((arr >= t).mean())
            ax_cdf.axvline(t, color="gray", linestyle=":", linewidth=0.8)
            ax_cdf.text(t, 0.02, f"≥{t}\n{frac_ge:.2f}", ha="center", va="bottom", fontsize=7, color="gray")
        ax_cdf.set_title(f"{label}: TM1 CDF ({title_suffix})")
        ax_cdf.set_xlabel("TM-score (TM1)")
        ax_cdf.set_ylabel("cumulative fraction")
        ax_cdf.set_xlim(0, 1)
        ax_cdf.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare top-1 AF2Rank template/prediction between two Proteina replicas")
    parser.add_argument("--replica_a_dir", required=True)
    parser.add_argument("--replica_b_dir", required=True)
    parser.add_argument("--replica_a_label", default="a")
    parser.add_argument("--replica_b_label", default="b")
    parser.add_argument("--input", required=True, help="Input CSV with protein IDs to analyze")
    parser.add_argument("--id_col", default="id", help="Column name for protein ID in --input CSV (default: id)")
    parser.add_argument("--max_sample_index", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--tmscore_mode", type=int, default=5)
    parser.add_argument("--ptm_cutoff", type=float, default=0.7,
                        help="pTM cutoff (default 0.7); used to flag passes_cutoff_a/b/either columns")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    a_dir = Path(args.replica_a_dir).resolve()
    b_dir = Path(args.replica_b_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_ids = _load_protein_ids_from_input(args.input, args.id_col)
    if not protein_ids:
        logger.error("No protein IDs found in %s[%s]; nothing to do.", args.input, args.id_col)
        return 1
    logger.info("Comparing %d proteins from %s across replicas (labels: %s vs %s)",
                len(protein_ids), args.input, args.replica_a_label, args.replica_b_label)

    results: List[Dict[str, object]] = []
    if args.num_workers > 1 and len(protein_ids) > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(compare_protein, pid, str(a_dir), str(b_dir), args.max_sample_index, args.tmscore_mode, args.ptm_cutoff): pid
                for pid in protein_ids
            }
            for i, fut in enumerate(as_completed(futures), 1):
                pid = futures[fut]
                row = fut.result()
                results.append(row)
                if i % 25 == 0 or i == len(protein_ids):
                    logger.info("Processed %d/%d (last: %s, status: %s)", i, len(protein_ids), pid, row["status"])
    else:
        for i, pid in enumerate(protein_ids, 1):
            row = compare_protein(pid, str(a_dir), str(b_dir), args.max_sample_index, args.tmscore_mode, args.ptm_cutoff)
            results.append(row)
            logger.info("Processed %d/%d (%s, status: %s)", i, len(protein_ids), pid, row["status"])

    df = pd.DataFrame(results).sort_values("protein_id").reset_index(drop=True)
    csv_path = out_dir / "compare_replicas_topk.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))

    ok_all = df[df["status"] == "ok"]
    ok_passing = ok_all[ok_all["passes_cutoff_either"]]

    variants = [
        ("unfiltered", False, ok_all,
         out_dir / "compare_replicas_summary.json",
         out_dir / "compare_replicas_tm_distribution.png",
         f"[all ok, n={len(ok_all)}]"),
        ("passing", True, ok_passing,
         out_dir / "compare_replicas_summary_passing.json",
         out_dir / "compare_replicas_tm_distribution_passing.png",
         f"n={len(ok_passing)}]"),
    ]
    for name, passing_only, subset_df, json_path, plot_path, extra_title in variants:
        summary = _aggregate_stats(df, args, passing_only=passing_only)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Wrote %s", json_path)

        tm_templates = _finite_series(subset_df["tm_templates_TM1"].tolist())
        tm_predictions = _finite_series(subset_df["tm_predictions_TM1"].tolist())
        _plot_tm_distribution(
            tm_templates, tm_predictions, plot_path,
            args.replica_a_label, args.replica_b_label,
            extra_title=extra_title,
        )
        logger.info("Wrote %s", plot_path)

        logger.info("[%s] compared=%d skipped=%d", name, summary["num_targets_compared"], summary["num_skipped"])
        if summary.get("stats_tm_templates_TM1", {}).get("n", 0):
            logger.info("[%s] tm_templates_TM1 mean=%.4f median=%.4f",
                        name, summary["stats_tm_templates_TM1"]["mean"], summary["stats_tm_templates_TM1"]["median"])
        if summary.get("stats_tm_predictions_TM1", {}).get("n", 0):
            logger.info("[%s] tm_predictions_TM1 mean=%.4f median=%.4f",
                        name, summary["stats_tm_predictions_TM1"]["mean"], summary["stats_tm_predictions_TM1"]["median"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
