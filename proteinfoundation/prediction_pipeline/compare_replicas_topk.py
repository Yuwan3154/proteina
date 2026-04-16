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
        "status": status,
    }
    return row


def compare_protein(
    protein_id: str,
    replica_a_dir: str,
    replica_b_dir: str,
    max_sample_index: int,
    tmscore_mode: int,
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
        "status": "ok",
    }
    return row


def _enumerate_protein_ids(
    replica_a_dir: Path, replica_b_dir: Path,
    dataset_csv: Optional[str], id_column: str,
) -> List[str]:
    if dataset_csv:
        df = pd.read_csv(dataset_csv)
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not in {dataset_csv}")
        ids = [str(x).strip() for x in df[id_column].dropna().tolist() if str(x).strip()]
        return sorted(set(ids))
    a_ids = {p.name for p in replica_a_dir.iterdir() if p.is_dir()} if replica_a_dir.exists() else set()
    b_ids = {p.name for p in replica_b_dir.iterdir() if p.is_dir()} if replica_b_dir.exists() else set()
    return sorted(a_ids & b_ids)


def _finite_series(values: List[float]) -> np.ndarray:
    arr = np.array([v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    return arr


def _aggregate_stats(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    ok = df[df["status"] == "ok"].copy()
    skipped_counts = df[df["status"] != "ok"]["status"].value_counts().to_dict()
    summary: Dict[str, object] = {
        "replica_a_dir": args.replica_a_dir,
        "replica_b_dir": args.replica_b_dir,
        "replica_a_label": args.replica_a_label,
        "replica_b_label": args.replica_b_label,
        "max_sample_index": args.max_sample_index,
        "tmscore_mode": args.tmscore_mode,
        "num_targets_total": int(len(df)),
        "num_targets_compared": int(len(ok)),
        "num_skipped": int(len(df) - len(ok)),
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
    for thresh in (0.7, 0.8, 0.9):
        summary[f"fraction_tm_templates_ge_{thresh}"] = float((ok["tm_templates_TM1"] >= thresh).mean())
        summary[f"fraction_tm_predictions_ge_{thresh}"] = float((ok["tm_predictions_TM1"] >= thresh).mean())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare top-1 AF2Rank template/prediction between two Proteina replicas")
    parser.add_argument("--replica_a_dir", required=True)
    parser.add_argument("--replica_b_dir", required=True)
    parser.add_argument("--replica_a_label", default="a")
    parser.add_argument("--replica_b_label", default="b")
    parser.add_argument("--dataset_csv", default=None)
    parser.add_argument("--id_column", default="id")
    parser.add_argument("--max_sample_index", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--tmscore_mode", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    a_dir = Path(args.replica_a_dir).resolve()
    b_dir = Path(args.replica_b_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_ids = _enumerate_protein_ids(a_dir, b_dir, args.dataset_csv, args.id_column)
    if not protein_ids:
        logger.error("No protein IDs enumerated; nothing to do.")
        return 1
    logger.info("Comparing %d proteins across replicas (labels: %s vs %s)", len(protein_ids), args.replica_a_label, args.replica_b_label)

    results: List[Dict[str, object]] = []
    if args.num_workers > 1 and len(protein_ids) > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(compare_protein, pid, str(a_dir), str(b_dir), args.max_sample_index, args.tmscore_mode): pid
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
            row = compare_protein(pid, str(a_dir), str(b_dir), args.max_sample_index, args.tmscore_mode)
            results.append(row)
            logger.info("Processed %d/%d (%s, status: %s)", i, len(protein_ids), pid, row["status"])

    df = pd.DataFrame(results).sort_values("protein_id").reset_index(drop=True)
    csv_path = out_dir / "compare_replicas_topk.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))

    summary = _aggregate_stats(df, args)
    json_path = out_dir / "compare_replicas_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", json_path)

    logger.info("Summary: compared=%d skipped=%d", summary["num_targets_compared"], summary["num_skipped"])
    if summary.get("stats_tm_templates_TM1", {}).get("n", 0):
        logger.info("tm_templates_TM1 mean=%.4f median=%.4f", summary["stats_tm_templates_TM1"]["mean"], summary["stats_tm_templates_TM1"]["median"])
    if summary.get("stats_tm_predictions_TM1", {}).get("n", 0):
        logger.info("tm_predictions_TM1 mean=%.4f median=%.4f", summary["stats_tm_predictions_TM1"]["mean"], summary["stats_tm_predictions_TM1"]["median"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
