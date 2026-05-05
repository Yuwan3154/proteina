#!/usr/bin/env python3
"""
Central TM-score analysis for Proteina outputs.

This module owns:
- sample-vs-sample diversity via ``USalign -dir``
- reference-vs-template / reference-vs-prediction via ``USalign -dir2``
- matched template-vs-prediction TM-score enrichment
- regeneration of ProteinEBM / AF2Rank per-protein summaries after scorer CSVs exist

Compatibility note:
- ``compute_pairwise_tm`` / ``compute_diversity_for_proteins`` keep the old
  diversity-only API
- ``run_analysis_for_protein`` / ``compute_analysis_for_proteins`` implement the
  broader post-hoc analysis stage
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteinfoundation.af2rank_evaluation.cif_chain_mapping import resolve_ground_truth_usalign_chain
from proteinfoundation.af2rank_evaluation.protein_tar_utils import (
    pack_protein_dirs,
    protein_glob_members,
    read_protein_text,
    restore_selected_protein_dirs,
)
from proteinfoundation.af2rank_evaluation.sharding_utils import (
    add_shard_args,
    default_progress_check_workers,
    filter_proteins_threaded,
    lengths_from_csv,
    resolve_shard_args,
    shard_proteins,
)
from proteinfoundation.af2rank_evaluation.topk_summary_utils import generate_topk_summary_csv
from proteinfoundation.af2rank_evaluation.usalign_tabular import (
    normalize_usalign_structure_name,
    parse_usalign_outfmt2_named_rows,
    parse_usalign_pair_outfmt2,
)

logger = logging.getLogger(__name__)

_plot_lock = threading.Lock()

_USALIGN_PARALLEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def resolve_num_workers(explicit: Optional[int]) -> int:
    if explicit is not None:
        return max(1, min(64, int(explicit)))
    n = os.cpu_count() or 1
    return max(1, min(64, n))


def _normalized_folder(path: str) -> str:
    folder = os.path.abspath(path)
    if not folder.endswith(os.sep):
        folder = folder + os.sep
    return folder


def _default_chain(protein_id: str) -> Optional[str]:
    if "_" in protein_id:
        return protein_id.split("_", 1)[1]
    return None


def _find_reference_cif(protein_id: str, cif_dir: Optional[str]) -> Optional[str]:
    if not cif_dir:
        return None
    pdb_id = protein_id.split("_")[0]
    base = Path(cif_dir)
    direct = base / f"{pdb_id}.cif"
    if direct.exists():
        return str(direct)
    for child in sorted(base.iterdir()):
        candidate = child / f"{pdb_id}.cif"
        if candidate.exists():
            return str(candidate)
    return None


def _discover_template_paths(protein_dir: str, protein_id: str) -> List[str]:
    pattern = os.path.join(protein_dir, f"{protein_id}_*.pdb")
    return sorted(glob.glob(pattern))


def _write_list_file(names: Sequence[str]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    for name in names:
        handle.write(str(name) + "\n")
    handle.close()
    return handle.name


def _parse_usalign_output(lines: Sequence[str]) -> Dict[str, float]:
    def _parse_float(field: str) -> float:
        return float(field.split("=")[1].split()[0])

    out = {"rms": 0.0, "tms": 0.0, "gdt": 0.0}
    for line in lines:
        line = line.rstrip()
        if line.startswith("RMSD"):
            out["rms"] = _parse_float(line)
        elif line.startswith("TM-score"):
            out["tms"] = _parse_float(line)
        elif line.startswith("GDT-TS-score"):
            out["gdt"] = _parse_float(line)
    return out


def run_pairwise_tm(
    path_a: str,
    path_b: str,
    env: Optional[Dict[str, str]] = None,
    chain1: Optional[str] = None,
    chain2: Optional[str] = None,
    tmscore_exe: str = "USalign",
) -> Dict[str, float]:
    exe = shutil.which(tmscore_exe) or shutil.which("USalign") or shutil.which("TMscore")
    if exe is None:
        raise FileNotFoundError("Neither USalign nor TMscore found in PATH")

    cmd = [exe, path_a, path_b]
    if os.path.basename(exe) == "USalign":
        if chain1 is not None:
            cmd += ["-chain1", chain1]
        if chain2 is not None:
            cmd += ["-chain2", chain2]
        cmd += ["-TMscore", "5", "-outfmt", "2"]

    subprocess_env = None
    if env is not None:
        subprocess_env = os.environ.copy()
        subprocess_env.update(env)

    output = subprocess.check_output(cmd, text=True, env=subprocess_env)
    if os.path.basename(exe) == "USalign":
        return parse_usalign_pair_outfmt2(output)
    return _parse_usalign_output(output.splitlines())


def run_template_template_dir(
    protein_dir: str,
    basenames: Sequence[str],
    env: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    exe = shutil.which("USalign")
    if exe is None:
        raise FileNotFoundError("USalign not found in PATH")

    list_path = _write_list_file(basenames)
    try:
        cmd = [exe, "-dir", _normalized_folder(protein_dir), list_path, "-TMscore", "5", "-outfmt", "2"]
        subprocess_env = os.environ.copy()
        if env is not None:
            subprocess_env.update(env)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env)
    finally:
        os.unlink(list_path)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"USalign -dir failed (rc={proc.returncode}): {tail}")

    rows = parse_usalign_outfmt2_named_rows(proc.stdout)
    deduped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        name1 = str(row["structure_1_name"])
        name2 = str(row["structure_2_name"])
        if name1 == name2:
            continue
        key = tuple(sorted((name1, name2)))
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())


def run_dir2_rows(
    chain1_path: str,
    folder2: str,
    names2: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    chain1: Optional[str] = None,
    chain2: Optional[str] = None,
) -> List[Dict[str, object]]:
    exe = shutil.which("USalign")
    if exe is None:
        raise FileNotFoundError("USalign not found in PATH")

    list2 = _write_list_file(names2)
    try:
        cmd = [
            exe,
            os.path.abspath(chain1_path),
            "-dir2",
            _normalized_folder(folder2),
            list2,
        ]
        if chain1 is not None:
            cmd += ["-chain1", chain1]
        if chain2 is not None:
            cmd += ["-chain2", chain2]
        cmd += ["-TMscore", "5", "-outfmt", "2"]
        subprocess_env = os.environ.copy()
        if env is not None:
            subprocess_env.update(env)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env)
    finally:
        os.unlink(list2)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"USalign -dir2 failed (rc={proc.returncode}): {tail}")
    return parse_usalign_outfmt2_named_rows(proc.stdout)


def run_reference_vs_paths_dir2(
    reference_path: str,
    target_paths: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    chain1: Optional[str] = None,
    chain2: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    reference_abs = os.path.abspath(reference_path)
    reference_name = os.path.basename(reference_abs)
    resolved_chain1 = resolve_ground_truth_usalign_chain(reference_abs, chain1)
    if resolved_chain1 != chain1:
        logger.debug("Resolved reference chain for %s: %s -> %s", reference_abs, chain1, resolved_chain1)
    target_names = [os.path.basename(os.path.abspath(path)) for path in target_paths]
    by_folder: Dict[str, List[str]] = {}
    for path in target_paths:
        folder = str(Path(path).resolve().parent)
        by_folder.setdefault(folder, []).append(str(Path(path).resolve()))

    metrics_by_name: Dict[str, Dict[str, float]] = {}
    for folder, paths in by_folder.items():
        rows = run_dir2_rows(
            chain1_path=reference_abs,
            folder2=folder,
            names2=[os.path.basename(path) for path in paths],
            env=env,
            chain1=resolved_chain1,
            chain2=chain2,
        )
        for row in rows:
            name1 = str(row["structure_1_name"])
            name2 = str(row["structure_2_name"])
            if name1 == reference_name and name2 in target_names:
                metrics_by_name[name2] = {
                    "tms": float(row["tms"]),
                    "rms": float(row["rms"]),
                    "gdt": float(row["gdt"]),
                }
            elif name2 == reference_name and name1 in target_names:
                metrics_by_name[name1] = {
                    "tms": float(row["tms2"]),
                    "rms": float(row["rms"]),
                    "gdt": float(row["gdt"]),
                }
    return metrics_by_name


def run_matched_template_prediction_tm(
    template_prediction_pairs: Sequence[Tuple[str, str]],
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    metrics_by_name: Dict[str, Dict[str, float]] = {}
    for template_path, prediction_path in template_prediction_pairs:
        template_name = os.path.basename(template_path)
        metrics_by_name[template_name] = run_pairwise_tm(template_path, prediction_path, env=env)
    return metrics_by_name


def _spearmanr(x: List[float], y: List[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have same shape")
    if x_arr.size < 2:
        return float("nan")

    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
        sorted_a = a[order]
        i = 0
        while i < len(sorted_a):
            j = i + 1
            while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
                j += 1
            if j - i > 1:
                avg = (i + 1 + j) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    rx = rankdata(x_arr) - rankdata(x_arr).mean()
    ry = rankdata(y_arr) - rankdata(y_arr).mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def plot_pairwise_tm_histogram(
    tm_values: List[float],
    output_path: str,
    protein_id: str,
) -> None:
    with _plot_lock:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(tm_values, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
        ax.set_xlabel("Pairwise TM-score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Sample-to-Sample TM-score Distribution - {protein_id}", fontsize=13)
        tm_arr = np.array(tm_values)
        annotation = (
            f"N={len(tm_values)} pairs\n"
            f"mean={tm_arr.mean():.3f}, median={np.median(tm_arr):.3f}\n"
            f"std={tm_arr.std():.3f}"
        )
        ax.annotate(
            annotation,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _compute_pairwise_template_metrics(
    protein_dir: str,
    protein_id: str,
    use_usalign_dir: bool = True,
) -> Tuple[List[float], str, List[str]]:
    template_paths = _discover_template_paths(protein_dir, protein_id)
    if len(template_paths) < 2:
        return [], "none", template_paths

    basenames = [os.path.basename(path) for path in template_paths]
    tm_values: List[float] = []
    pairwise_mode = "per_pair"
    if use_usalign_dir and shutil.which("USalign"):
        rows = run_template_template_dir(protein_dir, basenames, env=None)
        tm_values = [float(row["tms"]) for row in rows]
        pairwise_mode = "usalign_dir"

    if not tm_values:
        for i, j in combinations(range(len(template_paths)), 2):
            metrics = run_pairwise_tm(template_paths[i], template_paths[j], env=None)
            tm_values.append(float(metrics["tms"]))
        pairwise_mode = "per_pair"

    return tm_values, pairwise_mode, template_paths


def compute_pairwise_tm(
    protein_dir: str,
    protein_id: str,
    output_subdir: str = "proteina_diversity",
    skip_existing: bool = True,
    pair_workers: int = 1,
    use_usalign_dir: bool = True,
) -> Optional[dict]:
    del pair_workers
    out_dir = os.path.join(str(protein_dir), output_subdir)
    summary_path = os.path.join(out_dir, f"diversity_summary_{protein_id}.json")
    if skip_existing and os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            existing = json.load(f)
        current_templates = _discover_template_paths(str(protein_dir), protein_id)
        cached_n = existing.get("n_samples", 0)
        if len(current_templates) <= cached_n:
            return existing
        logger.info(
            "[%s] Diversity sample count changed (%d → %d), recomputing",
            protein_id, cached_n, len(current_templates),
        )

    tm_values, pairwise_mode, template_paths = _compute_pairwise_template_metrics(
        protein_dir=str(protein_dir),
        protein_id=protein_id,
        use_usalign_dir=use_usalign_dir,
    )
    if not tm_values:
        return None

    tm_arr = np.array(tm_values, dtype=np.float64)
    summary = {
        "protein_id": protein_id,
        "n_samples": len(template_paths),
        "n_pairs": len(tm_values),
        "pairwise_tm_mode": pairwise_mode,
        "mean_tem_to_tem_tm": float(tm_arr.mean()),
        "std_tem_to_tem_tm": float(tm_arr.std()),
        "median_tem_to_tem_tm": float(np.median(tm_arr)),
        "min_tem_to_tem_tm": float(tm_arr.min()),
        "max_tem_to_tem_tm": float(tm_arr.max()),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    hist_path = os.path.join(out_dir, f"pairwise_tm_histogram_{protein_id}.png")
    plot_pairwise_tm_histogram(tm_values, hist_path, protein_id)
    return summary


def compute_diversity_for_proteins(
    inference_dir: str,
    protein_ids: List[str],
    output_subdir: str = "proteina_diversity",
    skip_existing: bool = True,
    num_workers: Optional[int] = None,
    use_usalign_dir: bool = True,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    worker_count = min(resolve_num_workers(num_workers), max(1, len(protein_ids)))
    if worker_count == 1 or len(protein_ids) <= 1:
        for protein_id in protein_ids:
            protein_dir = os.path.join(inference_dir, protein_id)
            if not os.path.isdir(protein_dir):
                continue
            summary = compute_pairwise_tm(
                protein_dir=protein_dir,
                protein_id=protein_id,
                output_subdir=output_subdir,
                skip_existing=skip_existing,
                use_usalign_dir=use_usalign_dir,
            )
            if summary is not None:
                results[protein_id] = summary
        return results

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                compute_pairwise_tm,
                os.path.join(inference_dir, protein_id),
                protein_id,
                output_subdir,
                skip_existing,
                1,
                use_usalign_dir,
            ): protein_id
            for protein_id in protein_ids
            if os.path.isdir(os.path.join(inference_dir, protein_id))
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                summary = future.result()
                if summary is not None:
                    results[str(summary["protein_id"])] = summary
            except Exception as e:
                logger.error("Analysis failed for %s: %s", pid, e)
    return results


def find_diversity_summaries(inference_dir: str, subdir: str = "proteina_diversity") -> List[str]:
    pattern = os.path.join(inference_dir, "*", subdir, "diversity_summary_*.json")
    return sorted(glob.glob(pattern))


def load_diversity_data(inference_dir: str, subdir: str = "proteina_diversity") -> dict:
    data: Dict[str, dict] = {}
    for path in find_diversity_summaries(inference_dir, subdir):
        with open(path, "r") as f:
            summary = json.load(f)
        protein_id = summary.get("protein_id") or Path(path).parent.parent.name
        data[str(protein_id)] = summary
    return data


def _plot_proteinebm_results(results: List[Dict[str, object]], output_dir: str, protein_id: str) -> Dict[str, str]:
    energies = np.asarray([float(r["energy"]) for r in results], dtype=np.float64)
    tm = np.asarray([float(r["tm_ref_template"]) for r in results], dtype=np.float64)
    score = -energies

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.scatter(tm, score, s=18, alpha=0.5)
    rho = _spearmanr(tm.tolist(), score.tolist()) if len(tm) > 1 else float("nan")
    plt.title(f"ProteinEBM: score(-energy) vs TM(ref, decoy)\n{protein_id} | Spearman rho={rho:.3f}")
    plt.xlabel("TM-score (Reference vs Decoy) [tm_ref_template]")
    plt.ylabel("ProteinEBM score (-energy, higher is better)")
    plt.grid(True, alpha=0.3)
    score_plot = os.path.join(output_dir, f"proteinebm_{protein_id}_score_vs_true_quality.png")
    plt.tight_layout()
    plt.savefig(score_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    valid_energies = energies[~np.isnan(energies)]
    fig = plt.figure(figsize=(8, 6), dpi=120)
    if len(valid_energies) > 0:
        plt.hist(valid_energies, bins=30, alpha=0.8)
    else:
        plt.text(0.5, 0.5, "No valid energies (all NaN)", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title(f"ProteinEBM energy histogram\n{protein_id}")
    plt.xlabel("Energy (lower is better)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    energy_hist = os.path.join(output_dir, f"proteinebm_{protein_id}_energy_hist.png")
    plt.tight_layout()
    plt.savefig(energy_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    valid_tm = tm[~np.isnan(tm)]
    fig = plt.figure(figsize=(8, 6), dpi=120)
    if len(valid_tm) > 0:
        plt.hist(valid_tm, bins=30, alpha=0.8)
    else:
        plt.text(0.5, 0.5, "No valid TM scores (all NaN)", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title(f"TM(ref, decoy) histogram\n{protein_id}")
    plt.xlabel("TM-score (Reference vs Decoy)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    tm_hist = os.path.join(output_dir, f"proteinebm_{protein_id}_tm_hist.png")
    plt.tight_layout()
    plt.savefig(tm_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "score_vs_true_quality": os.path.abspath(score_plot),
        "energy_hist": os.path.abspath(energy_hist),
        "tm_hist": os.path.abspath(tm_hist),
    }


def _rescale(a: np.ndarray, amin: Optional[float] = None, amax: Optional[float] = None) -> np.ndarray:
    out = np.copy(a)
    if amin is None:
        amin = out.min()
    if amax is None:
        amax = out.max()
    if amax == amin:
        return np.ones_like(out) * 0.5
    out[out < amin] = amin
    out[out > amax] = amax
    return (out - amin) / (amax - amin)


def _plot_metric(
    scores: List[Dict[str, object]],
    x: str,
    y: str,
    save_path: str,
    title: Optional[str] = None,
    diag: bool = False,
    scale_axis: bool = True,
) -> float:
    plt.figure(figsize=(8, 6), dpi=100)
    if title is not None:
        plt.title(title)

    x_vals = np.array([row.get(x, np.nan) for row in scores if "error" not in row], dtype=np.float64)
    y_vals = np.array([row.get(y, np.nan) for row in scores if "error" not in row], dtype=np.float64)
    color_vals = np.array([row.get("plddt", 0.7) for row in scores if "error" not in row], dtype=np.float64)
    color_vals = _rescale(color_vals, 0.5, 0.9) if len(color_vals) > 0 else color_vals

    correlation = 0.0
    if len(x_vals) == 0:
        plt.text(0.5, 0.5, "No valid data to plot", transform=plt.gca().transAxes, ha="center", va="center", fontsize=12)
    else:
        plt.scatter(x_vals, y_vals, c=color_vals * 0.75, s=20, vmin=0, vmax=1, cmap="gist_rainbow")
        if diag and len(x_vals) > 0:
            lims = [min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())]
            plt.plot(lims, lims, color="black", linestyle="--", alpha=0.5)

        valid_mask = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
        if valid_mask.sum() > 1:
            correlation = _spearmanr(x_vals[valid_mask].tolist(), y_vals[valid_mask].tolist())
            if not np.isnan(correlation):
                plt.text(
                    0.05,
                    0.95,
                    f"Spearman R: {correlation:.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            else:
                correlation = 0.0

    labels = {
        "ptm": "Predicted TM-score (pTM)",
        "plddt": "Predicted LDDT (pLDDT)",
        "composite": "AF2Rank Score (pTM x pLDDT)",
        "pae_mean": "Predicted Aligned Error",
        "tm_ref_template": "TM-score (Reference vs Template)",
        "tm_ref_pred": "TM-score (Reference vs Prediction)",
        "tm_template_pred": "TM-score (Template vs Prediction)",
        "rmsd_ref_template": "RMSD (Reference vs Template)",
        "rmsd_ref_pred": "RMSD (Reference vs Prediction)",
        "rmsd_template_pred": "RMSD (Template vs Prediction)",
        "gdt_ref_template": "GDT-TS (Reference vs Template)",
        "gdt_ref_pred": "GDT-TS (Reference vs Prediction)",
        "gdt_template_pred": "GDT-TS (Template vs Prediction)",
    }
    plt.xlabel(labels.get(x, x.replace("_", " ").title()))
    plt.ylabel(labels.get(y, y.replace("_", " ").title()))

    if scale_axis and x in {"ptm", "plddt", "composite"} and y in {"ptm", "plddt", "composite"}:
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

    if len(x_vals) > 0:
        cbar = plt.colorbar()
        cbar.set_label("pLDDT (scaled)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return correlation


def _plot_af2rank_results(scores: List[Dict[str, object]], output_dir: str, protein_id: str) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    plot_configs = [
        {
            "x": "tm_ref_template",
            "y": "composite",
            "title": "AF2Rank Analysis: Template Quality vs AF2 Confidence",
            "filename": f"af2rank_{protein_id}_template_quality_vs_composite.png",
        },
        {
            "x": "tm_ref_pred",
            "y": "ptm",
            "title": "AF2Rank Analysis: Prediction Quality vs pTM",
            "filename": f"af2rank_{protein_id}_prediction_quality_vs_ptm.png",
        },
        {
            "x": "tm_ref_template",
            "y": "tm_ref_pred",
            "diag": True,
            "title": "AF2Rank Analysis: Template vs Prediction Quality",
            "filename": f"af2rank_{protein_id}_template_vs_prediction_quality.png",
        },
        {
            "x": "composite",
            "y": "tm_ref_pred",
            "title": "AF2Rank Analysis: AF2 Composite Score vs True Quality",
            "filename": f"af2rank_{protein_id}_composite_vs_true_quality.png",
        },
        {
            "x": "ptm",
            "y": "composite",
            "title": "AF2Rank Analysis: pTM vs Composite Score",
            "filename": f"af2rank_{protein_id}_ptm_vs_composite.png",
        },
        {
            "x": "plddt",
            "y": "composite",
            "title": "AF2Rank Analysis: pLDDT vs Composite Score",
            "filename": f"af2rank_{protein_id}_plddt_vs_composite.png",
        },
        {
            "x": "ptm",
            "y": "plddt",
            "title": "AF2 Internal: pTM vs pLDDT correlation",
            "filename": f"af2rank_{protein_id}_ptm_vs_plddt.png",
        },
        {
            "x": "pae_mean",
            "y": "composite",
            "title": "AF2Rank Analysis: PAE vs Composite Score",
            "filename": f"af2rank_{protein_id}_pae_mean_vs_composite.png",
        },
    ]

    correlations: Dict[str, float] = {}
    for config in plot_configs:
        filename = str(config["filename"])
        correlations[filename] = _plot_metric(
            scores=scores,
            x=str(config["x"]),
            y=str(config["y"]),
            title=str(config["title"]),
            diag=bool(config.get("diag", False)),
            save_path=os.path.join(output_dir, filename),
        )
    return correlations


def _build_af2rank_summary(
    scores: List[Dict[str, object]],
    protein_id: str,
    reference_structure: Optional[str],
    inference_directory: str,
    output_directory: str,
    chain: Optional[str],
    recycles: Optional[int],
    scores_csv_path: str,
) -> Dict[str, object]:
    successful_scores = [score for score in scores if "error" not in score]
    spearman_rho_composite = None
    spearman_rho_ptm = None
    max_tm_ref_template = None
    max_tm_ref_pred = None
    tm_ref_template_at_max_composite = None
    tm_ref_pred_at_max_composite = None
    tm_ref_pred_at_max_ptm = None
    top_1_tm_ref_template = None
    top_5_tm_ref_template = None
    top_1_tm_ref_pred = None
    top_5_tm_ref_pred = None

    tm_ref_template_scores: List[float] = []
    tm_ref_pred_scores: List[float] = []
    composite_scores: List[float] = []
    ptm_scores: List[float] = []

    for score in successful_scores:
        tm_template = score.get("tm_ref_template")
        tm_pred = score.get("tm_ref_pred")
        composite = score.get("composite")
        ptm = score.get("ptm")
        if pd.notna(tm_template) and pd.notna(tm_pred) and pd.notna(composite) and pd.notna(ptm):
            tm_ref_template_scores.append(float(tm_template))
            tm_ref_pred_scores.append(float(tm_pred))
            composite_scores.append(float(composite))
            ptm_scores.append(float(ptm))

    if len(tm_ref_template_scores) > 1:
        spearman_rho_composite = _spearmanr(tm_ref_template_scores, composite_scores)
        spearman_rho_ptm = _spearmanr(tm_ref_pred_scores, ptm_scores)
        max_tm_ref_template = float(max(tm_ref_template_scores))
        max_tm_ref_pred = float(max(tm_ref_pred_scores))
        max_composite_idx = int(np.argmax(np.asarray(composite_scores)))
        max_ptm_idx = int(np.argmax(np.asarray(ptm_scores)))
        tm_ref_template_at_max_composite = float(tm_ref_template_scores[max_composite_idx])
        tm_ref_pred_at_max_composite = float(tm_ref_pred_scores[max_composite_idx])
        tm_ref_pred_at_max_ptm = float(tm_ref_pred_scores[max_ptm_idx])
        top_1_tm_ref_template = float(tm_ref_template_scores[max_composite_idx])
        top_5_tm_ref_template = float(max([tm_ref_template_scores[idx] for idx in np.argsort(-np.asarray(composite_scores))[:5]]))
        top_1_tm_ref_pred = float(tm_ref_pred_scores[max_ptm_idx])
        top_5_tm_ref_pred = float(max([tm_ref_pred_scores[idx] for idx in np.argsort(-np.asarray(ptm_scores))[:5]]))

    return {
        "protein_id": protein_id,
        "total_structures": len(scores),
        "successful_scores": len(successful_scores),
        "failed_scores": len([score for score in scores if "error" in score]),
        "reference_structure": reference_structure,
        "inference_directory": inference_directory,
        "output_directory": output_directory,
        "af2rank_directory": output_directory,
        "chain": chain,
        "recycles": recycles,
        "spearman_correlation_rho_composite": spearman_rho_composite,
        "spearman_correlation_rho_ptm": spearman_rho_ptm,
        "max_tm_ref_template": max_tm_ref_template,
        "max_tm_ref_pred": max_tm_ref_pred,
        "tm_ref_template_at_max_composite": tm_ref_template_at_max_composite,
        "tm_ref_pred_at_max_composite": tm_ref_pred_at_max_composite,
        "tm_ref_pred_at_max_ptm": tm_ref_pred_at_max_ptm,
        "scores_csv": scores_csv_path,
        "top_1_tm_ref_template": top_1_tm_ref_template,
        "top_5_tm_ref_template": top_5_tm_ref_template,
        "top_1_tm_ref_pred": top_1_tm_ref_pred,
        "top_5_tm_ref_pred": top_5_tm_ref_pred,
    }


def _build_proteinebm_summary(
    results: List[Dict[str, object]],
    protein_id: str,
    metadata: Dict[str, object],
    scores_csv_path: str,
    plot_paths: Dict[str, str],
) -> Dict[str, object]:
    spearman_rho_energy = None
    max_tm_ref_template = None
    tm_ref_template_at_min_energy = None
    top_1_tm_ref_template = None
    top_5_tm_ref_template = None

    if len(results) > 1:
        tm_vals = [float(row["tm_ref_template"]) for row in results if pd.notna(row.get("tm_ref_template"))]
        energy_vals = [float(row["energy"]) for row in results if pd.notna(row.get("tm_ref_template"))]
        if len(tm_vals) > 1:
            score_vals = [-energy for energy in energy_vals]
            spearman_rho_energy = _spearmanr(tm_vals, score_vals)
            max_tm_ref_template = float(max(tm_vals))
            best_idx = int(np.argmin(np.asarray(energy_vals)))
            tm_ref_template_at_min_energy = float(tm_vals[best_idx])
            top_1_tm_ref_template = tm_ref_template_at_min_energy
            top5_idx = np.argsort(np.asarray(energy_vals))[:5]
            top_5_tm_ref_template = float(max([tm_vals[int(i)] for i in top5_idx]))

    return {
        "protein_id": protein_id,
        "total_structures": len(results),
        "successful_scores": len(results),
        "runtime_seconds": metadata.get("runtime_seconds"),
        "t": metadata.get("t"),
        "reference_structure": metadata.get("reference_structure"),
        "chain": metadata.get("chain"),
        "proteinebm_config": metadata.get("proteinebm_config"),
        "proteinebm_checkpoint": metadata.get("proteinebm_checkpoint"),
        "template_self_condition": metadata.get("template_self_condition"),
        "scores_csv": os.path.abspath(scores_csv_path),
        "plots": plot_paths,
        "spearman_correlation_rho_composite": spearman_rho_energy,
        "spearman_correlation_rho_energy": spearman_rho_energy,
        "max_tm_ref_template": max_tm_ref_template,
        "tm_ref_template_at_max_composite": tm_ref_template_at_min_energy,
        "tm_ref_template_at_min_energy": tm_ref_template_at_min_energy,
        "top_1_tm_ref_template": top_1_tm_ref_template,
        "top_5_tm_ref_template": top_5_tm_ref_template,
        "status": "completed",
    }


def _ensure_af2rank_prediction_columns(scores_df: pd.DataFrame, scores_csv_path: str) -> pd.DataFrame:
    out_df = scores_df.copy()
    predicted_dir = Path(scores_csv_path).parent / "predicted_structures"
    if "predicted_structure_file" not in out_df.columns:
        out_df["predicted_structure_file"] = out_df["structure_file"].astype(str)
    if "predicted_structure_path" not in out_df.columns:
        out_df["predicted_structure_path"] = out_df["predicted_structure_file"].astype(str).apply(
            lambda name: str(predicted_dir / name)
        )
    return out_df


def enrich_proteinebm_outputs(
    protein_id: str,
    protein_dir: str,
    reference_cif: Optional[str],
    reference_chain: Optional[str],
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
) -> Optional[Dict[str, object]]:
    output_dir = Path(protein_dir) / proteinebm_analysis_subdir
    scores_csv = output_dir / f"proteinebm_scores_{protein_id}.csv"
    if not scores_csv.exists():
        return None

    scores_df = pd.read_csv(scores_csv)
    if "structure_file" not in scores_df.columns:
        scores_df["structure_file"] = scores_df["structure_path"].astype(str).apply(lambda path: Path(path).name)

    metrics_by_name: Dict[str, Dict[str, float]] = {}
    if reference_cif is not None and "structure_path" in scores_df.columns:
        metrics_by_name = run_reference_vs_paths_dir2(
            reference_path=reference_cif,
            target_paths=[str(path) for path in scores_df["structure_path"].astype(str).tolist()],
            env=None,
            chain1=reference_chain,
        )

    scores_df["tm_ref_template"] = scores_df["structure_file"].astype(str).apply(
        lambda name: metrics_by_name[name]["tms"] if name in metrics_by_name else float("nan")
    )
    scores_df["rmsd_ref_template"] = scores_df["structure_file"].astype(str).apply(
        lambda name: metrics_by_name[name]["rms"] if name in metrics_by_name else float("nan")
    )
    scores_df["gdt_ref_template"] = scores_df["structure_file"].astype(str).apply(
        lambda name: metrics_by_name[name]["gdt"] if name in metrics_by_name else float("nan")
    )
    scores_df.to_csv(scores_csv, index=False)

    summary_path = output_dir / f"proteinebm_summary_{protein_id}.json"
    existing_summary = json.load(open(summary_path)) if summary_path.exists() else {}
    existing_summary.setdefault("reference_structure", reference_cif)
    existing_summary.setdefault("chain", reference_chain)
    plot_paths = _plot_proteinebm_results(scores_df.to_dict("records"), str(output_dir), protein_id)
    summary = _build_proteinebm_summary(
        results=scores_df.to_dict("records"),
        protein_id=protein_id,
        metadata=existing_summary,
        scores_csv_path=str(scores_csv),
        plot_paths=plot_paths,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def enrich_af2rank_output_dir(
    protein_id: str,
    inference_dir: str,
    output_dir: str,
    reference_cif: Optional[str],
    reference_chain: Optional[str],
) -> Optional[Dict[str, object]]:
    scores_csv = Path(output_dir) / f"af2rank_scores_{protein_id}.csv"
    if not scores_csv.exists():
        return None

    scores_df = pd.read_csv(scores_csv)
    scores_df = _ensure_af2rank_prediction_columns(scores_df, str(scores_csv))

    template_metrics: Dict[str, Dict[str, float]] = {}
    prediction_metrics: Dict[str, Dict[str, float]] = {}
    if reference_cif is not None:
        template_metrics = run_reference_vs_paths_dir2(
            reference_path=reference_cif,
            target_paths=[str(path) for path in scores_df["structure_path"].astype(str).tolist()],
            env=None,
            chain1=reference_chain,
        )
        prediction_paths = [str(path) for path in scores_df["predicted_structure_path"].astype(str).tolist() if Path(path).exists()]
        if prediction_paths:
            prediction_metrics = run_reference_vs_paths_dir2(
                reference_path=reference_cif,
                target_paths=prediction_paths,
                env=None,
                chain1=reference_chain,
            )

    matched_pairs = [
        (str(template_path), str(prediction_path))
        for template_path, prediction_path in zip(
            scores_df["structure_path"].astype(str).tolist(),
            scores_df["predicted_structure_path"].astype(str).tolist(),
        )
        if Path(str(prediction_path)).exists()
    ]
    template_prediction_metrics = run_matched_template_prediction_tm(matched_pairs, env=None) if matched_pairs else {}

    scores_df["tm_ref_template"] = scores_df["structure_file"].astype(str).apply(
        lambda name: template_metrics[name]["tms"] if name in template_metrics else float("nan")
    )
    scores_df["rmsd_ref_template"] = scores_df["structure_file"].astype(str).apply(
        lambda name: template_metrics[name]["rms"] if name in template_metrics else float("nan")
    )
    scores_df["gdt_ref_template"] = scores_df["structure_file"].astype(str).apply(
        lambda name: template_metrics[name]["gdt"] if name in template_metrics else float("nan")
    )
    scores_df["tm_ref_pred"] = scores_df["predicted_structure_file"].astype(str).apply(
        lambda name: prediction_metrics[name]["tms"] if name in prediction_metrics else float("nan")
    )
    scores_df["rmsd_ref_pred"] = scores_df["predicted_structure_file"].astype(str).apply(
        lambda name: prediction_metrics[name]["rms"] if name in prediction_metrics else float("nan")
    )
    scores_df["gdt_ref_pred"] = scores_df["predicted_structure_file"].astype(str).apply(
        lambda name: prediction_metrics[name]["gdt"] if name in prediction_metrics else float("nan")
    )
    scores_df["tm_template_pred"] = scores_df["structure_file"].astype(str).apply(
        lambda name: template_prediction_metrics[name]["tms"] if name in template_prediction_metrics else float("nan")
    )
    scores_df["rmsd_template_pred"] = scores_df["structure_file"].astype(str).apply(
        lambda name: template_prediction_metrics[name]["rms"] if name in template_prediction_metrics else float("nan")
    )
    scores_df["gdt_template_pred"] = scores_df["structure_file"].astype(str).apply(
        lambda name: template_prediction_metrics[name]["gdt"] if name in template_prediction_metrics else float("nan")
    )
    scores_df.to_csv(scores_csv, index=False)

    summary_path = Path(output_dir) / f"af2rank_summary_{protein_id}.json"
    existing_summary = json.load(open(summary_path)) if summary_path.exists() else {}
    existing_summary.setdefault("reference_structure", reference_cif)
    existing_summary.setdefault("chain", reference_chain)
    existing_summary.setdefault("recycles", None)
    _plot_af2rank_results(scores_df.to_dict("records"), output_dir, protein_id)
    summary = _build_af2rank_summary(
        scores=scores_df.to_dict("records"),
        protein_id=protein_id,
        reference_structure=reference_cif,
        inference_directory=inference_dir,
        output_directory=output_dir,
        chain=reference_chain,
        recycles=existing_summary.get("recycles"),
        scores_csv_path=str(scores_csv),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _regenerate_topk_summary_if_present(
    protein_id: str,
    protein_dir: str,
    proteinebm_analysis_subdir: str,
) -> Optional[str]:
    protein_path = Path(protein_dir)
    topk_dir = protein_path / "af2rank_on_proteinebm_top_k"
    if not topk_dir.exists():
        return None

    m1_csv = topk_dir / "af2rank_analysis" / f"af2rank_scores_{protein_id}.csv"
    m2_csv = topk_dir / "af2rank_analysis_model_2_ptm" / f"af2rank_scores_{protein_id}.csv"
    if not m1_csv.exists() or not m2_csv.exists():
        return None

    scores_csv = protein_path / proteinebm_analysis_subdir / f"proteinebm_scores_{protein_id}.csv"
    if not scores_csv.exists():
        return None

    topk_df = pd.read_csv(scores_csv)
    staged_dir = topk_dir / "staged_topk_templates"
    if not staged_dir.exists():
        return None
    staged_files = sorted([path.name for path in staged_dir.iterdir() if path.is_file() or path.is_symlink()])
    topk_df["structure_file"] = topk_df["structure_path"].astype(str).apply(lambda path: Path(path).name)
    topk_df = topk_df[topk_df["structure_file"].isin(staged_files)].copy().reset_index(drop=True)

    cg2all_dir = topk_dir / "cg2all_topk_structures"
    allatom_map: Dict[str, str] = {}
    if cg2all_dir.exists():
        for row in topk_df.itertuples():
            candidate = cg2all_dir / f"{Path(str(row.structure_path)).stem}_allatom.pdb"
            if candidate.exists():
                allatom_map[str(row.structure_path)] = str(candidate)

    generate_topk_summary_csv(
        protein_id=protein_id,
        topk_df=topk_df,
        m1_csv=m1_csv,
        m2_csv=m2_csv,
        allatom_map=allatom_map,
        out_dir=topk_dir,
    )
    return str(topk_dir / f"af2rank_topk_summary_{protein_id}.csv")


def run_analysis_for_protein(
    protein_id: str,
    inference_dir: str,
    output_subdir: str = "proteina_analysis",
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    skip_existing: bool = True,
    cif_dir: Optional[str] = None,
    reference_cif: Optional[str] = None,
    reference_chain: Optional[str] = None,
    use_usalign_dir: bool = True,
    skip_diversity: bool = False,
) -> Optional[Dict[str, object]]:
    protein_dir = os.path.join(inference_dir, protein_id)
    if not os.path.isdir(protein_dir):
        return None

    analysis_dir = os.path.join(protein_dir, output_subdir)
    summary_path = os.path.join(analysis_dir, f"analysis_summary_{protein_id}.json")
    if skip_existing and os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            existing = json.load(f)
        # Detect incremental samples: if PDB count on disk exceeds the
        # n_samples recorded in the summary, the analysis is stale.
        current_templates = _discover_template_paths(protein_dir, protein_id)
        cached_n = existing.get("n_samples", 0)
        if len(current_templates) > cached_n:
            logger.info(
                "[%s] Sample count changed (%d → %d), re-running analysis",
                protein_id, cached_n, len(current_templates),
            )
        else:
            # Still run enrichment to generate any missing per-protein plots even when the
            # analysis JSON already exists (e.g. from a prior run or a code update).
            _ref = existing.get("reference_structure") or reference_cif or _find_reference_cif(protein_id, cif_dir)
            _chain = existing.get("reference_chain") or (reference_chain if reference_chain is not None else _default_chain(protein_id))
            try:
                enrich_proteinebm_outputs(
                    protein_id=protein_id,
                    protein_dir=protein_dir,
                    reference_cif=_ref,
                    reference_chain=_chain,
                    proteinebm_analysis_subdir=proteinebm_analysis_subdir,
                )
                for _af2rank_dir in [
                    os.path.join(protein_dir, "af2rank_analysis"),
                    os.path.join(protein_dir, "af2rank_analysis_model_2_ptm"),
                    os.path.join(protein_dir, "af2rank_on_proteinebm_top_k", "af2rank_analysis"),
                    os.path.join(protein_dir, "af2rank_on_proteinebm_top_k", "af2rank_analysis_model_2_ptm"),
                ]:
                    enrich_af2rank_output_dir(
                        protein_id=protein_id,
                        inference_dir=protein_dir,
                        output_dir=_af2rank_dir,
                        reference_cif=_ref,
                        reference_chain=_chain,
                    )
            except Exception as _e:
                logger.warning("[%s] enrichment in skip_existing path raised: %s", protein_id, _e)
            return existing

    resolved_reference_cif = reference_cif or _find_reference_cif(protein_id, cif_dir)
    resolved_reference_chain = reference_chain if reference_chain is not None else _default_chain(protein_id)

    if skip_diversity:
        template_paths = _discover_template_paths(protein_dir, protein_id)
        if not template_paths:
            return None
        tm_values = []
        pairwise_mode = None
        pairwise_arr = None
    else:
        tm_values, pairwise_mode, template_paths = _compute_pairwise_template_metrics(
            protein_dir=protein_dir,
            protein_id=protein_id,
            use_usalign_dir=use_usalign_dir,
        )
        if not tm_values:
            logger.warning(
                "[%s] Pairwise diversity computation yielded no values (insufficient templates or "
                "USalign failure). Proceeding without diversity metrics.",
                protein_id,
            )
            pairwise_arr = None
        else:
            pairwise_arr = np.array(tm_values, dtype=np.float64)

    os.makedirs(analysis_dir, exist_ok=True)
    pairwise_hist_path = None
    if not skip_diversity:
        pairwise_hist_path = os.path.join(analysis_dir, f"pairwise_tm_histogram_{protein_id}.png")
        if tm_values:
            plot_pairwise_tm_histogram(tm_values, pairwise_hist_path, protein_id)

    proteinebm_summary = enrich_proteinebm_outputs(
        protein_id=protein_id,
        protein_dir=protein_dir,
        reference_cif=resolved_reference_cif,
        reference_chain=resolved_reference_chain,
        proteinebm_analysis_subdir=proteinebm_analysis_subdir,
    )

    af2rank_dirs = [
        os.path.join(protein_dir, "af2rank_analysis"),
        os.path.join(protein_dir, "af2rank_analysis_model_2_ptm"),
        os.path.join(protein_dir, "af2rank_on_proteinebm_top_k", "af2rank_analysis"),
        os.path.join(protein_dir, "af2rank_on_proteinebm_top_k", "af2rank_analysis_model_2_ptm"),
    ]
    af2rank_summaries: Dict[str, Dict[str, object]] = {}
    for af2rank_dir in af2rank_dirs:
        summary = enrich_af2rank_output_dir(
            protein_id=protein_id,
            inference_dir=protein_dir,
            output_dir=af2rank_dir,
            reference_cif=resolved_reference_cif,
            reference_chain=resolved_reference_chain,
        )
        if summary is not None:
            af2rank_summaries[Path(af2rank_dir).name] = summary

    topk_summary_csv = _regenerate_topk_summary_if_present(
        protein_id=protein_id,
        protein_dir=protein_dir,
        proteinebm_analysis_subdir=proteinebm_analysis_subdir,
    )

    analysis_summary = {
        "protein_id": protein_id,
        "reference_structure": resolved_reference_cif,
        "reference_chain": resolved_reference_chain,
        "n_samples": len(template_paths),
        "n_pairs": len(tm_values) if tm_values else None,
        "pairwise_tm_mode": pairwise_mode,
        "mean_tem_to_tem_tm": float(pairwise_arr.mean()) if pairwise_arr is not None else None,
        "std_tem_to_tem_tm": float(pairwise_arr.std()) if pairwise_arr is not None else None,
        "median_tem_to_tem_tm": float(np.median(pairwise_arr)) if pairwise_arr is not None else None,
        "min_tem_to_tem_tm": float(pairwise_arr.min()) if pairwise_arr is not None else None,
        "max_tem_to_tem_tm": float(pairwise_arr.max()) if pairwise_arr is not None else None,
        "pairwise_tm_histogram": os.path.abspath(pairwise_hist_path) if pairwise_hist_path else None,
        "proteinebm_summary": proteinebm_summary,
        "af2rank_summaries": af2rank_summaries,
        "af2rank_topk_summary_csv": topk_summary_csv,
    }
    with open(summary_path, "w") as f:
        json.dump(analysis_summary, f, indent=2)
    return analysis_summary


def analysis_complete_in_tar_or_dir(
    inference_dir: str,
    protein_id: str,
    output_subdir: str = "proteina_analysis",
) -> bool:
    summary_rel = Path(output_subdir) / f"analysis_summary_{protein_id}.json"
    summary_text = read_protein_text(inference_dir, protein_id, summary_rel)
    if summary_text is None:
        return False
    summary = json.loads(summary_text)
    current_templates = protein_glob_members(inference_dir, protein_id, f"{protein_id}_*.pdb")
    return len(current_templates) <= int(summary.get("n_samples", 0) or 0)


def compute_analysis_for_proteins(
    inference_dir: str,
    protein_ids: List[str],
    output_subdir: str = "proteina_analysis",
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    skip_existing: bool = True,
    num_workers: Optional[int] = None,
    cif_dir: Optional[str] = None,
    use_usalign_dir: bool = True,
    skip_diversity: bool = False,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    worker_count = min(resolve_num_workers(num_workers), max(1, len(protein_ids)))
    if worker_count == 1 or len(protein_ids) <= 1:
        for protein_id in protein_ids:
            summary = run_analysis_for_protein(
                protein_id=protein_id,
                inference_dir=inference_dir,
                output_subdir=output_subdir,
                proteinebm_analysis_subdir=proteinebm_analysis_subdir,
                skip_existing=skip_existing,
                cif_dir=cif_dir,
                use_usalign_dir=use_usalign_dir,
                skip_diversity=skip_diversity,
            )
            if summary is not None:
                results[protein_id] = summary
        return results

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                run_analysis_for_protein,
                protein_id,
                inference_dir,
                output_subdir,
                proteinebm_analysis_subdir,
                skip_existing,
                cif_dir,
                None,
                None,
                use_usalign_dir,
                skip_diversity,
            ): protein_id
            for protein_id in protein_ids
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                summary = future.result()
                if summary is not None:
                    results[str(summary["protein_id"])] = summary
            except Exception as e:
                logger.error("Analysis failed for %s: %s", pid, e)
    return results


def find_analysis_summaries(inference_dir: str, subdir: str = "proteina_analysis") -> List[str]:
    pattern = os.path.join(inference_dir, "*", subdir, "analysis_summary_*.json")
    return sorted(glob.glob(pattern))


def load_analysis_data(inference_dir: str, subdir: str = "proteina_analysis") -> dict:
    data: Dict[str, dict] = {}
    for path in find_analysis_summaries(inference_dir, subdir):
        with open(path, "r") as f:
            summary = json.load(f)
        protein_id = summary.get("protein_id") or Path(path).parent.parent.name
        data[str(protein_id)] = summary
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Central Proteina TM-score analysis")
    parser.add_argument("--inference_dir", required=True, help="Base inference directory")
    parser.add_argument("--protein_ids", nargs="*", help="Protein IDs to process")
    parser.add_argument("--csv_file", help="CSV file with protein IDs")
    parser.add_argument("--csv_col", default="id", help="Protein ID column in CSV (default: id)")
    parser.add_argument("--cif_dir", default="", help="Optional directory with reference CIF files")
    parser.add_argument("--output_subdir", default="proteina_analysis", help="Per-protein analysis output subdir")
    parser.add_argument(
        "--proteinebm_analysis_subdir",
        default="proteinebm_v2_cathmd_analysis",
        help="Per-protein ProteinEBM scorer output subdir",
    )
    parser.add_argument("--rerun", action="store_true", help="Recompute even if analysis summary exists")
    parser.add_argument("--num_workers", type=int, default=None, help="Max one-process-per-protein worker count")
    parser.add_argument("--no_usalign_dir", action="store_true", help="Disable USalign -dir for template all-to-all")
    parser.add_argument("--skip_diversity", action="store_true", help="Skip pairwise template-to-template diversity computation")
    parser.add_argument(
        "--tar_protein_dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-protein inference directories as uncompressed <protein_id>.tar archives (default: True).",
    )
    parser.add_argument("--progress_check_workers", type=int, default=default_progress_check_workers(),
                        help="Thread workers for progress checks (default: min(32, cpu_count * 4)).")
    parser.add_argument("--dynamic_resharding", action=argparse.BooleanOptionalAction, default=True,
                        help="Filter global progress before sharding each step to reduce idle shards (default: True).")
    add_shard_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.protein_ids:
        protein_ids = args.protein_ids
    elif args.csv_file:
        df = pd.read_csv(args.csv_file)
        protein_ids = df[args.csv_col].dropna().astype(str).str.strip().unique().tolist()
    else:
        protein_ids = sorted(
            [
                name
                for name in os.listdir(args.inference_dir)
                if os.path.isdir(os.path.join(args.inference_dir, name))
            ]
        )

    if not protein_ids:
        raise ValueError("No protein IDs found")

    global_protein_ids = list(protein_ids)
    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)
    lengths = lengths_from_csv(getattr(args, "csv_file", None) or "", args.csv_col, args.len_col)
    data_dir = os.environ.get("DATA_PATH", str(Path(__file__).resolve().parents[2] / "data"))
    if shard_index is not None:
        static_protein_ids = shard_proteins(
            global_protein_ids, shard_index, num_shards, lengths=lengths, data_dir=None if lengths is not None else data_dir
        )
        protein_ids = list(static_protein_ids)
        logger.info(f"Central analysis shard {shard_index}/{num_shards}: {len(protein_ids)} proteins selected")

    analysis_ids = list(protein_ids)
    complete_ids: List[str] = []
    if args.tar_protein_dirs and not args.rerun:
        check_start = time.perf_counter()
        check_candidates = global_protein_ids if args.dynamic_resharding and shard_index is not None else protein_ids
        needing_ids, complete_ids = filter_proteins_threaded(
            check_candidates,
            lambda protein_id: analysis_complete_in_tar_or_dir(args.inference_dir, protein_id, args.output_subdir),
            max_workers=args.progress_check_workers,
        )
        if args.dynamic_resharding and shard_index is not None:
            analysis_ids = shard_proteins(
                needing_ids, shard_index, num_shards, lengths=lengths, data_dir=None if lengths is not None else data_dir
            )
            complete_ids = []
        else:
            analysis_ids = needing_ids
        logger.info(
            "tar_check central analysis: checked=%d complete=%d needing_work=%d elapsed_seconds=%.3f",
            len(check_candidates), len(check_candidates) - len(needing_ids), len(analysis_ids),
            time.perf_counter() - check_start,
        )
    elif args.tar_protein_dirs:
        if args.dynamic_resharding and shard_index is not None:
            analysis_ids = shard_proteins(
                global_protein_ids, shard_index, num_shards, lengths=lengths, data_dir=None if lengths is not None else data_dir
            )
        logger.info(
            "tar_check central analysis: checked=%d complete=0 needing_work=%d elapsed_seconds=0.000",
            len(analysis_ids), len(analysis_ids),
        )

    if args.tar_protein_dirs:
        stats = restore_selected_protein_dirs(args.inference_dir, analysis_ids)
        logger.info("tar_restore central analysis: %s", stats)

    results = compute_analysis_for_proteins(
        inference_dir=args.inference_dir,
        protein_ids=analysis_ids,
        output_subdir=args.output_subdir,
        proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
        skip_existing=not args.rerun,
        num_workers=args.num_workers,
        cif_dir=args.cif_dir.strip() or None,
        use_usalign_dir=not args.no_usalign_dir,
        skip_diversity=args.skip_diversity,
    )
    for protein_id in complete_ids:
        summary_text = read_protein_text(
            args.inference_dir,
            protein_id,
            Path(args.output_subdir) / f"analysis_summary_{protein_id}.json",
        )
        if summary_text is not None:
            results[protein_id] = json.loads(summary_text)
    if args.tar_protein_dirs:
        stats = pack_protein_dirs(args.inference_dir, analysis_ids, delete_after=True)
        logger.info("tar_pack central analysis finalization: %s", stats)
    logger.info(f"Done. {len(results)} proteins with analysis metrics.")


if __name__ == "__main__":
    main()
