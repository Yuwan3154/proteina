#!/usr/bin/env python3
"""Post-prediction sampling convergence analysis for Proteina.

For each protein, compute the maximum AF2Rank performance achievable using only
the first N samples, for N in a configurable set of cutoffs (default
128, 256, 512, 1024).  Reads tarred per-protein outputs produced by
``run_prediction_pipeline.py``.

When a cutoff window contains no globally-top-k template (so no cached AF2Rank
score for any sample in that window), this script runs AF2Rank on the minimum
energy template within that window and appends the score to the existing per
protein AF2Rank score CSVs (m1 and m2).  Those rows are *additive* — the
canonical top-k summary CSVs/JSONs filter on the original top-k filenames, so
new rows do not perturb the main analysis.

Two modes (mirroring ``run_prediction_pipeline.py``):
- Prediction mode: primary metric ``min_ptm`` = ``min(ptm_m1, ptm_m2)``.
- Evaluation mode (``--cif_dir`` set): adds a separate set of plots driven by
  ``tm_ref_pred`` (AF2Rank predicted structure vs ground-truth CIF).
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proteinfoundation.prediction_pipeline.protein_tar_utils import (
    list_protein_members,
    pack_protein_dirs,
    protein_tar_path,
    read_protein_text,
    restore_selected_protein_dirs,
)
from proteinfoundation.prediction_pipeline.sharding_utils import (
    add_shard_args,
    default_progress_check_workers,
    filter_proteins_threaded,
    lengths_from_csv,
    resolve_shard_args,
    shard_proteins,
    wait_for_step,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

AF2RANK_TOPK_SUBDIR = "af2rank_on_proteinebm_top_k"
M1_SUBDIR = "af2rank_analysis"
M2_SUBDIR = "af2rank_analysis_model_2_ptm"
PREDICTED_SUBDIR = "predicted_structures"
PER_PROTEIN_CACHE_NAME = "sampling_convergence_summary.json"

PRIMARY_METRIC = "max_min_ptm"
SECONDARY_METRIC = "max_min_tm_ref_pred"
METRIC_LABELS = {
    "max_min_ptm": "max min(pTM)",
    "max_min_tm_ref_pred": "max TM(pred vs GT)",
}
PLOT_PREFIX = {
    "max_min_ptm": "convergence_min_ptm",
    "max_min_tm_ref_pred": "convergence_tm_ref_pred",
}


# ---------------------------------------------------------------------------
# Read-only per-protein helpers
# ---------------------------------------------------------------------------

def _parse_sample_index(structure_file: str) -> Optional[int]:
    """Parse Proteina sample index from the last '_'-delimited token of the stem."""
    stem = Path(str(structure_file)).stem
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1]
    if not tail.lstrip("-").isdigit():
        return None
    return int(tail)


def _read_energy_df(
    inference_dir: Path, protein_id: str, proteinebm_subdir: str
) -> Optional[pd.DataFrame]:
    """Read the ProteinEBM scores CSV (all samples) from tar/loose dir."""
    rel = Path(proteinebm_subdir) / f"proteinebm_scores_{protein_id}.csv"
    text = read_protein_text(inference_dir, protein_id, rel)
    if text is None:
        return None
    df = pd.read_csv(io.StringIO(text))
    for col in ("structure_file", "structure_path", "energy"):
        if col not in df.columns:
            return None
    df = df.dropna(subset=["structure_file", "structure_path", "energy"]).copy()
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df = df.dropna(subset=["energy"]).copy()
    df["sample_index"] = df["structure_file"].apply(_parse_sample_index)
    df = df.dropna(subset=["sample_index"]).copy()
    df["sample_index"] = df["sample_index"].astype(int)
    return df


def _read_af2rank_scores(
    inference_dir: Path, protein_id: str
) -> Optional[pd.DataFrame]:
    """Read m1+m2 AF2Rank score CSVs from tar/loose dir; inner-join; compute min_*.

    Returns DataFrame keyed by ``structure_file`` with columns
    ``ptm_m1, ptm_m2, min_ptm, composite_m1, composite_m2, min_composite,
    plddt_m1, plddt_m2, tm_ref_pred_m1, tm_ref_pred_m2, min_tm_ref_pred``.
    """
    m1_rel = Path(AF2RANK_TOPK_SUBDIR) / M1_SUBDIR / f"af2rank_scores_{protein_id}.csv"
    m2_rel = Path(AF2RANK_TOPK_SUBDIR) / M2_SUBDIR / f"af2rank_scores_{protein_id}.csv"
    m1_text = read_protein_text(inference_dir, protein_id, m1_rel)
    m2_text = read_protein_text(inference_dir, protein_id, m2_rel)
    if m1_text is None or m2_text is None:
        return None
    m1_df = pd.read_csv(io.StringIO(m1_text))
    m2_df = pd.read_csv(io.StringIO(m2_text))
    if "structure_file" not in m1_df.columns or "structure_file" not in m2_df.columns:
        return None

    keep = ["structure_file", "ptm", "composite", "plddt", "tm_ref_pred"]

    def _safe(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in keep if c in df.columns]
        return df[cols].copy()

    m1 = _safe(m1_df).rename(
        columns={c: f"{c}_m1" for c in keep if c != "structure_file"}
    )
    m2 = _safe(m2_df).rename(
        columns={c: f"{c}_m2" for c in keep if c != "structure_file"}
    )
    merged = m1.merge(m2, on="structure_file", how="inner")
    if merged.empty:
        return None
    for metric in ("ptm", "composite", "plddt", "tm_ref_pred"):
        c1, c2 = f"{metric}_m1", f"{metric}_m2"
        if c1 in merged.columns and c2 in merged.columns:
            merged[c1] = pd.to_numeric(merged[c1], errors="coerce")
            merged[c2] = pd.to_numeric(merged[c2], errors="coerce")
            merged[f"min_{metric}"] = merged[[c1, c2]].min(axis=1)
    return merged


def _empty_cutoff_result(cutoff: int) -> Dict[str, Any]:
    nan = float("nan")
    return {
        "cutoff": int(cutoff),
        "n_samples_in_cutoff": 0,
        "n_scored_in_cutoff": 0,
        "fallback_used": False,
        "fallback_sample_index": None,
        "fallback_energy": None,
        "max_min_ptm": nan,
        "max_ptm_m1": nan,
        "max_ptm_m2": nan,
        "max_min_composite": nan,
        "max_min_tm_ref_pred": nan,
        "best_structure_file_min_ptm": "",
        "best_structure_file_min_tm_ref_pred": "",
    }


def _fill_max_metrics(result: Dict[str, Any], scored: pd.DataFrame) -> None:
    """Populate ``max_*`` fields from a DataFrame of AF2Rank-scored templates."""
    if scored.empty:
        return
    for key, col in (
        ("max_min_ptm", "min_ptm"),
        ("max_ptm_m1", "ptm_m1"),
        ("max_ptm_m2", "ptm_m2"),
        ("max_min_composite", "min_composite"),
        ("max_min_tm_ref_pred", "min_tm_ref_pred"),
    ):
        if col in scored.columns:
            arr = pd.to_numeric(scored[col], errors="coerce").dropna()
            if len(arr) > 0:
                result[key] = float(arr.max())
    if "min_ptm" in scored.columns:
        s = pd.to_numeric(scored["min_ptm"], errors="coerce")
        if s.notna().any():
            result["best_structure_file_min_ptm"] = str(scored.loc[s.idxmax(), "structure_file"])
    if "min_tm_ref_pred" in scored.columns:
        s = pd.to_numeric(scored["min_tm_ref_pred"], errors="coerce")
        if s.notna().any():
            result["best_structure_file_min_tm_ref_pred"] = str(
                scored.loc[s.idxmax(), "structure_file"]
            )


def _analyze_protein_readonly(
    inference_dir: Path,
    protein_id: str,
    proteinebm_subdir: str,
    cutoffs: List[int],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Read-only analysis for one protein.

    Returns:
        per_protein_cache:  dict with cutoff results (or None on read failure)
        fallback_items:     list of templates that need new AF2Rank scoring,
                            one per cutoff that falls into Case B
        status:             ``None`` on success, otherwise a short error tag
    """
    energy_df = _read_energy_df(inference_dir, protein_id, proteinebm_subdir)
    if energy_df is None or energy_df.empty:
        return None, [], "missing_or_empty_energy_csv"

    af2_df = _read_af2rank_scores(inference_dir, protein_id)
    scored_files = set()
    if af2_df is not None:
        scored_files = set(af2_df["structure_file"].astype(str))
    has_gt = (
        af2_df is not None
        and "min_tm_ref_pred" in af2_df.columns
        and pd.to_numeric(af2_df["min_tm_ref_pred"], errors="coerce").notna().any()
    )

    total_samples = int(len(energy_df))
    max_sample_index = int(energy_df["sample_index"].max()) if total_samples else -1
    cutoff_results: Dict[int, Dict[str, Any]] = {}
    fallback_items: List[Dict[str, Any]] = []

    for N in cutoffs:
        result = _empty_cutoff_result(N)
        subset = energy_df[energy_df["sample_index"] < N]
        result["n_samples_in_cutoff"] = int(len(subset))
        if len(subset) == 0:
            cutoff_results[N] = result
            continue

        scored_in_subset = set(subset["structure_file"].astype(str)) & scored_files
        result["n_scored_in_cutoff"] = len(scored_in_subset)

        if scored_in_subset:
            assert af2_df is not None
            sub_scores = af2_df[af2_df["structure_file"].astype(str).isin(scored_in_subset)]
            _fill_max_metrics(result, sub_scores)
        else:
            min_row = subset.sort_values("energy", ascending=True).iloc[0]
            result["fallback_used"] = True
            result["fallback_sample_index"] = int(min_row["sample_index"])
            result["fallback_energy"] = float(min_row["energy"])
            fallback_items.append({
                "protein_id": protein_id,
                "structure_file": str(min_row["structure_file"]),
                "structure_path": str(min_row["structure_path"]),
                "sample_index": int(min_row["sample_index"]),
                "energy": float(min_row["energy"]),
                "cutoff": int(N),
            })
        cutoff_results[N] = result

    cache = {
        "protein_id": protein_id,
        "total_samples": total_samples,
        "max_sample_index": max_sample_index,
        "cutoffs": {str(k): v for k, v in cutoff_results.items()},
        "has_gt": bool(has_gt),
    }
    return cache, fallback_items, None


# ---------------------------------------------------------------------------
# Fallback AF2Rank scoring (in-process)
# ---------------------------------------------------------------------------

def _topk_df_for_reference(
    inference_dir: Path, protein_id: str, proteinebm_subdir: str, top_k: int
) -> Optional[pd.DataFrame]:
    """Build an energy-ranked dataframe to feed ``_pick_reference``.

    Mirrors ``run_af2rank_prediction._select_topk_df`` but reads from tar so we
    don't need to restore the protein dir just for reference selection.
    """
    rel = Path(proteinebm_subdir) / f"proteinebm_scores_{protein_id}.csv"
    text = read_protein_text(inference_dir, protein_id, rel)
    if text is None:
        return None
    df = pd.read_csv(io.StringIO(text))
    if "structure_path" not in df.columns or "energy" not in df.columns:
        return None
    df = df.dropna(subset=["structure_path", "energy"]).copy()
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df = df.dropna(subset=["energy"])
    if df.empty:
        return None
    df = df.sort_values("energy").head(top_k).reset_index(drop=True)
    # Rebase paths to the local protein dir (matches _select_topk_df logic)
    protein_dir = inference_dir / protein_id

    def _rebase(p: str) -> str:
        orig = Path(p)
        local = protein_dir / orig.name
        if local.exists():
            return str(local)
        if orig.exists():
            return str(orig)
        return str(local)

    df["structure_path"] = df["structure_path"].astype(str).apply(_rebase)
    return df


def _resolve_local_template_path(inference_dir: Path, protein_id: str, structure_file: str) -> Path:
    """Return the on-disk path for a sample PDB (expects the protein dir to be extracted)."""
    return inference_dir / protein_id / structure_file


def _append_score_row(scores_csv: Path, new_row: Dict[str, Any]) -> None:
    """Append a row to ``scores_csv`` (creating it if missing).

    Existing rows are preserved; if ``structure_file`` already exists, the new
    row replaces it.  Column union is preserved so old columns survive.
    """
    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    if scores_csv.exists():
        existing = pd.read_csv(scores_csv)
    else:
        existing = pd.DataFrame()
    # Drop any prior row for the same structure_file
    if not existing.empty and "structure_file" in existing.columns:
        existing = existing[existing["structure_file"].astype(str) != str(new_row.get("structure_file"))]
    combined = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    combined.to_csv(scores_csv, index=False)


def _run_fallback_scoring(
    inference_dir: Path,
    fallback_by_protein: Dict[str, List[Dict[str, Any]]],
    cif_dir: Optional[str],
    proteinebm_subdir: str,
    top_k: int,
    recycles: int,
    use_deepspeed_evoformer_attention: bool,
    use_cuequivariance_attention: bool,
    use_cuequivariance_multiplicative_update: bool,
    tar_protein_dirs: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run AF2Rank (m1 + m2) on fallback templates.

    Returns: ``{protein_id: {structure_file: {model: row_dict}}}`` for downstream
    re-merging into per-cutoff results.  Also appends the rows into the on-disk
    ``af2rank_scores_{pid}.csv`` files so future runs hit the cache.
    """
    if not fallback_by_protein:
        return {}

    # Re-use existing pipeline utilities (deferred imports — heavy deps)
    from proteinfoundation.prediction_pipeline.run_af2rank_prediction import (
        _batch_reconstruct_all_proteins,
        _persist_allatom_files,
        _pick_reference,
    )

    protein_ids = sorted(fallback_by_protein.keys())
    logger.info("fallback AF2Rank scoring: %d proteins, %d templates total",
                len(protein_ids),
                sum(len(v) for v in fallback_by_protein.values()))

    # Build per-protein topk_df for reference selection (energy-ranked, top_k)
    topk_dfs: Dict[str, pd.DataFrame] = {}
    skipped_no_topk: List[str] = []
    for pid in protein_ids:
        df = _topk_df_for_reference(inference_dir, pid, proteinebm_subdir, top_k)
        if df is None or df.empty:
            skipped_no_topk.append(pid)
        else:
            topk_dfs[pid] = df
    for pid in skipped_no_topk:
        logger.warning("%s: cannot pick AF2Rank reference (no energy CSV); skipping fallback", pid)
        fallback_by_protein.pop(pid, None)
    protein_ids = sorted(fallback_by_protein.keys())
    if not protein_ids:
        return {}

    # Restore tars for proteins needing work (templates and references live in
    # the per-protein directories).  Only when tar layout is active; with loose
    # layout the dirs already exist.
    if tar_protein_dirs:
        restore_stats = restore_selected_protein_dirs(inference_dir, protein_ids)
        logger.info("tar_restore sampling_convergence: %s", restore_stats)

    # Build dedup'd template list per protein and resolve local PDB paths
    per_protein_templates: Dict[str, List[Dict[str, Any]]] = {}
    for pid in protein_ids:
        by_file: Dict[str, Dict[str, Any]] = {}
        for item in fallback_by_protein[pid]:
            by_file.setdefault(item["structure_file"], item)
        templates: List[Dict[str, Any]] = []
        for sf, item in by_file.items():
            local = _resolve_local_template_path(inference_dir, pid, sf)
            if not local.exists():
                logger.warning("%s: template PDB missing on disk after restore: %s", pid, local)
                continue
            templates.append({
                "structure_file": sf,
                "structure_path": str(local),
                "sample_index": item["sample_index"],
                "energy": item["energy"],
            })
        if templates:
            per_protein_templates[pid] = templates
    if not per_protein_templates:
        # Repack and exit
        if tar_protein_dirs:
            pack_protein_dirs(inference_dir, protein_ids, delete_after=True)
        return {}

    # Build protein_configs analogous to run_af2rank_prediction
    protein_configs: List[Dict[str, Any]] = []
    for pid, templates in per_protein_templates.items():
        topk_df = topk_dfs[pid]
        try:
            ref_path, ref_chain = _pick_reference(topk_df, cif_dir, pid)
        except Exception as e:
            logger.warning("%s: reference selection failed: %s; skipping", pid, e)
            continue
        protein_configs.append({
            "protein_id": pid,
            "templates": templates,
            "reference_path": ref_path,
            "chain": ref_chain,
            # topk_df is kept around for cg2all batch reconstruction
            "topk_df": pd.DataFrame([
                {"structure_path": t["structure_path"]} for t in templates
            ]),
        })

    if not protein_configs:
        if tar_protein_dirs:
            pack_protein_dirs(inference_dir, list(per_protein_templates.keys()), delete_after=True)
        return {}

    # Pre-reconstruct CA-only templates once (shared across both model passes)
    allatom_map = _batch_reconstruct_all_proteins(protein_configs)
    allatom_map = _persist_allatom_files(allatom_map, protein_configs, inference_dir)
    has_ground_truth = cif_dir is not None

    # Lazy import scorer (requires the proteina/openfold env)
    from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank

    results: Dict[str, Dict[str, Dict[str, Any]]] = {pid: {} for pid in per_protein_templates}

    def _existing_scored_files(out_subdir: str) -> Dict[str, set]:
        """Return {pid: set of structure_file values already in the AF2Rank CSV}."""
        out: Dict[str, set] = {}
        for cfg in protein_configs:
            pid = cfg["protein_id"]
            scores_csv = inference_dir / pid / AF2RANK_TOPK_SUBDIR / out_subdir / f"af2rank_scores_{pid}.csv"
            if scores_csv.exists():
                try:
                    df = pd.read_csv(scores_csv)
                    out[pid] = set(df["structure_file"].astype(str)) if "structure_file" in df.columns else set()
                except Exception:
                    out[pid] = set()
            else:
                out[pid] = set()
        return out

    def _run_pass(model_name: str, out_subdir: str) -> None:
        # Skip the entire pass if nothing needs scoring (cache hit on rerun).
        already = _existing_scored_files(out_subdir)
        todo = [
            (cfg, [t for t in cfg["templates"] if t["structure_file"] not in already.get(cfg["protein_id"], set())])
            for cfg in protein_configs
        ]
        todo = [(cfg, tpls) for cfg, tpls in todo if tpls]
        if not todo:
            logger.info("AF2Rank %s: all fallback templates already cached, skipping model load",
                        model_name)
            return

        logger.info("Loading AF2Rank %s for fallback scoring (%d proteins, %d templates)...",
                    model_name, len(todo), sum(len(tpls) for _, tpls in todo))
        first_cfg = todo[0][0]
        scorer = OpenFoldAF2Rank(
            reference_pdb=first_cfg["reference_path"],
            chain=first_cfg["chain"],
            model_name=model_name,
            recycles=recycles,
            use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            skip_ref_metrics=not has_ground_truth,
        )
        try:
            for i, (cfg, tpls) in enumerate(todo):
                pid = cfg["protein_id"]
                if i > 0:
                    try:
                        scorer.reset_reference(cfg["reference_path"], cfg["chain"])
                    except Exception as e:
                        logger.error("%s: reset_reference failed: %s; skipping", pid, e)
                        continue
                out_dir = inference_dir / pid / AF2RANK_TOPK_SUBDIR / out_subdir
                predicted_dir = out_dir / PREDICTED_SUBDIR
                predicted_dir.mkdir(parents=True, exist_ok=True)
                scores_csv = out_dir / f"af2rank_scores_{pid}.csv"
                for tpl in tpls:
                    ca_pdb = tpl["structure_path"]
                    prebuilt = allatom_map.get(ca_pdb)
                    scored_pdb = prebuilt if prebuilt and Path(prebuilt).exists() else ca_pdb
                    original_pdb = ca_pdb if scored_pdb != ca_pdb else None
                    pdb_filename = Path(ca_pdb).name
                    output_pdb = str(predicted_dir / pdb_filename)
                    try:
                        row = scorer.score_structure(
                            scored_pdb,
                            decoy_chain="A",
                            recycles=recycles,
                            output_pdb=output_pdb,
                            verbose=False,
                            _original_pdb=original_pdb,
                        )
                    except Exception as e:
                        logger.error("%s/%s (%s): score_structure failed: %s",
                                     pid, pdb_filename, model_name, e)
                        continue
                    row.update({
                        "protein_id": pid,
                        "structure_file": pdb_filename,
                        "structure_path": ca_pdb,
                        "predicted_structure_path": output_pdb,
                        "predicted_structure_file": pdb_filename,
                    })
                    _append_score_row(scores_csv, row)
                    results[pid].setdefault(pdb_filename, {})[model_name] = row
                logger.info("  fallback %s [%d/%d] %s: scored %d templates",
                            model_name, i + 1, len(todo), pid, len(tpls))
        finally:
            del scorer
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    _run_pass("model_1_ptm", M1_SUBDIR)
    _run_pass("model_2_ptm", M2_SUBDIR)
    # Always populate `results` with the set of proteins that had queued
    # fallbacks, so the post-fallback recomputation in main() runs even on a
    # re-run where every score was already cached (skip-cache path above).
    for pid in per_protein_templates.keys():
        results.setdefault(pid, {})
    return results


# ---------------------------------------------------------------------------
# Per-protein cache JSON (lives inside the protein dir)
# ---------------------------------------------------------------------------

def _per_protein_cache_rel_path() -> Path:
    return Path(PER_PROTEIN_CACHE_NAME)


def _read_per_protein_cache(inference_dir: Path, protein_id: str) -> Optional[Dict[str, Any]]:
    text = read_protein_text(inference_dir, protein_id, _per_protein_cache_rel_path())
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _write_per_protein_cache(inference_dir: Path, protein_id: str, cache: Dict[str, Any]) -> None:
    """Write the cache JSON into the (already-extracted) protein dir."""
    protein_dir = inference_dir / protein_id
    if not protein_dir.is_dir():
        logger.warning("%s: cannot write per-protein cache; dir not extracted", protein_id)
        return
    (protein_dir / PER_PROTEIN_CACHE_NAME).write_text(json.dumps(cache, indent=2))


def _cache_matches(cache: Optional[Dict[str, Any]], cutoffs: List[int]) -> bool:
    if not cache:
        return False
    cached_cutoffs = set(int(k) for k in cache.get("cutoffs", {}).keys())
    return set(cutoffs).issubset(cached_cutoffs)


def _cache_has_no_pending_fallbacks(cache: Dict[str, Any], cutoffs: List[int]) -> bool:
    """Every cutoff result is either Case A (n_scored>0) or has filled fallback metrics."""
    for N in cutoffs:
        rec = cache.get("cutoffs", {}).get(str(N))
        if rec is None:
            return False
        if rec.get("n_samples_in_cutoff", 0) == 0:
            continue
        if rec.get("fallback_used"):
            # Fallback finalised iff the max_min_ptm is no longer NaN
            v = rec.get("max_min_ptm")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
    return True


# ---------------------------------------------------------------------------
# Aggregation: long CSV, wide CSV, summary JSON
# ---------------------------------------------------------------------------

def _cache_to_long_rows(cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    pid = cache.get("protein_id")
    rows: List[Dict[str, Any]] = []
    for k, rec in cache.get("cutoffs", {}).items():
        row = {"protein_id": pid, "cutoff": int(k)}
        row.update(rec)
        # cutoff key duplicated by _empty_cutoff_result; canonicalise to int
        row["cutoff"] = int(rec.get("cutoff", k))
        rows.append(row)
    return rows


def _write_long_csv(rows: List[Dict[str, Any]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["protein_id", "cutoff"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))
    return df


def _write_wide_csv(long_df: pd.DataFrame, cutoffs: List[int], path: Path, has_gt: bool) -> None:
    if long_df.empty:
        pd.DataFrame(columns=["protein_id"]).to_csv(path, index=False)
        return
    metrics = ["max_min_ptm"]
    if has_gt:
        metrics.append("max_min_tm_ref_pred")
    wide = long_df.pivot_table(
        index="protein_id",
        columns="cutoff",
        values=metrics,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{int(N)}" for metric, N in wide.columns]
    wide = wide.reset_index().sort_values("protein_id").reset_index(drop=True)
    wide.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows, %d cols)", path, len(wide), wide.shape[1])


def _summary_stats(long_df: pd.DataFrame, cutoffs: List[int], ptm_cutoff: float, tm_cutoff: float, has_gt: bool) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_proteins": int(long_df["protein_id"].nunique()) if not long_df.empty else 0,
        "cutoffs": [int(N) for N in cutoffs],
        "has_gt": bool(has_gt),
        "ptm_cutoff": float(ptm_cutoff),
        "tm_cutoff": float(tm_cutoff),
        "per_cutoff": {},
        "fallback_invocations_per_cutoff": {},
    }
    metric_specs = [("max_min_ptm", ptm_cutoff)]
    if has_gt:
        metric_specs.append(("max_min_tm_ref_pred", tm_cutoff))
    for N in cutoffs:
        block: Dict[str, Any] = {}
        sub = long_df[long_df["cutoff"] == int(N)] if not long_df.empty else long_df
        block["n_proteins"] = int(len(sub))
        for metric, thresh in metric_specs:
            arr = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy() if metric in sub.columns else np.array([])
            stat: Dict[str, Any] = {"n": int(arr.size)}
            if arr.size:
                stat.update({
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std(ddof=0)),
                    "p25": float(np.percentile(arr, 25)),
                    "p75": float(np.percentile(arr, 75)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    f"fraction_ge_{thresh}": float((arr >= thresh).mean()),
                })
            block[metric] = stat
        if not sub.empty and "fallback_used" in sub.columns:
            n_fb = int(sub["fallback_used"].astype(bool).sum())
            summary["fallback_invocations_per_cutoff"][str(int(N))] = n_fb
        else:
            summary["fallback_invocations_per_cutoff"][str(int(N))] = 0
        summary["per_cutoff"][str(int(N))] = block
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_distribution(
    long_df: pd.DataFrame, cutoffs: List[int], metric: str, path: Path,
) -> None:
    """Per-cutoff box plot with mean overlaid as a connecting line.

    Replaces the older separate "violin distribution" + "mean/median + IQR curve"
    pair.  Each cutoff column shows a single box (Q1–Q3 with median line and
    whiskers); the mean across proteins is plotted as a connected line on top,
    so the eye can follow the trend.
    """
    if long_df.empty or metric not in long_df.columns:
        return
    data = [
        pd.to_numeric(long_df[long_df["cutoff"] == int(N)][metric], errors="coerce").dropna().to_numpy()
        for N in cutoffs
    ]
    if not any(len(a) for a in data):
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(cutoffs)), 5), dpi=120)
    positions = np.arange(len(cutoffs)) + 1
    box_data = [d if len(d) else np.array([np.nan]) for d in data]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        showfliers=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4, "markeredgecolor": "none"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    for box in bp["boxes"]:
        box.set(facecolor="tab:blue", alpha=0.40, edgecolor="black")

    # Jittered individual points (low alpha) so distribution shape is visible
    for i, arr in enumerate(data):
        if len(arr):
            x = np.full_like(arr, positions[i], dtype=float)
            x += (np.random.RandomState(0).rand(len(arr)) - 0.5) * 0.30
            ax.scatter(x, arr, s=8, color="black", alpha=0.30, linewidths=0)

    # Mean across proteins, connected by a line
    means = [float(arr.mean()) if len(arr) else np.nan for arr in data]
    ns = [len(arr) for arr in data]
    ax.plot(
        positions, means,
        marker="D", markersize=8, markerfacecolor="tab:red", markeredgecolor="black",
        markeredgewidth=0.8,
        color="tab:red", linewidth=2.0, label="mean",
    )

    # Annotate each cutoff with its protein count
    yshow_min = np.nanmin([np.min(a) for a in data if len(a)])
    for pos, n in zip(positions, ns):
        ax.text(pos, max(0.0, yshow_min - 0.05), f"n={n}", ha="center", va="top", fontsize=8, color="dimgray")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(N)) for N in cutoffs])
    ax.set_xlabel("sample count cutoff (N)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs. sample count")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _plot_fraction_passing(
    long_df: pd.DataFrame, cutoffs: List[int], metric: str, threshold: float, path: Path,
) -> None:
    if long_df.empty or metric not in long_df.columns:
        return
    fracs, ns = [], []
    for N in cutoffs:
        arr = pd.to_numeric(long_df[long_df["cutoff"] == int(N)][metric], errors="coerce").dropna().to_numpy()
        ns.append(len(arr))
        fracs.append(float((arr >= threshold).mean()) if len(arr) else np.nan)
    if not any(n > 0 for n in ns):
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    x = np.asarray(cutoffs, dtype=float)
    ax.plot(x, fracs, marker="o", color="tab:blue", linewidth=2)
    for xi, yi, n in zip(x, fracs, ns):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.2f}\n(n={n})", (xi, yi), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color="black")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("sample count cutoff (N) — log₂ scale")
    ax.set_ylabel(f"fraction with {METRIC_LABELS.get(metric, metric)} ≥ {threshold:g}")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"fraction of proteins passing threshold ({METRIC_LABELS.get(metric, metric)} ≥ {threshold:g})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _plot_gain(
    long_df: pd.DataFrame, cutoffs: List[int], metric: str, path: Path,
) -> None:
    if long_df.empty or metric not in long_df.columns or len(cutoffs) < 2:
        return
    pivot = long_df.pivot_table(index="protein_id", columns="cutoff", values=metric, aggfunc="first")
    pivot = pivot.reindex(columns=[int(N) for N in cutoffs])
    diffs: List[np.ndarray] = []
    labels: List[str] = []
    for prev_N, cur_N in zip(cutoffs[:-1], cutoffs[1:]):
        delta = pd.to_numeric(pivot[int(cur_N)], errors="coerce") - pd.to_numeric(pivot[int(prev_N)], errors="coerce")
        delta = delta.dropna().to_numpy()
        diffs.append(delta)
        labels.append(f"{int(prev_N)}→{int(cur_N)}")
    if not any(len(d) for d in diffs):
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(diffs)), 5), dpi=120)
    positions = np.arange(len(diffs)) + 1
    bp = ax.boxplot(
        [d if len(d) else np.array([np.nan]) for d in diffs],
        positions=positions, widths=0.45, showfliers=True, patch_artist=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="tab:green", alpha=0.45, edgecolor="black")
    for median in bp["medians"]:
        median.set(color="black")
    for i, d in enumerate(diffs):
        if len(d):
            x = np.full_like(d, positions[i], dtype=float)
            x += (np.random.RandomState(0).rand(len(d)) - 0.5) * 0.22
            ax.scatter(x, d, s=10, color="black", alpha=0.35, linewidths=0)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("cutoff transition")
    ax.set_ylabel(f"Δ {METRIC_LABELS.get(metric, metric)}")
    ax.set_title(f"marginal gain in {METRIC_LABELS.get(metric, metric)} per cutoff doubling")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _generate_all_plots(
    long_df: pd.DataFrame,
    cutoffs: List[int],
    has_gt: bool,
    ptm_cutoff: float,
    tm_cutoff: float,
    output_dir: Path,
) -> None:
    metrics = [(PRIMARY_METRIC, ptm_cutoff)]
    if has_gt:
        metrics.append((SECONDARY_METRIC, tm_cutoff))
    for metric, threshold in metrics:
        prefix = PLOT_PREFIX[metric]
        # Main convergence plot: per-cutoff box plot with mean overlaid as a
        # connecting line.  Replaces the older separate "distribution" + "mean
        # curve" pair, and the (visually noisy) heatmap.
        _plot_distribution(long_df, cutoffs, metric, output_dir / f"{prefix}_box_plus_mean.png")
        _plot_fraction_passing(long_df, cutoffs, metric, threshold, output_dir / f"{prefix}_fraction_passing.png")
        _plot_gain(long_df, cutoffs, metric, output_dir / f"{prefix}_gain.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_protein_ids(csv_file: str, id_col: str) -> List[str]:
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    if id_col not in df.columns:
        raise KeyError(f"CSV missing column '{id_col}'. Available: {sorted(df.columns)}")
    return [str(v).strip() for v in df[id_col].dropna().unique() if str(v).strip()]


def _parse_cutoffs(spec: str) -> List[int]:
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("--cutoffs must contain at least one integer")
    return sorted(set(out))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sampling convergence analysis: max performance vs. sample count cutoff."
    )
    p.add_argument("--inference_dir", required=True,
                   help="Base inference directory (per-protein <id>.tar archives or loose dirs).")
    p.add_argument("--csv_file", required=True,
                   help="CSV listing protein IDs to analyze.")
    p.add_argument("--id_col", default="id", help="Protein ID column (default: id).")
    p.add_argument("--cif_dir", default=None,
                   help="Optional: directory of ground-truth CIFs. Enables tm_ref_pred plots.")
    p.add_argument("--cutoffs", default="128,256,512,1024",
                   help="Comma-separated sample-count cutoffs (default: 128,256,512,1024).")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for aggregate CSVs/JSON/plots.")
    p.add_argument("--af2rank_top_k", "--top_k", dest="af2rank_top_k", type=int, default=5,
                   help="Top-k size used during original prediction pipeline run (default: 5).")
    p.add_argument("--recycles", type=int, default=3,
                   help="AF2Rank recycles for fallback scoring (default: 3).")
    p.add_argument("--ptm_cutoff", type=float, default=0.7,
                   help="pTM threshold for 'fraction passing' plot (default: 0.7).")
    p.add_argument("--tm_cutoff", type=float, default=0.5,
                   help="TM-score threshold for 'fraction passing' plot (default: 0.5).")
    p.add_argument("--proteinebm_analysis_subdir", default="proteinebm_v2_cathmd_analysis",
                   help="Per-protein subdir containing proteinebm_scores_*.csv.")
    p.add_argument("--skip_af2rank", action="store_true",
                   help="Skip new AF2Rank scoring; cutoffs without cached top-k stay NaN.")
    p.add_argument("--filter_existing", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip proteins whose per-protein cache already covers all cutoffs (default: True).")
    p.add_argument("--use_deepspeed_evoformer_attention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_cuequivariance_attention", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_cuequivariance_multiplicative_update", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tar_protein_dirs", action=argparse.BooleanOptionalAction, default=True,
                   help="Per-protein outputs are stored as uncompressed <pid>.tar (default: True).")
    p.add_argument("--progress_check_workers", type=int, default=default_progress_check_workers(),
                   help="Threads for I/O-bound progress checks.")
    add_shard_args(p)
    p.add_argument("--shard_poll_interval", type=int, default=60,
                   help="Seconds between polls when shard 0 waits for peers (default: 60).")
    p.add_argument("--shard_timeout", type=int, default=86400,
                   help="Max seconds for shard 0 to wait (default: 86400).")
    return p


def _shard_long_csv_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return output_dir / f".shard_long_{shard_index}_of_{num_shards}.csv"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    inference_dir = Path(args.inference_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = args.cif_dir.strip() if args.cif_dir else None
    has_gt_mode = bool(cif_dir)
    cutoffs = _parse_cutoffs(args.cutoffs)
    logger.info("Cutoffs: %s; mode: %s",
                cutoffs, "evaluation (with GT)" if has_gt_mode else "prediction-only")

    global_protein_ids = _load_protein_ids(args.csv_file, args.id_col)
    logger.info("Loaded %d protein IDs from %s", len(global_protein_ids), args.csv_file)

    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)
    sharded = shard_index is not None and num_shards is not None
    if sharded:
        lengths = lengths_from_csv(args.csv_file, args.id_col, args.len_col)
        my_ids = shard_proteins(
            global_protein_ids, shard_index, num_shards,
            lengths=lengths,
            data_dir=None if lengths is not None else os.environ.get("DATA_PATH"),
        )
        logger.info("Shard %d/%d: %d proteins", shard_index, num_shards, len(my_ids))
    else:
        my_ids = list(global_protein_ids)

    # ── Phase 1: filter to proteins needing work (per-protein cache check) ──
    def _is_complete(pid: str) -> bool:
        if not args.filter_existing:
            return False
        cache = _read_per_protein_cache(inference_dir, pid)
        if not _cache_matches(cache, cutoffs):
            return False
        if not _cache_has_no_pending_fallbacks(cache, cutoffs):
            return False
        # If filter requires GT but cache has no GT, still consider it "complete"
        # for cached results; the aggregation step will see has_gt=False rows.
        return True

    if args.filter_existing:
        needing, done = filter_proteins_threaded(
            my_ids, _is_complete, max_workers=args.progress_check_workers,
        )
        logger.info("filter_existing: %d cached, %d need work", len(done), len(needing))
    else:
        needing, done = list(my_ids), []

    # ── Phase 2: per-protein read-only analysis ─────────────────────────────
    all_rows: List[Dict[str, Any]] = []
    fallback_by_protein: Dict[str, List[Dict[str, Any]]] = {}
    per_protein_caches: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}

    for pid in needing:
        cache, fb, status = _analyze_protein_readonly(
            inference_dir, pid, args.proteinebm_analysis_subdir, cutoffs,
        )
        if status is not None:
            skipped[pid] = status
            logger.warning("%s: skipping (%s)", pid, status)
            continue
        per_protein_caches[pid] = cache
        if fb and not args.skip_af2rank:
            fallback_by_protein[pid] = fb
        elif fb and args.skip_af2rank:
            # Record but don't queue; cutoffs stay NaN
            logger.info("%s: %d cutoff(s) need fallback AF2Rank but --skip_af2rank set; leaving NaN",
                        pid, len(fb))

    # Load cached results for proteins we already considered complete
    for pid in done:
        cache = _read_per_protein_cache(inference_dir, pid)
        if cache is None:
            logger.warning("%s: marked complete but cache read returned None; re-analyzing", pid)
            cache, fb, status = _analyze_protein_readonly(
                inference_dir, pid, args.proteinebm_analysis_subdir, cutoffs,
            )
            if status is not None:
                skipped[pid] = status
                continue
            if fb and not args.skip_af2rank:
                fallback_by_protein[pid] = fb
        per_protein_caches[pid] = cache

    # ── Phase 3: run fallback AF2Rank scoring if anything was queued ────────
    if fallback_by_protein:
        fallback_results = _run_fallback_scoring(
            inference_dir=inference_dir,
            fallback_by_protein=fallback_by_protein,
            cif_dir=cif_dir,
            proteinebm_subdir=args.proteinebm_analysis_subdir,
            top_k=args.af2rank_top_k,
            recycles=args.recycles,
            use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=args.use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
            tar_protein_dirs=args.tar_protein_dirs,
        )
        # After scoring, re-read AF2Rank CSVs and recompute the max metrics
        # for EVERY cutoff (not just the fallback ones).  A fallback template
        # scored for cutoff N also has sample_index < N′ for all larger N′ in
        # the cutoff list, so it should contribute to those larger cutoffs too
        # (otherwise monotonicity can be violated when the newly-scored
        # fallback template outperforms the original top-k entries that lie
        # within the larger window).
        for pid in fallback_results.keys():
            cache = per_protein_caches.get(pid)
            if cache is None:
                continue
            energy_df = _read_energy_df(inference_dir, pid, args.proteinebm_analysis_subdir)
            af2_df = _read_af2rank_scores(inference_dir, pid)
            if energy_df is None or af2_df is None:
                continue
            scored_files = set(af2_df["structure_file"].astype(str))
            for N_key, rec in cache["cutoffs"].items():
                N = int(N_key)
                subset = energy_df[energy_df["sample_index"] < N]
                if subset.empty:
                    continue
                scored_in_subset = set(subset["structure_file"].astype(str)) & scored_files
                if not scored_in_subset:
                    # Still no cached templates within this cutoff — fallback
                    # for this cutoff must have failed; leave the record as-is.
                    continue
                sub_scores = af2_df[af2_df["structure_file"].astype(str).isin(scored_in_subset)]
                # Reset max fields so _fill_max_metrics starts from a clean slate
                for key in ("max_min_ptm", "max_ptm_m1", "max_ptm_m2",
                            "max_min_composite", "max_min_tm_ref_pred"):
                    rec[key] = float("nan")
                rec["best_structure_file_min_ptm"] = ""
                rec["best_structure_file_min_tm_ref_pred"] = ""
                _fill_max_metrics(rec, sub_scores)
                rec["n_scored_in_cutoff"] = len(scored_in_subset)
                # fallback_used remains True for cutoffs that originally
                # needed a fallback; the (now-cached) template is what the
                # max metric was derived from.
            # has_gt may have changed: re-evaluate
            cache["has_gt"] = bool(
                "min_tm_ref_pred" in af2_df.columns
                and pd.to_numeric(af2_df["min_tm_ref_pred"], errors="coerce").notna().any()
            )
            _write_per_protein_cache(inference_dir, pid, cache)
        # Re-tar the protein dirs we restored for fallback scoring
        if args.tar_protein_dirs:
            scored_pids = sorted(fallback_by_protein.keys())
            stats = pack_protein_dirs(inference_dir, scored_pids, delete_after=True)
            logger.info("tar_pack sampling_convergence: %s", stats)
    else:
        # No fallbacks queued: persist any freshly computed caches.
        cache_to_persist = [pid for pid in needing if pid in per_protein_caches]
        if cache_to_persist and args.tar_protein_dirs:
            stats = restore_selected_protein_dirs(inference_dir, cache_to_persist)
            logger.info("tar_restore sampling_convergence (cache-only): %s", stats)
            for pid in cache_to_persist:
                _write_per_protein_cache(inference_dir, pid, per_protein_caches[pid])
            stats = pack_protein_dirs(inference_dir, cache_to_persist, delete_after=True)
            logger.info("tar_pack sampling_convergence (cache-only): %s", stats)
        elif cache_to_persist:
            # Loose layout — write only into dirs that exist on disk.
            for pid in cache_to_persist:
                if (inference_dir / pid).is_dir():
                    _write_per_protein_cache(inference_dir, pid, per_protein_caches[pid])

    # ── Phase 4: assemble long rows for this shard's proteins ───────────────
    for pid, cache in per_protein_caches.items():
        all_rows.extend(_cache_to_long_rows(cache))

    # ── Phase 5: shard fan-in or single-process output ──────────────────────
    if sharded:
        shard_csv = _shard_long_csv_path(output_dir, shard_index, num_shards)
        pd.DataFrame(all_rows).to_csv(shard_csv, index=False)
        logger.info("Shard %d/%d: wrote %s (%d rows)", shard_index, num_shards, shard_csv, len(all_rows))
        if shard_index != 0:
            # Mark shard complete and exit; shard 0 does aggregation.
            ok = wait_for_step(
                output_dir, "sampling_convergence", num_shards, shard_index,
                success=True, poll_interval=args.shard_poll_interval,
                timeout=args.shard_timeout,
            )
            return 0 if ok else 1
        # Shard 0 waits for peers, then aggregates
        ok = wait_for_step(
            output_dir, "sampling_convergence", num_shards, shard_index,
            success=True, poll_interval=args.shard_poll_interval,
            timeout=args.shard_timeout,
        )
        if not ok:
            return 1
        all_rows = []
        for i in range(num_shards):
            shard_path = _shard_long_csv_path(output_dir, i, num_shards)
            if not shard_path.exists():
                logger.warning("Shard CSV missing: %s", shard_path)
                continue
            df = pd.read_csv(shard_path)
            all_rows.extend(df.to_dict("records"))

    # ── Phase 6: aggregate outputs (long, wide, summary, plots) ─────────────
    long_path = output_dir / "sampling_convergence_long.csv"
    wide_path = output_dir / "sampling_convergence_wide.csv"
    summary_path = output_dir / "sampling_convergence_summary.json"

    long_df = _write_long_csv(all_rows, long_path)
    # has_gt at aggregate level: any row populated tm_ref_pred
    agg_has_gt = bool(
        not long_df.empty
        and "max_min_tm_ref_pred" in long_df.columns
        and pd.to_numeric(long_df["max_min_tm_ref_pred"], errors="coerce").notna().any()
    )
    if has_gt_mode and not agg_has_gt:
        logger.warning(
            "--cif_dir was set but no max_min_tm_ref_pred values are populated; "
            "GT plots will be skipped. Check that AF2Rank score CSVs include tm_ref_pred."
        )
    _write_wide_csv(long_df, cutoffs, wide_path, agg_has_gt)
    summary = _summary_stats(long_df, cutoffs, args.ptm_cutoff, args.tm_cutoff, agg_has_gt)
    summary["n_skipped"] = len(skipped)
    summary["skipped_reasons"] = {k: v for k, v in list(skipped.items())[:50]}
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)
    _generate_all_plots(long_df, cutoffs, agg_has_gt, args.ptm_cutoff, args.tm_cutoff, output_dir)
    logger.info("Sampling convergence analysis complete: %d proteins, %d cutoffs",
                long_df["protein_id"].nunique() if not long_df.empty else 0, len(cutoffs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
