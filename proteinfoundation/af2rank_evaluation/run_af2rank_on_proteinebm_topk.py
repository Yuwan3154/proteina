#!/usr/bin/env python3
"""
Run AF2Rank on the top-k ProteinEBM-ranked templates (per protein), then generate
cross-protein plots:
  - Reference TM score vs ProteinEBM energy (best/min energy)
  - Reference TM score vs AF2Rank pTM (best/max pTM within the top-k templates)

This is intentionally separate from run_full_pipeline.py for now.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from proteinfoundation.af2rank_evaluation.sharding_utils import (
    add_shard_args,
    resolve_shard_args,
    shard_proteins,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _load_dataset_map(dataset_file: str, id_column: str, tms_column: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(dataset_file)
    needed = {id_column, tms_column, "in_train", "length"}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise KeyError(f"Dataset missing columns {missing}. Columns: {sorted(df.columns.tolist())}")

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        protein_id = str(row[id_column])
        out[protein_id] = {
            "reference_tm": float(row[tms_column]),
            "in_train": bool(row["in_train"]),
            "length": float(row["length"]),
        }
    return out


def _find_proteinebm_scores(inference_dir: str, analysis_subdir: str = "proteinebm_v2_cathmd_analysis") -> List[Path]:
    base = Path(inference_dir)
    return sorted(base.glob(f"*/{analysis_subdir}/proteinebm_scores_*.csv"))


def _find_reference_cif(protein_id: str, cif_dir: str) -> Optional[str]:
    """Find the reference CIF file for a protein in cif_dir (same logic as parallel_af2rank_scoring)."""
    pdb_id = protein_id.split("_")[0]
    cif_path = Path(cif_dir)
    for subdir in cif_path.iterdir():
        if subdir.is_dir():
            potential = subdir / f"{pdb_id}.cif"
            if potential.exists():
                return str(potential)
    potential = cif_path / f"{pdb_id}.cif"
    if potential.exists():
        return str(potential)
    return None


def _read_proteinebm_summary(summary_path: Path) -> Dict[str, str | None]:
    with summary_path.open("r") as f:
        summary = json.load(f)
    ref = summary.get("reference_structure")
    chain = summary.get("chain")
    return {
        "reference_structure": str(ref) if ref else None,
        "chain": str(chain) if chain else None,
    }


def _select_topk_templates(scores_csv: Path, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    needed = {"structure_path", "energy"}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise KeyError(f"ProteinEBM scores CSV missing columns {missing}: {scores_csv}")
    df = df.dropna(subset=["structure_path", "energy"])
    df["energy"] = df["energy"].astype(float)
    df = df.sort_values("energy", ascending=True).head(int(top_k)).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No valid rows found in {scores_csv}")

    protein_dir = scores_csv.parent.parent
    def _rebase_path(p: str) -> str:
        """Resolve structure_path relative to the protein dir on the current machine."""
        orig = Path(p)
        local = protein_dir / orig.name
        if local.exists():
            return str(local)
        if orig.exists():
            return str(orig)
        return str(local)
    df["structure_path"] = df["structure_path"].astype(str).apply(_rebase_path)
    return df


def _extract_top1_top5_metrics(
    af2_df: pd.DataFrame, staged_filenames: set
) -> Dict[str, float]:
    """Extract top-1 (by pTM) and top-5 (best tm_ref_pred among top-5 by pTM) metrics.
    Returns dict with ptm, tm_ref_pred, composite, plddt for top_1 and top_5."""
    nan = float("nan")
    out = {
        "top_1_ptm": nan, "top_1_tm_ref_pred": nan, "top_1_composite": nan, "top_1_plddt": nan,
        "top_5_ptm": nan, "top_5_tm_ref_pred": nan, "top_5_composite": nan, "top_5_plddt": nan,
    }
    needed = {"structure_file", "ptm", "tm_ref_pred"}
    if not needed.issubset(af2_df.columns):
        return out
    df = af2_df[af2_df["structure_file"].astype(str).isin(staged_filenames)].copy()
    df = df.dropna(subset=["ptm", "tm_ref_pred"])
    if len(df) == 0:
        return out
    df["ptm"] = pd.to_numeric(df["ptm"], errors="coerce")
    df["tm_ref_pred"] = pd.to_numeric(df["tm_ref_pred"], errors="coerce")
    if "composite" in df.columns:
        df["composite"] = pd.to_numeric(df["composite"], errors="coerce")
    if "plddt" in df.columns:
        df["plddt"] = pd.to_numeric(df["plddt"], errors="coerce")
    df = df.dropna(subset=["ptm", "tm_ref_pred"])
    if len(df) == 0:
        return out
    df = df.sort_values("ptm", ascending=False).reset_index(drop=True)
    r1 = df.iloc[0]
    out["top_1_ptm"] = float(r1["ptm"])
    out["top_1_tm_ref_pred"] = float(r1["tm_ref_pred"])
    out["top_1_composite"] = float(r1["composite"]) if "composite" in r1 and pd.notna(r1.get("composite")) else nan
    out["top_1_plddt"] = float(r1["plddt"]) if "plddt" in r1 and pd.notna(r1.get("plddt")) else nan
    top5 = df.head(5)
    best_idx = int(top5["tm_ref_pred"].argmax()) if len(top5) > 0 else 0
    r5 = top5.iloc[best_idx]
    out["top_5_ptm"] = float(r5["ptm"])
    out["top_5_tm_ref_pred"] = float(r5["tm_ref_pred"])
    out["top_5_composite"] = float(r5["composite"]) if "composite" in r5 and pd.notna(r5.get("composite")) else nan
    out["top_5_plddt"] = float(r5["plddt"]) if "plddt" in r5 and pd.notna(r5.get("plddt")) else nan
    return out


def _merge_min_across_models(
    m1_df: pd.DataFrame, m2_df: pd.DataFrame, staged_filenames: set
) -> pd.DataFrame:
    """Merge model_1 and model_2 AF2Rank results, taking min of each metric per template."""
    needed = {"structure_file", "ptm", "tm_ref_pred"}
    for df in (m1_df, m2_df):
        if not needed.issubset(df.columns):
            return pd.DataFrame(columns=list(needed) + ["composite", "plddt"])
    m1 = m1_df[m1_df["structure_file"].astype(str).isin(staged_filenames)].copy()
    m2 = m2_df[m2_df["structure_file"].astype(str).isin(staged_filenames)].copy()
    for c in ["composite", "plddt"]:
        if c not in m1.columns:
            m1[c] = float("nan")
        if c not in m2.columns:
            m2[c] = float("nan")
    merged = m1.merge(
        m2,
        on="structure_file",
        how="inner",
        suffixes=("_m1", "_m2"),
    )
    merged["ptm"] = merged[["ptm_m1", "ptm_m2"]].min(axis=1)
    merged["tm_ref_pred"] = merged[["tm_ref_pred_m1", "tm_ref_pred_m2"]].min(axis=1)
    merged["composite"] = merged[["composite_m1", "composite_m2"]].min(axis=1)
    merged["plddt"] = merged[["plddt_m1", "plddt_m2"]].min(axis=1)
    return merged[["structure_file", "ptm", "tm_ref_pred", "composite", "plddt"]]


def _stage_templates_as_dir(topk_df: pd.DataFrame, staged_dir: Path) -> None:
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    for _, row in topk_df.iterrows():
        src = Path(str(row["structure_path"]))
        if not src.exists():
            raise FileNotFoundError(f"Template file not found: {src}")
        dst = staged_dir / src.name
        os.symlink(str(src), str(dst))


def _batch_reconstruct_ca_only(staged_dir: Path, direct_python: bool = False) -> Dict[str, str]:
    """Reconstruct all CA-only PDBs in staged_dir via cg2all (single model load).
    Returns mapping {original_path: reconstructed_path}. Empty dict if none are CA-only."""
    pdb_files = sorted(staged_dir.glob("*.pdb"))
    if not pdb_files:
        return {}

    # Quick CA-only check: any file where all ATOM records are CA
    def _is_ca_only(path: Path) -> bool:
        has_non_ca = False
        has_atom = False
        with open(path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    has_atom = True
                    if line[12:16].strip() != "CA":
                        has_non_ca = True
                        break
        return has_atom and not has_non_ca

    ca_only = [str(p) for p in pdb_files if _is_ca_only(p)]
    if not ca_only:
        return {}

    tmp_dir = tempfile.mkdtemp(prefix="cg2all_topk_")
    inputs_json = os.path.join(tmp_dir, "inputs.json")
    output_map_json = os.path.join(tmp_dir, "output_map.json")
    with open(inputs_json, "w") as f:
        json.dump(ca_only, f)

    script = str(Path(__file__).parent / "cg2all_reconstruct.py")
    if direct_python:
        cmd = [sys.executable, script, "--inputs", inputs_json, "--output_dir", tmp_dir, "--output_map", output_map_json]
    else:
        wrapper_script = str(Path(__file__).parent / "run_with_proteina_env.sh")
        cmd = [wrapper_script, "python", script, "--inputs", inputs_json, "--output_dir", tmp_dir, "--output_map", output_map_json]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        err_tail = result.stderr[-3000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"cg2all reconstruction failed:\n{err_tail}")
    with open(output_map_json) as f:
        mapping = json.load(f)
    return mapping


def _run_af2rank_subprocess(
    protein_id: str,
    reference_cif: str,
    chain: str,
    staged_templates_dir: Path,
    output_dir: Path,
    recycles: int,
    cuda_visible_devices: str,
    model_name: str = "model_1_ptm",
    backend: str = "colabdesign",
    allatom_map: Optional[Dict[str, str]] = None,
    direct_python: bool = False,
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = True,
    use_cuequivariance_multiplicative_update: bool = True,
) -> None:
    # Ensure each subprocess is pinned to a single GPU when requested.
    cuda_line = ""
    if cuda_visible_devices.strip():
        cuda_line = f"os.environ['CUDA_VISIBLE_DEVICES'] = {cuda_visible_devices!r}\n"

    wrapper_dir = str(Path(__file__).parent)
    if backend == "openfold":
        wrapper_script = os.path.join(wrapper_dir, "run_with_proteina_env.sh")
        py = f"""
import os
import sys
import glob
sys.path.append({wrapper_dir!r})

import pandas as pd

from af2rank_openfold_scorer import OpenFoldAF2Rank, plot_af2rank_results, save_af2rank_scores, load_af2rank_scores_from_csv

{cuda_line}

protein_id = {protein_id!r}
reference_cif = {reference_cif!r}
chain = {chain!r}
inference_output_dir = {str(staged_templates_dir)!r}
output_dir = {str(output_dir)!r}

scores_csv_path = os.path.join(output_dir, f"af2rank_scores_{{protein_id}}.csv")

existing_scores = []
processed_files = set()
if os.path.exists(scores_csv_path):
  existing_scores = load_af2rank_scores_from_csv(scores_csv_path)
  for s in existing_scores:
    sf = s.get("structure_file")
    if sf:
      processed_files.add(str(sf))

pdb_files = sorted(glob.glob(os.path.join(inference_output_dir, "*.pdb")))
to_score = [p for p in pdb_files if os.path.basename(p) not in processed_files]

new_scores = []
if len(to_score) > 0:
  scorer = OpenFoldAF2Rank(reference_cif, chain=chain, model_name={model_name!r}, recycles={int(recycles)},
    use_deepspeed_evoformer_attention={use_deepspeed_evoformer_attention},
    use_cuequivariance_attention={use_cuequivariance_attention},
    use_cuequivariance_multiplicative_update={use_cuequivariance_multiplicative_update})
  for pdb_path in to_score:
    pdb_filename = os.path.basename(pdb_path)
    structure_scores = scorer.score_structure(
      pdb_path,
      decoy_chain="A",
      recycles={int(recycles)},
      verbose=False,
    )
    structure_scores.update({{
      "protein_id": protein_id,
      "structure_file": pdb_filename,
      "structure_path": pdb_path,
    }})
    if "pred_coords" in structure_scores:
      del structure_scores["pred_coords"]
    new_scores.append(structure_scores)

all_scores = existing_scores + new_scores
save_af2rank_scores(all_scores, output_dir, protein_id)
staged_filenames = set(os.path.basename(p) for p in pdb_files)
plot_scores = [s for s in all_scores if s.get("structure_file") in staged_filenames]
plot_af2rank_results(plot_scores, output_dir, protein_id)
"""
    else:
        wrapper_script = os.path.join(wrapper_dir, "run_with_colabdesign_env.sh")
        allatom_map_repr = repr(allatom_map or {})
        py = f"""
import os
import sys
import glob
sys.path.append({wrapper_dir!r})

import pandas as pd

from af2rank_scorer import ModernAF2Rank, plot_af2rank_results, save_af2rank_scores, suppress_stdout, load_af2rank_scores_from_csv

{cuda_line}
os.environ['ALPHAFOLD_DATA_DIR'] = os.path.expanduser('~/openfold/openfold/resources/params')
os.environ['AF_PARAMS_DIR'] = os.path.expanduser('~/openfold/openfold/resources/params')

protein_id = {protein_id!r}
reference_cif = {reference_cif!r}
chain = {chain!r}
inference_output_dir = {str(staged_templates_dir)!r}
output_dir = {str(output_dir)!r}
# Pre-built cg2all reconstruction map (original_path -> allatom_path)
allatom_map = {allatom_map_repr}

scores_csv_path = os.path.join(output_dir, f"af2rank_scores_{{protein_id}}.csv")

existing_scores = []
processed_files = set()
if os.path.exists(scores_csv_path):
  existing_scores = load_af2rank_scores_from_csv(scores_csv_path)
  for s in existing_scores:
    sf = s.get("structure_file")
    if sf:
      processed_files.add(str(sf))

pdb_files = sorted(glob.glob(os.path.join(inference_output_dir, "*.pdb")))
to_score = [p for p in pdb_files if os.path.basename(p) not in processed_files]

new_scores = []
if len(to_score) > 0:
  with suppress_stdout():
    scorer = ModernAF2Rank(reference_cif, chain=chain, model_name={model_name!r})
  for pdb_path in to_score:
    pdb_filename = os.path.basename(pdb_path)
    with suppress_stdout():
      structure_scores = scorer.score_structure(
        pdb_path,
        decoy_chain="A",
        recycles={int(recycles)},
        verbose=False,
        _allatom_pdb=allatom_map.get(pdb_path),
      )
    structure_scores.update({{
      "protein_id": protein_id,
      "structure_file": pdb_filename,
      "structure_path": pdb_path,
    }})
    if "pred_coords" in structure_scores:
      del structure_scores["pred_coords"]
    new_scores.append(structure_scores)

all_scores = existing_scores + new_scores
save_af2rank_scores(all_scores, output_dir, protein_id)
staged_filenames = set(os.path.basename(p) for p in pdb_files)
plot_scores = [s for s in all_scores if s.get("structure_file") in staged_filenames]
plot_af2rank_results(plot_scores, output_dir, protein_id)
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    if direct_python:
        cmd = [sys.executable, "-c", py]
    else:
        cmd = [wrapper_script, "python", "-c", py]
    logger.info(f"Running AF2Rank subprocess for {protein_id} ({model_name}): {cmd[0]}")
    result = subprocess.run(cmd, cwd=wrapper_dir, capture_output=True, text=True)
    if result.returncode != 0:
        err_tail = result.stderr[-3000:] if result.stderr else "(no stderr)"
        out_tail = result.stdout[-1000:] if result.stdout else "(no stdout)"
        raise RuntimeError(
            f"AF2Rank subprocess failed for {protein_id} ({model_name}), "
            f"exit={result.returncode}:\nSTDERR:\n{err_tail}\nSTDOUT:\n{out_tail}"
        )


def _plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 8), dpi=140)
    valid = df.dropna(subset=[x_col, y_col, "in_train", "length"])
    if len(valid) == 0:
        raise ValueError(f"No valid points to plot for {x_col} vs {y_col}")

    colors = valid["in_train"].map({True: "blue", False: "red"})
    sizes = np.clip(valid["length"].astype(float).to_numpy() / 1.5, 20, 800)

    plt.scatter(valid[x_col], valid[y_col], c=colors, s=sizes, alpha=0.25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _process_one_protein(
    protein_id: str,
    scores_csv: str,
    dataset_ref: Dict[str, object],
    top_k: int,
    recycles: int,
    gpu_id: str,
    filter_existing: bool,
    dry_run: bool,
    backend: str = "colabdesign",
    direct_python: bool = False,
    cif_dir: str = "",
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = True,
    use_cuequivariance_multiplicative_update: bool = True,
) -> Dict[str, object]:
    scores_csv_path = Path(scores_csv)
    protein_dir = scores_csv_path.parent.parent

    protein_out_dir = protein_dir / "af2rank_on_proteinebm_top_k"
    af2rank_out_dir_m1 = protein_out_dir / "af2rank_analysis"
    af2rank_out_dir_m2 = protein_out_dir / "af2rank_analysis_model_2_ptm"
    af2rank_scores_csv_m1 = af2rank_out_dir_m1 / f"af2rank_scores_{protein_id}.csv"
    af2rank_scores_csv_m2 = af2rank_out_dir_m2 / f"af2rank_scores_{protein_id}.csv"

    summary_json = scores_csv_path.parent / f"proteinebm_summary_{protein_id}.json"
    meta = _read_proteinebm_summary(summary_json)
    reference_cif = meta["reference_structure"]
    chain = meta["chain"]

    if reference_cif and not os.path.exists(str(reference_cif)) and cif_dir:
        resolved = _find_reference_cif(protein_id, cif_dir)
        if resolved:
            reference_cif = resolved

    topk_df = _select_topk_templates(scores_csv_path, top_k)

    if reference_cif is None or not os.path.exists(str(reference_cif)):
        reference_cif = str(topk_df.iloc[0]["structure_path"])
        chain = "A"
    staged_dir = protein_out_dir / "staged_topk_templates"
    desired = set([Path(str(p)).name for p in topk_df["structure_path"].tolist()])
    staged_filenames = desired

    def _prefix_metrics(m: Dict[str, float], prefix: str) -> Dict[str, float]:
        return {f"{prefix}_{k}": v for k, v in m.items()}

    if filter_existing and af2rank_scores_csv_m1.exists() and af2rank_scores_csv_m2.exists():
        m1_df = pd.read_csv(af2rank_scores_csv_m1)
        m2_df = pd.read_csv(af2rank_scores_csv_m2)
        for df, path in [(m1_df, af2rank_scores_csv_m1), (m2_df, af2rank_scores_csv_m2)]:
            if "structure_file" not in df.columns:
                break
            processed = set(df["structure_file"].dropna().astype(str).tolist())
            if not desired.issubset(processed):
                break
        else:
            # Both models have all desired templates — return existing summary metrics
            m1_metrics = _extract_top1_top5_metrics(m1_df, desired)
            m2_metrics = _extract_top1_top5_metrics(m2_df, desired)
            min_df = _merge_min_across_models(m1_df, m2_df, desired)
            min_metrics = _extract_top1_top5_metrics(min_df, desired)
            topk_df2 = topk_df.copy()
            topk_df2["structure_file"] = topk_df2["structure_path"].apply(lambda p: Path(str(p)).name)
            j1 = topk_df2.merge(m1_df[["structure_file", "ptm"]], on="structure_file", how="left")
            j2 = topk_df2.merge(m2_df[["structure_file", "ptm"]], on="structure_file", how="left")
            max_ptm = float(pd.to_numeric(j1["ptm"], errors="coerce").dropna().max())
            m2_max_ptm = float(pd.to_numeric(j2["ptm"], errors="coerce").dropna().max())
            min_df = _merge_min_across_models(m1_df, m2_df, desired)
            jmin = topk_df2.merge(min_df[["structure_file", "ptm"]], on="structure_file", how="left")
            min_max_ptm = float(pd.to_numeric(jmin["ptm"], errors="coerce").dropna().max())
            return {
                "protein_id": protein_id,
                "reference_tm": float(dataset_ref["reference_tm"]),
                "in_train": bool(dataset_ref["in_train"]),
                "length": float(dataset_ref["length"]),
                "top_k": int(top_k),
                "min_energy_topk": float(topk_df["energy"].min()),
                "max_ptm_topk": max_ptm,
                "m2_max_ptm_topk": m2_max_ptm,
                "min_max_ptm_topk": min_max_ptm,
                **_prefix_metrics(m1_metrics, "m1"),
                **_prefix_metrics(m2_metrics, "m2"),
                **_prefix_metrics(min_metrics, "min"),
                "top_1_ptm": m1_metrics["top_1_ptm"],
                "top_1_tm_ref_pred": m1_metrics["top_1_tm_ref_pred"],
                "top_1_composite": m1_metrics["top_1_composite"],
                "top_1_plddt": m1_metrics["top_1_plddt"],
                "top_5_ptm": m1_metrics["top_5_ptm"],
                "top_5_tm_ref_pred": m1_metrics["top_5_tm_ref_pred"],
                "top_5_composite": m1_metrics["top_5_composite"],
                "top_5_plddt": m1_metrics["top_5_plddt"],
                "proteinebm_scores_csv": str(scores_csv_path),
                "af2rank_scores_csv": str(af2rank_scores_csv_m1),
            }

    # Stage templates (always rebuilds fresh to match current top_k).
    _stage_templates_as_dir(topk_df, staged_dir)

    max_ptm = float("nan")
    m2_max_ptm = float("nan")
    min_max_ptm = float("nan")
    af2rank_scores_csv_str = ""
    m1_metrics = _prefix_metrics(
        {"top_1_ptm": float("nan"), "top_1_tm_ref_pred": float("nan"), "top_1_composite": float("nan"), "top_1_plddt": float("nan"),
         "top_5_ptm": float("nan"), "top_5_tm_ref_pred": float("nan"), "top_5_composite": float("nan"), "top_5_plddt": float("nan")},
        "m1",
    )
    m2_metrics = _prefix_metrics(
        {"top_1_ptm": float("nan"), "top_1_tm_ref_pred": float("nan"), "top_1_composite": float("nan"), "top_1_plddt": float("nan"),
         "top_5_ptm": float("nan"), "top_5_tm_ref_pred": float("nan"), "top_5_composite": float("nan"), "top_5_plddt": float("nan")},
        "m2",
    )
    min_metrics = _prefix_metrics(
        {"top_1_ptm": float("nan"), "top_1_tm_ref_pred": float("nan"), "top_1_composite": float("nan"), "top_1_plddt": float("nan"),
         "top_5_ptm": float("nan"), "top_5_tm_ref_pred": float("nan"), "top_5_composite": float("nan"), "top_5_plddt": float("nan")},
        "min",
    )

    if not dry_run:
        # Batch-reconstruct all CA-only templates once upfront, reuse for both models
        allatom_map: Dict[str, str] = {}
        if backend == "colabdesign":
            allatom_map = _batch_reconstruct_ca_only(staged_dir, direct_python=direct_python)
        try:
            _run_af2rank_subprocess(
                protein_id=protein_id,
                reference_cif=reference_cif,
                chain=chain,
                staged_templates_dir=staged_dir,
                output_dir=af2rank_out_dir_m1,
                recycles=recycles,
                cuda_visible_devices=gpu_id,
                model_name="model_1_ptm",
                backend=backend,
                allatom_map=allatom_map,
                direct_python=direct_python,
                use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            )
            _run_af2rank_subprocess(
                protein_id=protein_id,
                reference_cif=reference_cif,
                chain=chain,
                staged_templates_dir=staged_dir,
                output_dir=af2rank_out_dir_m2,
                recycles=recycles,
                cuda_visible_devices=gpu_id,
                model_name="model_2_ptm",
                backend=backend,
                allatom_map=allatom_map,
                direct_python=direct_python,
                use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            )
        finally:
            # Delete reconstructed all-atom temp files after both models are done
            for tmp_path in allatom_map.values():
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        m1_df = pd.read_csv(af2rank_scores_csv_m1)
        m2_df = pd.read_csv(af2rank_scores_csv_m2)
        if "ptm" not in m1_df.columns:
            raise KeyError(f"AF2Rank scores missing 'ptm': {af2rank_scores_csv_m1}")
        topk_df = topk_df.copy()
        topk_df["structure_file"] = topk_df["structure_path"].apply(lambda p: Path(str(p)).name)
        staged_filenames = set(topk_df["structure_file"].astype(str).tolist())
        j1 = topk_df.merge(m1_df[["structure_file", "ptm"]], on="structure_file", how="left")
        j2 = topk_df.merge(m2_df[["structure_file", "ptm"]], on="structure_file", how="left")
        max_ptm = float(pd.to_numeric(j1["ptm"], errors="coerce").dropna().max())
        min_df = _merge_min_across_models(m1_df, m2_df, staged_filenames)
        jmin = topk_df.merge(min_df[["structure_file", "ptm"]], on="structure_file", how="left")
        m2_max_ptm = float(pd.to_numeric(j2["ptm"], errors="coerce").dropna().max())
        min_max_ptm = float(pd.to_numeric(jmin["ptm"], errors="coerce").dropna().max())
        af2rank_scores_csv_str = str(af2rank_scores_csv_m1)
        m1_metrics = _prefix_metrics(_extract_top1_top5_metrics(m1_df, staged_filenames), "m1")
        m2_metrics = _prefix_metrics(_extract_top1_top5_metrics(m2_df, staged_filenames), "m2")
        min_metrics = _prefix_metrics(_extract_top1_top5_metrics(min_df, staged_filenames), "min")

    min_energy = float(topk_df["energy"].min())
    m1_raw = {k.replace("m1_", ""): v for k, v in m1_metrics.items()}
    m2_max_ptm_val = m2_max_ptm if not dry_run else float("nan")
    min_max_ptm_val = min_max_ptm if not dry_run else float("nan")
    return {
        "protein_id": protein_id,
        "reference_tm": float(dataset_ref["reference_tm"]),
        "in_train": bool(dataset_ref["in_train"]),
        "length": float(dataset_ref["length"]),
        "top_k": int(top_k),
        "min_energy_topk": min_energy,
        "max_ptm_topk": max_ptm,
        "m2_max_ptm_topk": m2_max_ptm_val,
        "min_max_ptm_topk": min_max_ptm_val,
        **m1_metrics,
        **m2_metrics,
        **min_metrics,
        "top_1_ptm": m1_raw["top_1_ptm"],
        "top_1_tm_ref_pred": m1_raw["top_1_tm_ref_pred"],
        "top_1_composite": m1_raw["top_1_composite"],
        "top_1_plddt": m1_raw["top_1_plddt"],
        "top_5_ptm": m1_raw["top_5_ptm"],
        "top_5_tm_ref_pred": m1_raw["top_5_tm_ref_pred"],
        "top_5_composite": m1_raw["top_5_composite"],
        "top_5_plddt": m1_raw["top_5_plddt"],
        "proteinebm_scores_csv": str(scores_csv_path),
        "af2rank_scores_csv": af2rank_scores_csv_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AF2Rank on ProteinEBM top-k templates and plot cross-protein diagnostics")
    parser.add_argument("--inference_dir", required=True, help="Base inference directory containing per-protein folders")
    parser.add_argument("--cif_dir", default="", help="Directory containing reference CIF files (resolves hardcoded paths from ProteinEBM summary)")
    parser.add_argument("--dataset_file", default="", help="Optional dataset CSV used for cross-protein plots (reference TM / in_train / length).")
    parser.add_argument("--id_column", default="natives_rcsb", help="Dataset column for protein ID")
    parser.add_argument("--tms_column", default="tms_single", help="Dataset column for reference TM score")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top templates to select by ProteinEBM (min energy)")
    parser.add_argument("--recycles", type=int, default=3, help="AF2 recycles for AF2Rank runs")
    parser.add_argument("--output_dir", default="", help="Directory to write cross-protein outputs (default: <inference_dir>/af2rank_on_proteinebm_top_k_cross_protein_analysis)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for AF2Rank subprocess parallelism")
    parser.add_argument("--limit", type=int, default=0, help="Limit proteins processed (0 = no limit)")
    parser.add_argument("--filter_existing", action=argparse.BooleanOptionalAction, default=True, help="Skip proteins whose per-protein AF2Rank-topk output already exists")
    parser.add_argument("--cuda_visible_devices", default="", help="Comma-separated GPU ids to use (e.g. '0,1,2'). If empty, uses 0..num_gpus-1.")
    parser.add_argument("--dry_run", action="store_true", help="Validate top-k selection and dataset joins without running AF2Rank (no pTM plot)")
    parser.add_argument("--proteinebm_analysis_subdir", default="proteinebm_v2_cathmd_analysis", help="Per-protein subdir containing ProteinEBM scores (default: proteinebm_v2_cathmd_analysis)")
    parser.add_argument("--backend", choices=["colabdesign", "openfold"], default="colabdesign",
                       help="AF2Rank backend: colabdesign (JAX) or openfold (PyTorch)")
    parser.add_argument("--use_deepspeed_evoformer_attention", action=argparse.BooleanOptionalAction, default=True,
                       help="Use DeepSpeed evoformer attention (openfold backend, default: True)")
    parser.add_argument("--use_cuequivariance_attention", action=argparse.BooleanOptionalAction, default=True,
                       help="Use cuEquivariance attention kernels (openfold backend, default: True)")
    parser.add_argument("--use_cuequivariance_multiplicative_update", action=argparse.BooleanOptionalAction, default=True,
                       help="Use cuEquivariance multiplicative update (openfold backend, default: True)")
    parser.add_argument("--direct_python", action="store_true", default=False,
                       help="Use current Python interpreter for inner subprocesses instead of shell wrappers (for HPC)")
    add_shard_args(parser)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")

    if args.num_gpus <= 0:
        raise ValueError("--num_gpus must be > 0")

    gpu_ids: List[str]
    if args.cuda_visible_devices.strip():
        gpu_ids = [g.strip() for g in args.cuda_visible_devices.split(",") if g.strip()]
    else:
        gpu_ids = [str(i) for i in range(int(args.num_gpus))]
    if len(gpu_ids) == 0:
        raise ValueError("No GPU ids provided/derived for AF2Rank subprocesses")

    score_csvs = _find_proteinebm_scores(args.inference_dir, args.proteinebm_analysis_subdir)
    logger.info(f"Found {len(score_csvs)} ProteinEBM score CSVs under {args.inference_dir}")
    if len(score_csvs) == 0:
        raise FileNotFoundError(f"No proteinebm_scores_*.csv found under {args.inference_dir}")

    has_dataset = bool(args.dataset_file.strip())
    dataset_map: Dict[str, Dict[str, float]] = {}
    if has_dataset:
        dataset_map = _load_dataset_map(args.dataset_file, args.id_column, args.tms_column)
        logger.info(f"Loaded {len(dataset_map)} proteins from dataset CSV")

    scores_by_protein: Dict[str, Path] = {p.parent.parent.name: p for p in score_csvs}
    logger.info(f"scores_by_protein has {len(scores_by_protein)} entries")

    out_dir = args.output_dir.strip()
    if not out_dir:
        out_dir = str(Path(args.inference_dir) / "af2rank_on_proteinebm_top_k_cross_protein_analysis")
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)
    if has_dataset and shard_index is not None:
        all_ids_for_sharding = list(dataset_map.keys())
        data_dir = os.environ.get("DATA_PATH", os.path.join(Path(__file__).resolve().parents[2], "data"))
        logger.info(f"Sharding {len(all_ids_for_sharding)} dataset proteins into shard {shard_index}/{num_shards} (data_dir={data_dir})")
        shard_ids = set(shard_proteins(all_ids_for_sharding, shard_index, num_shards, data_dir=data_dir))
        logger.info(f"Shard {shard_index} assigned {len(shard_ids)} proteins")
    elif shard_index is not None:
        all_ids_for_sharding = list(scores_by_protein.keys())
        data_dir = os.environ.get("DATA_PATH", os.path.join(Path(__file__).resolve().parents[2], "data"))
        logger.info(f"Sharding {len(all_ids_for_sharding)} score proteins into shard {shard_index}/{num_shards} (data_dir={data_dir})")
        shard_ids = set(shard_proteins(all_ids_for_sharding, shard_index, num_shards, data_dir=data_dir))
        logger.info(f"Shard {shard_index} assigned {len(shard_ids)} proteins")
    else:
        shard_ids = None

    tasks: List[Tuple[str, str, Dict[str, object], str]] = []
    n_considered = 0
    n_no_scores = 0
    candidate_ids = list(shard_ids) if shard_ids is not None else (list(dataset_map.keys()) if has_dataset else list(scores_by_protein.keys()))
    for protein_id in candidate_ids:
        if protein_id not in scores_by_protein:
            n_no_scores += 1
            continue
        if has_dataset and protein_id not in dataset_map:
            continue
        scores_csv = scores_by_protein[protein_id]
        gpu_id = gpu_ids[len(tasks) % len(gpu_ids)]
        if has_dataset:
            ref = dataset_map[protein_id]
            dataset_ref = {"reference_tm": float(ref["reference_tm"]), "in_train": bool(ref["in_train"]), "length": float(ref["length"])}
        else:
            dataset_ref = {"reference_tm": float("nan"), "in_train": False, "length": float("nan")}
        tasks.append((protein_id, str(scores_csv), dataset_ref, gpu_id))
        n_considered += 1
        if args.limit and n_considered >= int(args.limit):
            break

    logger.info(f"Built {len(tasks)} tasks (candidates={len(candidate_ids)}, skipped_no_scores={n_no_scores}, filter_existing={args.filter_existing}, direct_python={args.direct_python})")
    if len(tasks) > 0:
        logger.info(f"First task: {tasks[0][0]}, last task: {tasks[-1][0]}")

    cif_dir = args.cif_dir.strip() if hasattr(args, 'cif_dir') and args.cif_dir else ""

    def _submit_args(protein_id, scores_csv, ref, gpu_id):
        return (protein_id, scores_csv, ref, int(args.top_k), int(args.recycles),
                gpu_id, bool(args.filter_existing), bool(args.dry_run), args.backend,
                bool(args.direct_python), cif_dir,
                bool(args.use_deepspeed_evoformer_attention),
                bool(args.use_cuequivariance_attention),
                bool(args.use_cuequivariance_multiplicative_update))

    if not has_dataset:
        # Score only; skip cross-protein plots (they require reference TM / metadata).
        with tqdm(total=len(tasks), desc="AF2Rank top-k", unit="protein", file=sys.stderr) as pbar:
            if args.dry_run or int(args.num_gpus) == 1:
                for protein_id, scores_csv, ref, gpu_id in tasks:
                    try:
                        _process_one_protein(*_submit_args(protein_id, scores_csv, ref, gpu_id))
                    except Exception as e:
                        logger.error("FAILED %s: %s", protein_id, e)
                    pbar.update(1)
            else:
                with ProcessPoolExecutor(max_workers=int(args.num_gpus)) as ex:
                    futs = {
                        ex.submit(_process_one_protein, *_submit_args(pid, csv, ref, gid)): pid
                        for pid, csv, ref, gid in tasks
                    }
                    for fut in as_completed(futs):
                        pid = futs[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            logger.error("FAILED %s: %s", pid, e)
                        pbar.update(1)
        return

    rows: List[Dict[str, object]] = []
    n_failed = 0
    n_skipped_existing = 0
    with tqdm(total=len(tasks), desc="AF2Rank top-k", unit="protein", file=sys.stderr) as pbar:
        if args.dry_run or int(args.num_gpus) == 1:
            for protein_id, scores_csv, ref, gpu_id in tasks:
                try:
                    row = _process_one_protein(*_submit_args(protein_id, scores_csv, ref, gpu_id))
                    if row:
                        rows.append(row)
                        if row.get("af2rank_scores_csv"):
                            csv_path = Path(str(row["af2rank_scores_csv"]))
                            if csv_path.exists():
                                n_skipped_existing += 0
                except Exception as e:
                    n_failed += 1
                    logger.error("FAILED %s: %s", protein_id, e)
                pbar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=int(args.num_gpus)) as ex:
                futs = {
                    ex.submit(_process_one_protein, *_submit_args(pid, csv, ref, gid)): pid
                    for pid, csv, ref, gid in tasks
                }
                for fut in as_completed(futs):
                    pid = futs[fut]
                    try:
                        row = fut.result()
                        if row:
                            rows.append(row)
                    except Exception as e:
                        n_failed += 1
                        logger.error("FAILED %s: %s", pid, e)
                    pbar.update(1)

    logger.info(f"Processing complete: {len(rows)} rows collected, {n_failed} failed, from {len(tasks)} tasks")

    if len(rows) == 0:
        logger.warning("No protein rows collected (all skipped by filter_existing or all failed). "
                        "Cross-protein plots may be generated from pre-existing data by generate_cross_protein_plots.py.")
        if n_failed > 0:
            logger.error(f"{n_failed}/{len(tasks)} proteins failed — check errors above")
            sys.exit(1)
        return

    if shard_index is not None and shard_index != 0:
        logger.info("Per-protein work complete (non-zero shard), skipping cross-protein summary/plots.")
        return

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir_path / f"af2rank_on_proteinebm_top_{int(args.top_k)}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    _plot_scatter(
        summary_df,
        x_col="reference_tm",
        y_col="min_energy_topk",
        title=f"Reference TM vs ProteinEBM energy (top-{int(args.top_k)} templates by min energy)",
        xlabel="Reference TM score",
        ylabel="ProteinEBM energy (lower is better)",
        out_path=out_dir_path / f"ref_tm_vs_proteinebm_energy_topk{int(args.top_k)}.png",
    )

    if not args.dry_run:
        _plot_scatter(
            summary_df,
            x_col="reference_tm",
            y_col="max_ptm_topk",
            title=f"Reference TM vs AF2Rank pTM (best pTM within ProteinEBM top-{int(args.top_k)})",
            xlabel="Reference TM score",
            ylabel="AF2Rank pTM (higher is better)",
            out_path=out_dir_path / f"ref_tm_vs_af2rank_ptm_topk{int(args.top_k)}.png",
        )

        # TM_ref_pred vs pTM: does AF2 confidence correlate with prediction quality?
        if "top_1_ptm" in summary_df.columns and "top_1_tm_ref_pred" in summary_df.columns:
            _plot_scatter(
                summary_df,
                x_col="top_1_ptm",
                y_col="top_1_tm_ref_pred",
                title=f"Top-1 by pTM: Prediction TM (tm_ref_pred) vs AF2 pTM",
                xlabel="AF2 pTM (confidence)",
                ylabel="TM(ref vs pred)",
                out_path=out_dir_path / f"tm_ref_pred_vs_ptm_top1_topk{int(args.top_k)}.png",
            )
        if "top_5_ptm" in summary_df.columns and "top_5_tm_ref_pred" in summary_df.columns:
            _plot_scatter(
                summary_df,
                x_col="top_5_ptm",
                y_col="top_5_tm_ref_pred",
                title=f"Top-5 by pTM (best tm_ref_pred): Prediction TM vs AF2 pTM",
                xlabel="AF2 pTM (confidence)",
                ylabel="TM(ref vs pred)",
                out_path=out_dir_path / f"tm_ref_pred_vs_ptm_top5_topk{int(args.top_k)}.png",
            )

        if "top_1_composite" in summary_df.columns and "top_1_tm_ref_pred" in summary_df.columns:
            _plot_scatter(
                summary_df,
                x_col="top_1_composite",
                y_col="top_1_tm_ref_pred",
                title=f"Top-1 by pTM: Prediction TM vs composite score",
                xlabel="Composite score",
                ylabel="TM(ref vs pred)",
                out_path=out_dir_path / f"tm_ref_pred_vs_composite_top1_topk{int(args.top_k)}.png",
            )
        if "top_5_composite" in summary_df.columns and "top_5_tm_ref_pred" in summary_df.columns:
            _plot_scatter(
                summary_df,
                x_col="top_5_composite",
                y_col="top_5_tm_ref_pred",
                title=f"Top-5 by pTM (best tm_ref_pred): Prediction TM vs composite score",
                xlabel="Composite score",
                ylabel="TM(ref vs pred)",
                out_path=out_dir_path / f"tm_ref_pred_vs_composite_top5_topk{int(args.top_k)}.png",
            )

        if "top_1_plddt" in summary_df.columns and "top_1_tm_ref_pred" in summary_df.columns:
            _plot_scatter(
                summary_df,
                x_col="top_1_plddt",
                y_col="top_1_tm_ref_pred",
                title=f"Top-1 by pTM: Prediction TM vs pLDDT",
                xlabel="pLDDT",
                ylabel="TM(ref vs pred)",
                out_path=out_dir_path / f"tm_ref_pred_vs_plddt_top1_topk{int(args.top_k)}.png",
            )
        if "top_5_plddt" in summary_df.columns and "top_5_tm_ref_pred" in summary_df.columns:
            _plot_scatter(
                summary_df,
                x_col="top_5_plddt",
                y_col="top_5_tm_ref_pred",
                title=f"Top-5 by pTM (best tm_ref_pred): Prediction TM vs pLDDT",
                xlabel="pLDDT",
                ylabel="TM(ref vs pred)",
                out_path=out_dir_path / f"tm_ref_pred_vs_plddt_top5_topk{int(args.top_k)}.png",
            )

        # Model 2 (model_2_ptm) plots
        if "m2_max_ptm_topk" in summary_df.columns:
            valid = summary_df.dropna(subset=["reference_tm", "m2_max_ptm_topk", "in_train", "length"])
            if len(valid) > 0:
                _plot_scatter(
                    summary_df,
                    x_col="reference_tm",
                    y_col="m2_max_ptm_topk",
                    title=f"Reference TM vs AF2Rank pTM (model_2_ptm, top-{int(args.top_k)})",
                    xlabel="Reference TM score",
                    ylabel="AF2Rank pTM model_2_ptm",
                    out_path=out_dir_path / f"ref_tm_vs_af2rank_ptm_topk{int(args.top_k)}_model_2_ptm.png",
                )
        for suffix, x, y, top_num in [
            ("ptm", "m2_top_1_ptm", "m2_top_1_tm_ref_pred", "1"),
            ("ptm", "m2_top_5_ptm", "m2_top_5_tm_ref_pred", "5"),
            ("composite", "m2_top_1_composite", "m2_top_1_tm_ref_pred", "1"),
            ("composite", "m2_top_5_composite", "m2_top_5_tm_ref_pred", "5"),
            ("plddt", "m2_top_1_plddt", "m2_top_1_tm_ref_pred", "1"),
            ("plddt", "m2_top_5_plddt", "m2_top_5_tm_ref_pred", "5"),
        ]:
            if x in summary_df.columns and y in summary_df.columns:
                valid = summary_df.dropna(subset=[x, y, "in_train", "length"])
                if len(valid) > 0:
                    _plot_scatter(
                        summary_df,
                        x_col=x,
                        y_col=y,
                        title=f"Top-{top_num} by pTM (model_2_ptm): TM vs {suffix}",
                        xlabel="AF2 pTM" if suffix == "ptm" else f"Composite score" if suffix == "composite" else "pLDDT",
                        ylabel="TM(ref vs pred)",
                        out_path=out_dir_path / f"tm_ref_pred_vs_{suffix}_top{top_num}_topk{int(args.top_k)}_model_2_ptm.png",
                    )
        # Min across models plots
        if "min_max_ptm_topk" in summary_df.columns:
            valid = summary_df.dropna(subset=["reference_tm", "min_max_ptm_topk", "in_train", "length"])
            if len(valid) > 0:
                _plot_scatter(
                    summary_df,
                    x_col="reference_tm",
                    y_col="min_max_ptm_topk",
                    title=f"Reference TM vs min(pTM_1,pTM_2) (top-{int(args.top_k)})",
                    xlabel="Reference TM score",
                    ylabel="min(pTM_1, pTM_2)",
                    out_path=out_dir_path / f"ref_tm_vs_af2rank_ptm_topk{int(args.top_k)}_min.png",
                )
        for suffix, x, y, top_num in [
            ("ptm", "min_top_1_ptm", "min_top_1_tm_ref_pred", "1"),
            ("ptm", "min_top_5_ptm", "min_top_5_tm_ref_pred", "5"),
            ("composite", "min_top_1_composite", "min_top_1_tm_ref_pred", "1"),
            ("composite", "min_top_5_composite", "min_top_5_tm_ref_pred", "5"),
            ("plddt", "min_top_1_plddt", "min_top_1_tm_ref_pred", "1"),
            ("plddt", "min_top_5_plddt", "min_top_5_tm_ref_pred", "5"),
        ]:
            if x in summary_df.columns and y in summary_df.columns:
                valid = summary_df.dropna(subset=[x, y, "in_train", "length"])
                if len(valid) > 0:
                    _plot_scatter(
                        summary_df,
                        x_col=x,
                        y_col=y,
                        title=f"Top-{top_num} by min(pTM): TM vs {suffix}",
                        xlabel="min(pTM)" if suffix == "ptm" else f"min(composite)" if suffix == "composite" else "min(pLDDT)",
                        ylabel="TM(ref vs pred)",
                        out_path=out_dir_path / f"tm_ref_pred_vs_{suffix}_top{top_num}_topk{int(args.top_k)}_min.png",
                    )


if __name__ == "__main__":
    main()

