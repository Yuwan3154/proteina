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
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _find_proteinebm_scores(inference_dir: str) -> List[Path]:
    base = Path(inference_dir)
    return sorted(base.glob("*/proteinebm_analysis/proteinebm_scores_*.csv"))


def _read_proteinebm_summary(summary_path: Path) -> Dict[str, str]:
    with summary_path.open("r") as f:
        summary = json.load(f)
    ref = summary.get("reference_structure")
    chain = summary.get("chain")
    if not ref or not chain:
        raise ValueError(f"Missing reference_structure/chain in {summary_path}")
    return {"reference_structure": str(ref), "chain": str(chain)}


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
    return df


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


def _run_af2rank_subprocess(
    protein_id: str,
    reference_cif: str,
    chain: str,
    staged_templates_dir: Path,
    output_dir: Path,
    recycles: int,
    cuda_visible_devices: str,
) -> None:
    wrapper_script = "/home/ubuntu/proteina/af2rank_evaluation/run_with_colabdesign_env.sh"

    # Ensure each subprocess is pinned to a single GPU when requested.
    cuda_line = ""
    if cuda_visible_devices.strip():
        cuda_line = f"os.environ['CUDA_VISIBLE_DEVICES'] = {cuda_visible_devices!r}\n"

    py = f"""
import os
import sys
import glob
sys.path.append('/home/ubuntu/proteina/af2rank_evaluation')

import pandas as pd

from af2rank_scorer import ModernAF2Rank, plot_af2rank_results, save_af2rank_scores, suppress_stdout, load_af2rank_scores_from_csv

{cuda_line}
os.environ['ALPHAFOLD_DATA_DIR'] = os.path.expanduser('~/params')
os.environ['AF_PARAMS_DIR'] = os.path.expanduser('~/params')

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
  with suppress_stdout():
    scorer = ModernAF2Rank(reference_cif, chain=chain)
  for pdb_path in to_score:
    pdb_filename = os.path.basename(pdb_path)
    with suppress_stdout():
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
plot_af2rank_results(all_scores, output_dir, protein_id)
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [wrapper_script, "python", "-c", py]
    subprocess.run(cmd, cwd="/home/ubuntu/proteina/af2rank_evaluation", check=True)


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
) -> Dict[str, object]:
    scores_csv_path = Path(scores_csv)
    protein_dir = scores_csv_path.parent.parent

    protein_out_dir = protein_dir / "af2rank_on_proteinebm_top_k"
    af2rank_out_dir = protein_out_dir / "af2rank_analysis"
    af2rank_scores_csv = af2rank_out_dir / f"af2rank_scores_{protein_id}.csv"

    summary_json = scores_csv_path.parent / f"proteinebm_summary_{protein_id}.json"
    meta = _read_proteinebm_summary(summary_json)
    reference_cif = meta["reference_structure"]
    chain = meta["chain"]

    topk_df = _select_topk_templates(scores_csv_path, top_k)
    staged_dir = protein_out_dir / "staged_topk_templates"
    _stage_templates_as_dir(topk_df, staged_dir)

    if filter_existing and af2rank_scores_csv.exists():
        existing_df = pd.read_csv(af2rank_scores_csv)
        if "structure_file" in existing_df.columns:
            processed = set(existing_df["structure_file"].dropna().astype(str).tolist())
            desired = set([Path(str(p)).name for p in topk_df["structure_path"].tolist()])
            if desired.issubset(processed):
                return {}

    max_ptm = float("nan")
    af2rank_scores_csv_str = ""

    if not dry_run:
        _run_af2rank_subprocess(
            protein_id=protein_id,
            reference_cif=reference_cif,
            chain=chain,
            staged_templates_dir=staged_dir,
            output_dir=af2rank_out_dir,
            recycles=recycles,
            cuda_visible_devices=gpu_id,
        )

        af2_df = pd.read_csv(af2rank_scores_csv)
        if "ptm" not in af2_df.columns:
            raise KeyError(f"AF2Rank scores missing 'ptm': {af2rank_scores_csv}")
        af2_df = af2_df.dropna(subset=["structure_file", "ptm"])

        topk_df = topk_df.copy()
        topk_df["structure_file"] = topk_df["structure_path"].apply(lambda p: Path(str(p)).name)
        joined = topk_df.merge(af2_df[["structure_file", "ptm"]], on="structure_file", how="left")
        max_ptm = float(pd.to_numeric(joined["ptm"], errors="coerce").dropna().max())
        af2rank_scores_csv_str = str(af2rank_scores_csv)

    min_energy = float(topk_df["energy"].min())
    return {
        "protein_id": protein_id,
        "reference_tm": float(dataset_ref["reference_tm"]),
        "in_train": bool(dataset_ref["in_train"]),
        "length": float(dataset_ref["length"]),
        "top_k": int(top_k),
        "min_energy_topk": min_energy,
        "max_ptm_topk": max_ptm,
        "proteinebm_scores_csv": str(scores_csv_path),
        "af2rank_scores_csv": af2rank_scores_csv_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AF2Rank on ProteinEBM top-k templates and plot cross-protein diagnostics")
    parser.add_argument("--inference_dir", required=True, help="Base inference directory containing per-protein folders")
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

    score_csvs = _find_proteinebm_scores(args.inference_dir)
    if len(score_csvs) == 0:
        raise FileNotFoundError(f"No proteinebm_scores_*.csv found under {args.inference_dir}")

    has_dataset = bool(args.dataset_file.strip())
    dataset_map: Dict[str, Dict[str, float]] = {}
    if has_dataset:
        dataset_map = _load_dataset_map(args.dataset_file, args.id_column, args.tms_column)

    out_dir = args.output_dir.strip()
    if not out_dir:
        out_dir = str(Path(args.inference_dir) / "af2rank_on_proteinebm_top_k_cross_protein_analysis")
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, Dict[str, object], str]] = []
    n_considered = 0
    for scores_csv in score_csvs:
        protein_id = scores_csv.parent.parent.name
        if has_dataset and protein_id not in dataset_map:
            continue
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

    if not has_dataset:
        # Score only; skip cross-protein plots (they require reference TM / metadata).
        if args.dry_run or int(args.num_gpus) == 1:
            for protein_id, scores_csv, ref, gpu_id in tasks:
                _process_one_protein(
                    protein_id=protein_id,
                    scores_csv=scores_csv,
                    dataset_ref=ref,
                    top_k=int(args.top_k),
                    recycles=int(args.recycles),
                    gpu_id=gpu_id,
                    filter_existing=bool(args.filter_existing),
                    dry_run=bool(args.dry_run),
                )
        else:
            with ProcessPoolExecutor(max_workers=int(args.num_gpus)) as ex:
                futs = []
                for protein_id, scores_csv, ref, gpu_id in tasks:
                    futs.append(
                        ex.submit(
                            _process_one_protein,
                            protein_id,
                            scores_csv,
                            ref,
                            int(args.top_k),
                            int(args.recycles),
                            gpu_id,
                            bool(args.filter_existing),
                            bool(args.dry_run),
                        )
                    )
                for fut in futs:
                    fut.result()
        return

    rows: List[Dict[str, object]] = []
    if args.dry_run or int(args.num_gpus) == 1:
        for protein_id, scores_csv, ref, gpu_id in tasks:
            row = _process_one_protein(
                protein_id=protein_id,
                scores_csv=scores_csv,
                dataset_ref=ref,
                top_k=int(args.top_k),
                recycles=int(args.recycles),
                gpu_id=gpu_id,
                filter_existing=bool(args.filter_existing),
                dry_run=bool(args.dry_run),
            )
            if row:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_gpus)) as ex:
            futs = []
            for protein_id, scores_csv, ref, gpu_id in tasks:
                futs.append(
                    ex.submit(
                        _process_one_protein,
                        protein_id,
                        scores_csv,
                        ref,
                        int(args.top_k),
                        int(args.recycles),
                        gpu_id,
                        bool(args.filter_existing),
                        bool(args.dry_run),
                    )
                )
            for fut in futs:
                row = fut.result()
                if row:
                    rows.append(row)

    if len(rows) == 0:
        raise ValueError("No proteins processed (check dataset id_column/tms_column and inference_dir contents).")

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


if __name__ == "__main__":
    main()

