#!/usr/bin/env python3
"""
AF2Rank scoring for the prediction pipeline (no ground-truth).

Differences from run_af2rank_on_proteinebm_topk.py:
  - Does NOT require a proteinebm_summary_*.json file; reads top-k templates
    directly from proteinebm_scores_*.csv.
  - Initialises each AF2Rank model variant (model_1_ptm, model_2_ptm) exactly
    ONCE and reuses it across all proteins via reset_reference(), avoiding
    the 2N model-load overhead of the subprocess-per-protein approach.
  - CA-only templates (Proteina backbone-only outputs) are reconstructed to
    all-atom PDBs via cg2all exactly ONCE, before any model pass, and the
    reconstructed files are shared across both passes.  Without this, every
    template would be reconstructed twice (once per model variant).
  - When no ground-truth CIF is available the top-energy decoy is used as the
    self-reference (so pTM / pLDDT are still meaningful; TM-score columns are NaN).

Output layout (same as run_af2rank_on_proteinebm_topk.py so step_collect_results
can read it unchanged):
  <inference_base>/<protein_id>/af2rank_on_proteinebm_top_k/
      af2rank_analysis/af2rank_scores_<protein_id>.csv          (model_1_ptm)
      af2rank_analysis_model_2_ptm/af2rank_scores_<protein_id>.csv
"""

import argparse
import gc
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from proteinfoundation.af2rank_evaluation.sharding_utils import (
    add_shard_args,
    resolve_shard_args,
    shard_proteins,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_protein_ids(csv_file: str, id_column: str) -> List[str]:
    df = pd.read_csv(csv_file)
    if id_column not in df.columns:
        raise KeyError(f"CSV missing column '{id_column}'. Available: {sorted(df.columns)}")
    return [str(v).strip() for v in df[id_column].dropna().unique() if str(v).strip()]


def _select_topk(scores_csv: Path, top_k: int) -> pd.DataFrame:
    """Return the top-k rows from a ProteinEBM scores CSV sorted by energy (ascending)."""
    df = pd.read_csv(scores_csv)
    for col in ("structure_path", "energy"):
        if col not in df.columns:
            raise KeyError(f"ProteinEBM scores CSV missing column '{col}': {scores_csv}")
    df = df.dropna(subset=["structure_path", "energy"])
    df["energy"] = df["energy"].astype(float)
    df = df.sort_values("energy").head(top_k).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid rows in {scores_csv}")

    # Re-base paths to the protein dir in case the CSV was generated on a different
    # machine / mount point.
    protein_dir = scores_csv.parent.parent
    def _rebase(p: str) -> str:
        orig = Path(p)
        local = protein_dir / orig.name
        return str(local) if local.exists() else (str(orig) if orig.exists() else str(local))
    df["structure_path"] = df["structure_path"].astype(str).apply(_rebase)
    return df


def _is_ca_only(pdb_path: str) -> bool:
    has_atom = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                has_atom = True
                if line[12:16].strip() != "CA":
                    return False
    return has_atom


def _stage_templates(topk_df: pd.DataFrame, staged_dir: Path) -> None:
    """Create a fresh staged directory with symlinks to top-k template PDBs."""
    if staged_dir.exists():
        import shutil
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True)
    for _, row in topk_df.iterrows():
        src = Path(str(row["structure_path"]))
        if not src.exists():
            logger.warning(f"Template not found, skipping: {src}")
            continue
        dst = staged_dir / src.name
        if not dst.exists():
            os.symlink(str(src), str(dst))


def _pick_reference(topk_df: pd.DataFrame, cif_dir: Optional[str], protein_id: str) -> Tuple[str, str]:
    """Return (reference_path, chain).

    Priority:
      1. CIF found in cif_dir (ground-truth).
      2. Top-energy decoy used as self-reference (prediction mode).
    """
    if cif_dir:
        pdb_id = protein_id.split("_")[0]
        for candidate in [
            Path(cif_dir) / f"{pdb_id}.cif",
            *(Path(cif_dir) / sub / f"{pdb_id}.cif"
              for sub in Path(cif_dir).iterdir() if Path(cif_dir).is_dir() and (Path(cif_dir) / sub).is_dir()),
        ]:
            if Path(str(candidate)).exists():
                chain = protein_id.split("_", 1)[1] if "_" in protein_id else "A"
                return str(candidate), chain

    # Fall back to the top-energy decoy as self-reference
    top_decoy = str(topk_df.iloc[0]["structure_path"])
    return top_decoy, "A"


# ---------------------------------------------------------------------------
# cg2all pre-reconstruction (once for all proteins, shared across model passes)
# ---------------------------------------------------------------------------

def _batch_reconstruct_all_proteins(
    protein_configs: List[Dict],
) -> Dict[str, str]:
    """Reconstruct CA-only templates for all proteins via batched cg2all.

    Returns a mapping {original_ca_pdb_path: temp_allatom_pdb_path}.
    Raises on any cg2all failure — callers must not proceed without reconstruction.
    """
    # Collect every unique CA-only template path across all proteins
    all_ca_paths: List[str] = []
    seen: set = set()
    for cfg in protein_configs:
        for _, row in cfg["topk_df"].iterrows():
            p = str(row["structure_path"])
            if p not in seen and Path(p).exists() and _is_ca_only(p):
                all_ca_paths.append(p)
                seen.add(p)

    if not all_ca_paths:
        logger.info("No CA-only templates found; skipping pre-reconstruction.")
        return {}

    logger.info(
        f"Pre-reconstructing {len(all_ca_paths)} CA-only templates via cg2all ..."
    )
    from proteinfoundation.af2rank_evaluation.af2rank_openfold_scorer import (
        _get_cg2all_reconstructor,
    )
    reconstructor = _get_cg2all_reconstructor()
    allatom_map = reconstructor.reconstruct_batch(all_ca_paths)
    n_ok = len(allatom_map)
    n_fail = len(all_ca_paths) - n_ok
    logger.info(f"Pre-reconstruction complete: {n_ok} succeeded, {n_fail} failed")
    return allatom_map


# ---------------------------------------------------------------------------
# Persistent cg2all file management
# ---------------------------------------------------------------------------

def _persist_allatom_files(
    allatom_map: Dict[str, str],
    protein_configs: List[Dict],
    inference_base: Path,
) -> Dict[str, str]:
    """Copy temp cg2all files to persistent per-protein cg2all_topk_structures/ dirs.

    The temp files are removed after copying.  The returned map uses the same
    original CA-only PDB paths as keys but points to the persistent copies.
    """
    if not allatom_map:
        return {}

    # Build lookup: original CA-only PDB path → protein_id
    template_to_protein: Dict[str, str] = {}
    for cfg in protein_configs:
        for _, row in cfg["topk_df"].iterrows():
            template_to_protein[str(row["structure_path"])] = cfg["protein_id"]

    persistent_map: Dict[str, str] = {}
    for ca_path, tmp_path in allatom_map.items():
        protein_id = template_to_protein.get(ca_path, Path(ca_path).parent.name)
        cg2all_dir = (
            inference_base / protein_id
            / "af2rank_on_proteinebm_top_k"
            / "cg2all_topk_structures"
        )
        cg2all_dir.mkdir(parents=True, exist_ok=True)
        dest = cg2all_dir / (Path(ca_path).stem + "_allatom.pdb")
        if tmp_path and Path(tmp_path).exists():
            shutil.copy2(tmp_path, dest)
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
        persistent_map[ca_path] = str(dest)

    n_copied = sum(1 for v in persistent_map.values() if Path(v).exists())
    logger.info(f"Saved {n_copied} cg2all reconstructed PDBs to persistent cg2all_topk_structures/ dirs")
    return persistent_map


# ---------------------------------------------------------------------------
# Per-protein scoring (reuses an already-loaded scorer)
# ---------------------------------------------------------------------------

def _score_protein_with_scorer(
    scorer,
    protein_id: str,
    topk_df: pd.DataFrame,
    output_dir: Path,
    recycles: int,
    filter_existing: bool,
    allatom_map: Dict[str, str],
) -> None:
    """Score a single protein's top-k templates with a pre-loaded scorer instance.

    Args:
        allatom_map: Pre-reconstructed {ca_pdb_path: allatom_pdb_path} mapping.
            When a template is present in this map its all-atom PDB is used
            directly (cg2all reconstruction is skipped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / f"af2rank_scores_{protein_id}.csv"

    existing_files: set = set()
    existing_scores: List[Dict] = []
    if filter_existing and scores_csv.exists():
        try:
            ex_df = pd.read_csv(scores_csv)
            if "structure_file" in ex_df.columns:
                existing_files = set(ex_df["structure_file"].dropna().astype(str))
                existing_scores = ex_df.to_dict("records")
        except Exception:
            pass

    desired_files = set(Path(str(p)).name for p in topk_df["structure_path"].tolist())
    to_score = [
        row for _, row in topk_df.iterrows()
        if Path(str(row["structure_path"])).name not in existing_files
    ]

    if not to_score and existing_files.issuperset(desired_files):
        logger.info(f"  {protein_id}: all {len(desired_files)} templates already scored, skipping")
        return

    # Build items list: (ca_pdb_path, scored_pdb, orig_pdb_for_ca_extraction)
    # Use pre-reconstructed all-atom PDB when available (avoids cg2all in background thread).
    items = []
    for row in to_score:
        ca_pdb_path = str(row["structure_path"])
        prebuilt_allatom = allatom_map.get(ca_pdb_path)
        if prebuilt_allatom and Path(prebuilt_allatom).exists():
            items.append((ca_pdb_path, prebuilt_allatom, ca_pdb_path))
        else:
            items.append((ca_pdb_path, ca_pdb_path, None))

    def _featurize_item(item):
        ca_pdb_path, scored_pdb, orig_pdb = item
        if not Path(ca_pdb_path).exists():
            raise FileNotFoundError(f"Template not found: {ca_pdb_path}")
        batch, template_coords = scorer._featurize(scored_pdb, decoy_chain="A", _original_pdb=orig_pdb)
        return ca_pdb_path, batch, template_coords

    new_scores: List[Dict] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_featurize_item, items[0])
        for i, (ca_pdb_path, scored_pdb, orig_pdb) in enumerate(items):
            pdb_filename = Path(ca_pdb_path).name
            if i + 1 < len(items):
                next_future = executor.submit(_featurize_item, items[i + 1])
            try:
                _, batch, template_coords = future.result()
                with torch.no_grad():
                    out = scorer.model.model(batch)
                structure_scores = scorer._extract_scores(out, template_coords)
                gc.collect()
                torch.cuda.empty_cache()
                structure_scores.update({
                    "protein_id": protein_id,
                    "structure_file": pdb_filename,
                    "structure_path": ca_pdb_path,
                })
                structure_scores.pop("pred_coords", None)
                new_scores.append(structure_scores)
            except Exception as e:
                logger.error(f"  {protein_id}: failed to score {pdb_filename}: {e}")
            if i + 1 < len(items):
                future = next_future

    all_scores = existing_scores + new_scores
    if not all_scores:
        logger.warning(f"  {protein_id}: no scores to write")
        return

    pd.DataFrame(all_scores).to_csv(scores_csv, index=False)
    logger.info(f"  {protein_id}: wrote {len(all_scores)} scores to {scores_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AF2Rank prediction scoring — initialises each model ONCE across all proteins"
    )
    parser.add_argument("--inference_dir", required=True,
                        help="Base inference directory (contains per-protein folders)")
    parser.add_argument("--csv_file", required=True,
                        help="CSV file listing proteins to score")
    parser.add_argument("--csv_column", default="id",
                        help="Column name in --csv_file for protein IDs (default: id)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top ProteinEBM templates per protein (default: 5)")
    parser.add_argument("--recycles", type=int, default=3,
                        help="AF2 recycling iterations (default: 3)")
    parser.add_argument("--proteinebm_analysis_subdir", default="proteinebm_v2_cathmd_analysis",
                        help="Per-protein subdir containing ProteinEBM scores (default: proteinebm_v2_cathmd_analysis)")
    parser.add_argument("--cif_dir", default="",
                        help="Optional: directory with ground-truth CIF files. "
                             "When absent the top-energy decoy is used as self-reference.")
    parser.add_argument("--filter_existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip proteins whose AF2Rank CSVs already cover all desired templates (default: True)")
    parser.add_argument("--use_deepspeed_evoformer_attention",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cuequivariance_attention",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_cuequivariance_multiplicative_update",
                        action=argparse.BooleanOptionalAction, default=True)
    add_shard_args(parser)

    args = parser.parse_args()

    inference_base = Path(args.inference_dir)
    cif_dir = args.cif_dir.strip() or None

    # ── Build per-protein work list ──────────────────────────────────────────
    protein_ids = _load_protein_ids(args.csv_file, args.csv_column)
    logger.info(f"Loaded {len(protein_ids)} proteins from {args.csv_file}")

    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)
    if shard_index is not None:
        data_dir = os.environ.get("DATA_PATH", str(inference_base.parent.parent / "data"))
        protein_ids = shard_proteins(protein_ids, shard_index, num_shards, data_dir=data_dir)
        logger.info(f"Shard {shard_index}/{num_shards}: {len(protein_ids)} proteins")

    ProteinConfig = Dict  # type alias for clarity
    protein_configs: List[ProteinConfig] = []
    for protein_id in protein_ids:
        protein_dir = inference_base / protein_id
        scores_csv = protein_dir / args.proteinebm_analysis_subdir / f"proteinebm_scores_{protein_id}.csv"
        if not scores_csv.exists():
            logger.warning(f"No ProteinEBM scores CSV for {protein_id}, skipping")
            continue
        try:
            topk_df = _select_topk(scores_csv, args.top_k)
        except Exception as e:
            logger.error(f"Failed to select top-k for {protein_id}: {e}")
            continue

        reference_path, chain = _pick_reference(topk_df, cif_dir, protein_id)

        out_dir_m1 = protein_dir / "af2rank_on_proteinebm_top_k" / "af2rank_analysis"
        out_dir_m2 = protein_dir / "af2rank_on_proteinebm_top_k" / "af2rank_analysis_model_2_ptm"

        # Quick skip check
        if args.filter_existing:
            desired = set(Path(str(p)).name for p in topk_df["structure_path"])
            m1_done = _all_scored(out_dir_m1 / f"af2rank_scores_{protein_id}.csv", desired)
            m2_done = _all_scored(out_dir_m2 / f"af2rank_scores_{protein_id}.csv", desired)
            if m1_done and m2_done:
                logger.info(f"{protein_id}: already fully scored, skipping")
                continue

        protein_configs.append({
            "protein_id": protein_id,
            "topk_df": topk_df,
            "reference_path": reference_path,
            "chain": chain,
            "out_dir_m1": out_dir_m1,
            "out_dir_m2": out_dir_m2,
        })

    if not protein_configs:
        logger.info("No proteins to score.")
        return

    logger.info(f"{len(protein_configs)} proteins to score")

    # ── Import scorer (requires proteina / openfold env) ────────────────────
    from proteinfoundation.af2rank_evaluation.af2rank_openfold_scorer import OpenFoldAF2Rank

    # ── Pre-reconstruct CA-only templates ONCE (shared across both model passes)
    allatom_map = _batch_reconstruct_all_proteins(protein_configs)

    # ── Persist cg2all outputs to per-protein cg2all_topk_structures/ dirs ──
    # Temp files are moved to persistent locations; any leftover temps are cleaned up.
    allatom_map = _persist_allatom_files(allatom_map, protein_configs, inference_base)

    # ── Model 1: model_1_ptm ─────────────────────────────────────────────────
    _run_model_pass(
        protein_configs=protein_configs,
        model_name="model_1_ptm",
        out_dir_key="out_dir_m1",
        recycles=args.recycles,
        filter_existing=args.filter_existing,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        use_cuequivariance_attention=args.use_cuequivariance_attention,
        use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
        OpenFoldAF2Rank=OpenFoldAF2Rank,
        allatom_map=allatom_map,
    )

    # ── Model 2: model_2_ptm ─────────────────────────────────────────────────
    _run_model_pass(
        protein_configs=protein_configs,
        model_name="model_2_ptm",
        out_dir_key="out_dir_m2",
        recycles=args.recycles,
        filter_existing=args.filter_existing,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        use_cuequivariance_attention=args.use_cuequivariance_attention,
        use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
        OpenFoldAF2Rank=OpenFoldAF2Rank,
        allatom_map=allatom_map,
    )

    # ── Generate per-protein summary CSVs ────────────────────────────────────
    from proteinfoundation.af2rank_evaluation.topk_summary_utils import generate_topk_summary_csv

    for cfg in protein_configs:
        protein_id = cfg["protein_id"]
        out_dir = inference_base / protein_id / "af2rank_on_proteinebm_top_k"
        m1_csv = out_dir / "af2rank_analysis" / f"af2rank_scores_{protein_id}.csv"
        m2_csv = out_dir / "af2rank_analysis_model_2_ptm" / f"af2rank_scores_{protein_id}.csv"
        if not m1_csv.exists() or not m2_csv.exists():
            logger.warning(f"{protein_id}: AF2Rank score CSVs not found; skipping summary CSV")
            continue
        try:
            generate_topk_summary_csv(
                protein_id, cfg["topk_df"], m1_csv, m2_csv,
                allatom_map, out_dir,
            )
        except Exception as e:
            logger.warning(f"{protein_id}: summary CSV generation failed: {e}")

    logger.info("AF2Rank prediction scoring complete.")


def _all_scored(scores_csv: Path, desired_files: set) -> bool:
    """Return True if scores_csv exists and covers every file in desired_files."""
    if not scores_csv.exists():
        return False
    try:
        df = pd.read_csv(scores_csv)
        if "structure_file" not in df.columns:
            return False
        return desired_files.issubset(set(df["structure_file"].dropna().astype(str)))
    except Exception:
        return False


def _run_model_pass(
    protein_configs: List[Dict],
    model_name: str,
    out_dir_key: str,
    recycles: int,
    filter_existing: bool,
    use_deepspeed_evoformer_attention: bool,
    use_cuequivariance_attention: bool,
    use_cuequivariance_multiplicative_update: bool,
    OpenFoldAF2Rank,
    allatom_map: Dict[str, str],
) -> None:
    """Load model_name ONCE, then iterate over all proteins.

    Args:
        allatom_map: Pre-reconstructed {ca_pdb_path: allatom_pdb_path} mapping
            produced before the first model pass.  Shared (read-only) across
            both model passes so reconstruction happens exactly once.
    """
    logger.info(f"Loading AF2Rank {model_name} ...")

    # Initialise with the first protein's reference to load the model.
    first = protein_configs[0]
    scorer = OpenFoldAF2Rank(
        reference_pdb=first["reference_path"],
        chain=first["chain"],
        model_name=model_name,
        recycles=recycles,
        use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
        use_cuequivariance_attention=use_cuequivariance_attention,
        use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
    )
    logger.info(f"{model_name} loaded. Scoring {len(protein_configs)} proteins ...")

    for i, cfg in enumerate(protein_configs):
        protein_id = cfg["protein_id"]
        out_dir = cfg[out_dir_key]
        topk_df = cfg["topk_df"]
        desired = set(Path(str(p)).name for p in topk_df["structure_path"])

        if filter_existing and _all_scored(Path(out_dir) / f"af2rank_scores_{protein_id}.csv", desired):
            logger.info(f"  [{i+1}/{len(protein_configs)}] {protein_id} ({model_name}): already done, skipping")
            continue

        logger.info(f"  [{i+1}/{len(protein_configs)}] {protein_id} ({model_name})")

        # Swap reference to this protein (no model reload)
        if i > 0:
            try:
                scorer.reset_reference(cfg["reference_path"], cfg["chain"])
            except Exception as e:
                logger.error(f"  {protein_id}: reset_reference failed: {e}; skipping")
                continue

        try:
            _score_protein_with_scorer(
                scorer=scorer,
                protein_id=protein_id,
                topk_df=topk_df,
                output_dir=Path(out_dir),
                recycles=recycles,
                filter_existing=filter_existing,
                allatom_map=allatom_map,
            )
        except Exception as e:
            logger.error(f"  {protein_id} ({model_name}): scoring failed: {e}")


if __name__ == "__main__":
    main()
