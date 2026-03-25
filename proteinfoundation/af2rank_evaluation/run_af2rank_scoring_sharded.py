#!/usr/bin/env python3
"""
Sharded AF2Rank scoring for Proteina outputs against ground-truth CIF structures.

Differences from parallel_af2rank_scoring.py:
  - Loads each AF2Rank model variant ONCE per shard and reuses it across all
    assigned proteins via reset_reference(), avoiding the N model-load overhead
    of the subprocess-per-protein approach.
  - Integrates SLURM array sharding: each array task scores its subset of proteins
    independently; no inter-shard barrier is needed (output is per-protein CSVs).
  - Uses the existing prefetch pattern inside score_proteina_structures_openfold()
    (featurize decoy i+1 in background while GPU runs forward on decoy i).
  - Scores all PDB files found in <inference_dir>/<protein>/ (same as the
    openfold backend path in parallel_af2rank_scoring.py).

SLURM array example:
    #SBATCH --array=0-3
    python run_af2rank_scoring_sharded.py \\
        --inference_dir /path/to/inference \\
        --csv_file proteins.csv \\
        --cif_dir /path/to/cifs \\
        --model_names model_1_ptm model_2_ptm

Single-node example (no sharding):
    python run_af2rank_scoring_sharded.py \\
        --inference_dir /path/to/inference \\
        --csv_file proteins.csv \\
        --cif_dir /path/to/cifs
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(_SCRIPT_DIR).parent.parent))  # proteina root
sys.path.insert(0, os.path.expanduser("~/openfold"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from sharding_utils import add_shard_args, resolve_shard_args, shard_proteins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_protein_ids(csv_file: str, id_column: str) -> List[str]:
    df = pd.read_csv(csv_file)
    if id_column not in df.columns:
        raise KeyError(f"CSV missing column '{id_column}'. Available: {sorted(df.columns)}")
    return [str(v).strip() for v in df[id_column].dropna().unique() if str(v).strip()]


def _find_reference_cif(protein_name: str, cif_dir: str) -> Optional[str]:
    """Find the reference CIF file for a protein by PDB ID (first part before '_')."""
    pdb_id = protein_name.split("_")[0]
    cif_path = Path(cif_dir)
    # Search in subdirectories first
    for subdir in cif_path.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{pdb_id}.cif"
            if candidate.exists():
                return str(candidate)
    # Direct path
    candidate = cif_path / f"{pdb_id}.cif"
    if candidate.exists():
        return str(candidate)
    return None


def _get_chain(protein_name: str) -> str:
    """Extract chain ID from protein_name (e.g. '1abc_A' → 'A')."""
    parts = protein_name.split("_")
    return parts[1] if len(parts) > 1 else "A"


def _all_scored(scores_csv: Path, pdb_files: List[str]) -> bool:
    """Return True if scores_csv exists, covers all pdb_files, and saved predictions exist."""
    if not scores_csv.exists():
        return False
    df = pd.read_csv(scores_csv)
    if "structure_file" not in df.columns:
        return False
    predicted_dir = scores_csv.parent / "predicted_structures"
    if "predicted_structure_path" not in df.columns:
        df["predicted_structure_path"] = df["structure_file"].astype(str).apply(lambda name: str(predicted_dir / name))
    existing = {
        str(row["structure_file"])
        for _, row in df.iterrows()
        if pd.notna(row["structure_file"]) and Path(str(row["predicted_structure_path"])).exists()
    }
    desired = {Path(p).name for p in pdb_files}
    return desired.issubset(existing)


# ---------------------------------------------------------------------------
# Per-model-variant pass: load once, iterate proteins
# ---------------------------------------------------------------------------

def _run_model_pass(
    protein_infos: List[Dict],
    model_name: str,
    out_subdir: str,
    recycles: int,
    filter_existing: bool,
    use_deepspeed_evoformer_attention: bool,
    use_cuequivariance_attention: bool,
    use_cuequivariance_multiplicative_update: bool,
    OpenFoldAF2Rank,
    run_af2rank_analysis_openfold,
) -> None:
    """Load model_name ONCE, then iterate over all proteins in this shard."""
    logger.info(f"Loading AF2Rank {model_name} ...")
    first = protein_infos[0]
    scorer = OpenFoldAF2Rank(
        reference_pdb=first["reference_cif"],
        chain=first["chain"],
        model_name=model_name,
        recycles=recycles,
        use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
        use_cuequivariance_attention=use_cuequivariance_attention,
        use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
    )
    logger.info(f"{model_name} loaded. Scoring {len(protein_infos)} proteins ...")

    for i, info in enumerate(protein_infos):
        protein_name = info["protein_name"]
        reference_cif = info["reference_cif"]
        chain = info["chain"]
        inference_output_dir = info["inference_output_dir"]
        pdb_files = info["pdb_files"]

        out_dir = Path(inference_output_dir) / out_subdir
        scores_csv = out_dir / f"af2rank_scores_{protein_name}.csv"

        if filter_existing and _all_scored(scores_csv, pdb_files):
            logger.info(f"  [{i+1}/{len(protein_infos)}] {protein_name} ({model_name}): already done, skipping")
            continue

        logger.info(f"  [{i+1}/{len(protein_infos)}] {protein_name} ({model_name})")

        # Swap reference (no model reload)
        if i > 0:
            try:
                scorer.reset_reference(reference_cif, chain)
            except Exception as e:
                logger.error(f"  {protein_name}: reset_reference failed: {e}; skipping")
                continue

        try:
            run_af2rank_analysis_openfold(
                protein_id=protein_name,
                reference_cif=reference_cif,
                inference_output_dir=inference_output_dir,
                output_dir=str(out_dir),
                chain=chain,
                recycles=recycles,
                model_name=model_name,
                regenerate_summary=False,
                scorer=scorer,
            )
        except Exception as e:
            logger.error(f"  {protein_name} ({model_name}): scoring failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sharded AF2Rank scoring — loads each model ONCE per shard"
    )
    parser.add_argument("--inference_dir", required=True,
                        help="Base inference directory (contains per-protein subdirs)")
    parser.add_argument("--csv_file", required=True,
                        help="CSV file listing proteins to score")
    parser.add_argument("--csv_column", default="id",
                        help="Column name in --csv_file for protein IDs (default: id)")
    parser.add_argument("--cif_dir", required=True,
                        help="Directory with ground-truth CIF files (searched recursively by PDB ID)")
    parser.add_argument("--model_names", nargs="+", default=["model_1_ptm"],
                        help="AF2Rank model variant(s) to run (default: model_1_ptm)")
    parser.add_argument("--output_subdir", default="af2rank_analysis",
                        help="Subdirectory under each protein dir to write scores (default: af2rank_analysis)")
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--filter_existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip proteins whose CSVs already cover all PDB files (default: True)")
    parser.add_argument("--use_deepspeed_evoformer_attention",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cuequivariance_attention",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_cuequivariance_multiplicative_update",
                        action=argparse.BooleanOptionalAction, default=True)
    add_shard_args(parser)
    args = parser.parse_args()

    inference_base = Path(args.inference_dir)

    # ── Sharding ─────────────────────────────────────────────────────────────
    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)

    # ── Build protein list ────────────────────────────────────────────────────
    all_protein_ids = _load_protein_ids(args.csv_file, args.csv_column)
    logger.info(f"Loaded {len(all_protein_ids)} proteins from {args.csv_file}")

    if shard_index is not None and num_shards is not None:
        protein_ids = shard_proteins(all_protein_ids, shard_index, num_shards)
        logger.info(f"Shard {shard_index}/{num_shards}: {len(protein_ids)} proteins assigned")
    else:
        protein_ids = all_protein_ids
        logger.info("No sharding; processing all proteins")

    # ── Build per-protein work configs ───────────────────────────────────────
    protein_infos: List[Dict] = []
    for protein_name in protein_ids:
        inference_output_dir = str(inference_base / protein_name)
        if not Path(inference_output_dir).exists():
            logger.warning(f"Inference dir not found for {protein_name}: {inference_output_dir}, skipping")
            continue

        pdb_files = sorted(Path(inference_output_dir).glob("*.pdb"))
        if not pdb_files:
            logger.warning(f"No PDB files found for {protein_name} in {inference_output_dir}, skipping")
            continue

        reference_cif = _find_reference_cif(protein_name, args.cif_dir)
        if reference_cif is None:
            logger.warning(f"Reference CIF not found for {protein_name} in {args.cif_dir}, skipping")
            continue

        chain = _get_chain(protein_name)

        protein_infos.append({
            "protein_name": protein_name,
            "reference_cif": reference_cif,
            "chain": chain,
            "inference_output_dir": inference_output_dir,
            "pdb_files": [str(p) for p in pdb_files],
        })

    if not protein_infos:
        logger.info("No proteins to score.")
        return

    logger.info(f"{len(protein_infos)} proteins to score")

    # ── Import scorer ─────────────────────────────────────────────────────────
    from af2rank_openfold_scorer import OpenFoldAF2Rank, run_af2rank_analysis_openfold

    # ── Score: one model load per variant ─────────────────────────────────────
    for model_name in args.model_names:
        # Use model_name-specific output subdir for additional variants
        if model_name == "model_1_ptm":
            out_subdir = args.output_subdir
        else:
            out_subdir = f"{args.output_subdir}_{model_name}"

        _run_model_pass(
            protein_infos=protein_infos,
            model_name=model_name,
            out_subdir=out_subdir,
            recycles=args.recycles,
            filter_existing=args.filter_existing,
            use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=args.use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
            OpenFoldAF2Rank=OpenFoldAF2Rank,
            run_af2rank_analysis_openfold=run_af2rank_analysis_openfold,
        )

    logger.info("AF2Rank sharded scoring complete.")


if __name__ == "__main__":
    main()
