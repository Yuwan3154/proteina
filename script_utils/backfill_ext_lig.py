#!/usr/bin/env python
"""Backfill ext_lig labels into existing .pt files without full reprocessing.

Usage:
    # Backfill PDB .pt files (compute from raw deposits):
    python script_utils/backfill_ext_lig.py \
        --processed_dir data/pdb_train/processed \
        --raw_dir data/pdb_train/raw \
        --format cif \
        --database pdb \
        --workers 16

    # Only backfill chains in the dataset (avoids OOM on huge complexes):
    DATA_PATH=/path/to/data python script_utils/backfill_ext_lig.py \
        --processed_dir data/pdb_train/processed \
        --raw_dir data/pdb_train/raw \
        --config configs/datasets_config/pdb/pdb_train_S25_max512_purge-test_cutoff-190828.yaml \
        --database pdb --workers 16

    # Or pass the dataset CSV directly:
    python script_utils/backfill_ext_lig.py \
        --processed_dir data/pdb_train/processed \
        --raw_dir data/pdb_train/raw \
        --allowlist_csv data/pdb_train/df_pdb_*.csv \
        --database pdb --workers 16

    # Backfill AFDB .pt files (set all unknown):
    python script_utils/backfill_ext_lig.py \
        --processed_dir data/d_FS/processed \
        --database afdb

    # Dry run to check how many files need backfill:
    python script_utils/backfill_ext_lig.py \
        --processed_dir data/pdb_train/processed \
        --dry_run
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from proteinfoundation.datasets.ext_lig_utils import (
    compute_ext_lig_from_df,
    make_unknown_ext_lig,
)
from proteinfoundation.graphein_utils.graphein_utils import read_pdb_to_dataframe


def _backfill_single(
    pt_path: str,
    raw_dir: str,
    format_type: str,
    database: str,
    overwrite: bool,
) -> dict:
    """Backfill ext_lig for a single .pt file. Stores ext_lig as int8 to save disk space.
    Returns stats dict."""
    stats = {"path": pt_path, "status": "skipped", "time_s": 0.0}
    t0 = time.time()
    try:
        graph = torch.load(pt_path, weights_only=False)

        # Case 1: ext_lig present, correct dtype (int8), and not overwriting
        if hasattr(graph, "ext_lig") and graph.ext_lig.dtype == torch.int8 and not overwrite:
            stats["status"] = "already_has_ext_lig_int8"
            return stats

        # Case 2: ext_lig present but wrong dtype — convert to int8 and save
        if hasattr(graph, "ext_lig") and graph.ext_lig.dtype != torch.int8 and not overwrite:
            graph.ext_lig = graph.ext_lig.to(torch.int8)
            torch.save(graph, pt_path)
            stats["status"] = "converted_to_int8"
            stats["time_s"] = time.time() - t0
            return stats

        # Case 3: ext_lig absent (or overwrite) — compute and save in int8
        # Some .pt files lack coords (e.g. PyG removes None attributes) — skip those
        coords = getattr(graph, "coords", None)
        if coords is None and hasattr(graph, "get"):
            coords = graph.get("coords")
        if coords is None:
            stats["status"] = "incompatible_format_no_coords"
            stats["time_s"] = time.time() - t0
            return stats

        L = coords.shape[0]

        if database == "afdb":
            graph.ext_lig = make_unknown_ext_lig(L).to(torch.int8)
        else:
            graph_id = graph.id
            parts = graph_id.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) <= 2:
                pdb_code, chain = parts
                chains = chain
            else:
                pdb_code = graph_id
                chains = "all"

            raw_path = _find_raw_file(raw_dir, pdb_code, format_type)
            if raw_path is None:
                stats["status"] = "raw_not_found"
                stats["time_s"] = time.time() - t0
                return stats

            full_df = read_pdb_to_dataframe(path=raw_path)
            graph.ext_lig = compute_ext_lig_from_df(
                full_df=full_df,
                self_chains=chains,
                graph_residue_ids=list(graph.residue_id),
            ).to(torch.int8)

        torch.save(graph, pt_path)
        stats["status"] = "backfilled"
        present = int((graph.ext_lig == 1).sum())
        absent = int((graph.ext_lig == 0).sum())
        unknown = int((graph.ext_lig == 2).sum())
        stats["present"] = present
        stats["absent"] = absent
        stats["unknown"] = unknown
    except Exception as e:
        stats["status"] = f"error: {e}"

    stats["time_s"] = time.time() - t0
    return stats


def _find_raw_file(raw_dir, pdb_code, format_type):
    for name in [
        f"{pdb_code.lower()}.{format_type}",
        f"{pdb_code}.{format_type}",
        f"{pdb_code.lower()}.{format_type}.gz",
        f"{pdb_code}.{format_type}.gz",
    ]:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            return p
    return None


def _resolve_oc_env(cfg_str: str) -> str:
    """Expand ${oc.env:VAR} and ${oc.env:VAR,default} placeholders in YAML."""
    import re

    def _replacer(m):
        content = m.group(1)
        if "," in content:
            var, default = content.split(",", 1)
        else:
            var, default = content, None
        val = os.environ.get(var.strip())
        if val is None and default is not None:
            return default.strip()
        if val is None:
            raise EnvironmentError(
                f"Environment variable '{var.strip()}' is required. Export it, e.g.: export {var.strip()}=/path"
            )
        return val

    return re.sub(r"\$\{oc\.env:([^}]+)\}", _replacer, cfg_str)


def _get_file_identifier_from_config(ds_cfg) -> str:
    """Build file_identifier from dataselector config (matches PDBLightningDataModule._get_file_identifier)."""
    ds = ds_cfg if hasattr(ds_cfg, "get") else dict(ds_cfg)
    get = lambda k, d=None: ds.get(k, d) if hasattr(ds, "get") else getattr(ds, k, d)
    return (
        f"df_pdb_f{get('fraction', 1)}_minl{get('min_length')}_maxl{get('max_length')}_mt{get('molecule_type')}"
        f"_et{''.join(get('experiment_types') or [])}"
        f"_mino{get('oligomeric_min')}_maxo{get('oligomeric_max')}"
        f"_minr{get('best_resolution')}_maxr{get('worst_resolution')}"
        f"_hl{''.join(get('has_ligands') or [])}"
        f"_rl{''.join(get('remove_ligands') or [])}"
        f"_rnsr{get('remove_non_standard_residues', True)}_rpu{get('remove_pdb_unavailable', True)}"
        f"_l{''.join(get('labels') or [])}"
        f"_rcu{get('remove_cath_unavailable', False)}"
    )


def _load_allowlist(config_path: str = None, allowlist_csv: str = None, data_dir: str = None) -> set:
    """Load set of chain IDs (pt file stems) from config or CSV. Returns empty set if neither provided."""
    if allowlist_csv:
        # Direct path or glob
        base = Path(root)
        csv_path = Path(allowlist_csv)
        if not csv_path.is_absolute():
            csv_path = base / csv_path
        if "*" in str(csv_path):
            matches = list(Path(csv_path.parent).glob(csv_path.name))
            if not matches:
                print(f"WARNING: No CSV matched {allowlist_csv}", file=sys.stderr)
                return set()
            csv_path = matches[0]
        if not csv_path.exists():
            print(f"WARNING: Allowlist CSV not found: {csv_path}", file=sys.stderr)
            return set()
        df = pd.read_csv(csv_path)
        if "id" not in df.columns:
            print(f"WARNING: CSV has no 'id' column, using first column", file=sys.stderr)
            id_col = df.columns[0]
        else:
            id_col = "id"
        return set(df[id_col].astype(str).tolist())

    if config_path and data_dir:
        from omegaconf import OmegaConf

        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = Path(root) / cfg_path
        raw = cfg_path.read_text()
        resolved = _resolve_oc_env(raw)
        cfg = OmegaConf.create(resolved)
        dm = cfg.get("datamodule") or cfg
        dd = dm.get("data_dir") or data_dir
        data_dir_resolved = Path(_resolve_oc_env(str(dd)))
        ds_cfg = dm.get("dataselector") or {}
        file_id = _get_file_identifier_from_config(ds_cfg)
        csv_path = data_dir_resolved / f"{file_id}.csv"
        if not csv_path.exists():
            print(f"WARNING: Dataset CSV not found: {csv_path}", file=sys.stderr)
            return set()
        df = pd.read_csv(csv_path)
        id_col = "id" if "id" in df.columns else df.columns[0]
        return set(df[id_col].astype(str).tolist())

    return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed_dir", required=True, help="Directory containing .pt files")
    parser.add_argument("--raw_dir", default=None, help="Directory containing raw structure files (required for PDB)")
    parser.add_argument("--format", default="cif", help="Raw file format (cif, pdb, mmtf)")
    parser.add_argument("--database", default="pdb", choices=["pdb", "afdb"], help="Database type")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ext_lig")
    parser.add_argument("--dry_run", action="store_true", help="Only count files needing backfill")
    parser.add_argument("--list_incompatible", action="store_true", help="Print paths of files with no coords (use with --dry_run)")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config YAML. With DATA_PATH set, only backfill chains in the dataset CSV (avoids OOM on huge complexes).",
    )
    parser.add_argument(
        "--allowlist_csv",
        default=None,
        help="Path to dataset CSV (id column = pt file stem). Alternative to --config. Only backfill these chains.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    pt_files = sorted(processed_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {processed_dir}")

    # Optionally restrict to chains in the dataset (avoids OOM on huge complexes)
    allowlist = set()
    if args.config or args.allowlist_csv:
        data_dir = str(processed_dir.parent)  # e.g. .../pdb_train
        allowlist = _load_allowlist(
            config_path=args.config,
            allowlist_csv=args.allowlist_csv,
            data_dir=data_dir,
        )
        if allowlist:
            before = len(pt_files)
            pt_files = [p for p in pt_files if p.stem in allowlist]
            print(f"Restricted to {len(pt_files)} chains in dataset (excluded {before - len(pt_files)} not in allowlist)")
        else:
            print("WARNING: Could not load allowlist, processing all files")

    if not pt_files:
        print("No .pt files to process.")
        return

    if args.dry_run:
        need_backfill = 0
        need_convert = 0
        no_coords = 0
        incompatible_paths = []
        for pt_path in tqdm(pt_files, desc="Checking"):
            graph = torch.load(str(pt_path), weights_only=False)
            coords = getattr(graph, "coords", None)
            if coords is None and hasattr(graph, "get"):
                coords = graph.get("coords")
            if coords is None:
                no_coords += 1
                if args.list_incompatible:
                    incompatible_paths.append(str(pt_path))
                continue
            if not hasattr(graph, "ext_lig"):
                need_backfill += 1
            elif graph.ext_lig.dtype != torch.int8:
                need_convert += 1
        print(f"{need_backfill} / {len(pt_files)} files need ext_lig backfill")
        print(f"{need_convert} / {len(pt_files)} files need ext_lig dtype conversion to int8")
        if no_coords:
            print(f"{no_coords} / {len(pt_files)} files have incompatible format (no coords)")
            if args.list_incompatible and incompatible_paths:
                print("\nIncompatible files:")
                for p in incompatible_paths:
                    print(f"  {p}")
        return

    if args.database == "pdb" and args.raw_dir is None:
        parser.error("--raw_dir is required for PDB backfill")

    from functools import partial
    fn = partial(
        _backfill_single,
        raw_dir=args.raw_dir or "",
        format_type=args.format,
        database=args.database,
        overwrite=args.overwrite,
    )

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fn, str(p)): p for p in pt_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Backfilling"):
                results.append(future.result())
    else:
        for p in tqdm(pt_files, desc="Backfilling"):
            results.append(fn(str(p)))

    statuses = {}
    total_time = 0.0
    incompatible_paths = []
    for r in results:
        s = r["status"]
        statuses[s] = statuses.get(s, 0) + 1
        total_time += r.get("time_s", 0.0)
        if s == "incompatible_format_no_coords" and args.list_incompatible:
            incompatible_paths.append(r.get("path", ""))

    print(f"\nResults ({len(results)} files, {total_time:.1f}s total):")
    for s, count in sorted(statuses.items()):
        print(f"  {s}: {count}")
    if args.list_incompatible and incompatible_paths:
        print("\nIncompatible files (no coords):")
        for p in incompatible_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
