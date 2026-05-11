"""Check whether specific PDB chain IDs are in train/val/test for a given dataset config.

The notebook this replaces hardcoded a single config and only reported a binary
``in_train``. Three failure modes go undiagnosed there:
  - the chain is in val or test (notebook reports False, looks like "filtered out")
  - the chain was filtered out by the dataselector (length / resolution / etc.)
  - the chain's cluster is not all-or-none in the same split (data leakage)

This script handles all three by reporting the per-chain split, cluster ID, cluster
size, and a per-cluster cohesion check. It mirrors the Hydra config-resolution path
from ``proteinfoundation/train.py`` so it queries exactly the same split assignment
the trainer would see.

Usage:
    DATA_PATH=/home/ubuntu/proteina/data \\
    python scripts/analysis/check_chains_in_dataset.py \\
        --config_name pdb_train_contact-CB-10A_S25_max384_purge-test_cutoff-190828 \\
        --chains 5dfm_A 5dfm_B 5doi_A 5doi_B 5doi_C 5doi_D \\
                 5dof_A 5dof_B 5dof_C 5dof_D 7uy5_K 7uy7_C

Or pass a CSV:
    ... --query_csv my_chains.csv --query_column pdb [--out_csv result.csv]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

_THIS = pathlib.Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))


def _normalize_pdb_chain(s: str, lower_pdb: bool = True) -> Optional[str]:
    """Normalize 'PDB_CHAIN' so it matches the dataset's file_names entries.

    The dataset stores PDB codes lowercase and chain IDs as-is — see
    df_pdb_*.csv schema and processed/{pdb}_{chain}.pt naming.
    """
    if "_" not in s:
        return None
    pdb, chain = s.split("_", 1)
    if lower_pdb:
        pdb = pdb.lower()
    return f"{pdb}_{chain}"


def _resolve_data_path(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    if os.environ.get("DATA_PATH"):
        return os.environ["DATA_PATH"]
    default = pathlib.Path.home() / "proteina" / "data"
    return str(default)


def _build_split_ids(dm) -> Dict[str, set]:
    """Build {split: set(file_names)} for train/val/test using _get_dataset.

    file_names from _get_dataset are like '5dfm_A' — no .pt extension
    (see pdb_data.py:1613). The .pt is appended only at access time at line 602.
    Constructing the PDBDataset shell here is cheap (no actual .pt loads yet).
    """
    splits = ["train", "val", "test"]
    out: Dict[str, set] = {}
    for split in splits:
        try:
            ds = dm._get_dataset(split)
            out[split] = set(ds.file_names) if ds.file_names else set()
        except KeyError:
            # Some configs may not produce all splits (e.g. some custom-folder runs)
            out[split] = set()
    return out


def _build_id_to_cluster(dm) -> Dict[str, Tuple[str, str, int, str]]:
    """Invert dm.clusterid_to_seqid_mappings into {seq_id: (split, cluster_id, size, rep)}.

    cluster_id IS the representative sequence ID by mmseqs2 convention (see
    cluster_utils.py read_cluster_tsv). For random-split datasets (no clustering),
    clusterid_to_seqid_mappings is None and this returns an empty dict.
    """
    out: Dict[str, Tuple[str, str, int, str]] = {}
    mappings = getattr(dm, "clusterid_to_seqid_mappings", None) or {}
    for split, cluster_dict in mappings.items():
        if cluster_dict is None:
            continue
        for cluster_id, members in cluster_dict.items():
            size = len(members)
            rep = cluster_id  # mmseqs convention: cluster key = representative
            for m in members:
                out[m] = (split, cluster_id, size, rep)
    return out


def _resolve_split(seq_id: str, split_ids: Dict[str, set]) -> str:
    for split in ("train", "val", "test"):
        if seq_id in split_ids.get(split, set()):
            return split
    return "not_in_any"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True,
                    help="Dataset config basename (no .yaml). e.g. "
                         "pdb_train_contact-CB-10A_S25_max384_purge-test_cutoff-190828")
    ap.add_argument("--config_subdir", default="pdb",
                    help="Subdir under configs/datasets_config/ (default: pdb)")
    ap.add_argument("--data_path", default=None,
                    help="Sets DATA_PATH env var (used by Hydra ${oc.env:DATA_PATH}). "
                         "Default: $DATA_PATH or ~/proteina/data")
    ap.add_argument("--chains", nargs="*", default=[],
                    help="Chain IDs to check, e.g. 5dfm_A 5dfm_B 7uy5_K")
    ap.add_argument("--query_csv", default=None,
                    help="Alternative to --chains: CSV with a column of chain IDs")
    ap.add_argument("--query_column", default="pdb",
                    help="Column name in --query_csv (default: pdb)")
    ap.add_argument("--out_csv", default=None,
                    help="Optional output CSV with per-chain results")
    ap.add_argument("--no_lower_pdb", action="store_true",
                    help="Disable PDB-code lowercasing (default: lowercase PDB, keep chain case)")
    args = ap.parse_args()

    # Set DATA_PATH env BEFORE importing anything that resolves configs.
    data_path = _resolve_data_path(args.data_path)
    os.environ["DATA_PATH"] = data_path
    print(f"[check_chains] DATA_PATH={data_path}")

    # Load dataset config via Hydra (mirror proteinfoundation/train.py:329-353).
    import hydra  # noqa: E402

    config_dir = _PROTEINA_ROOT / "configs" / "datasets_config" / args.config_subdir
    if not config_dir.exists():
        print(f"ERROR: config_dir does not exist: {config_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"[check_chains] config_dir={config_dir}")
    print(f"[check_chains] config_name={args.config_name}")

    # Use initialize_config_dir to allow custom config paths.
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.config_name)
    dm = hydra.utils.instantiate(cfg.datamodule)

    print("[check_chains] dm.prepare_data() ...")
    dm.prepare_data()
    print("[check_chains] dm.setup('fit') ...")
    dm.setup("fit")

    # Build per-split membership sets and the id->cluster lookup.
    split_ids = _build_split_ids(dm)
    for split, ids in split_ids.items():
        print(f"[check_chains]   |{split}|={len(ids)}")
    id_to_cluster = _build_id_to_cluster(dm)
    has_clusters = bool(id_to_cluster)
    if has_clusters:
        print(f"[check_chains] id_to_cluster: {len(id_to_cluster)} entries")
    else:
        print("[check_chains] No cluster mapping (random-split config or non-cluster dataset)")

    # Resolve queries.
    if args.query_csv:
        df_q = pd.read_csv(args.query_csv)
        if args.query_column not in df_q.columns:
            print(f"ERROR: column '{args.query_column}' not in {args.query_csv}", file=sys.stderr)
            sys.exit(2)
        queries = df_q[args.query_column].astype(str).tolist()
    else:
        queries = list(args.chains)

    if not queries:
        print("ERROR: no queries provided. Use --chains or --query_csv.", file=sys.stderr)
        sys.exit(2)

    rows = []
    for q in queries:
        normalized = _normalize_pdb_chain(q, lower_pdb=not args.no_lower_pdb)
        if normalized is None:
            rows.append({
                "query": q, "normalized": None, "in_split": "BAD_FORMAT",
                "cluster_id": None, "cluster_split": None, "cluster_size": None,
                "cluster_rep": None,
            })
            continue
        in_split = _resolve_split(normalized, split_ids)
        cluster_split, cluster_id, cluster_size, cluster_rep = (None, None, None, None)
        if has_clusters and normalized in id_to_cluster:
            cluster_split, cluster_id, cluster_size, cluster_rep = id_to_cluster[normalized]
        rows.append({
            "query": q,
            "normalized": normalized,
            "in_split": in_split,
            "cluster_id": cluster_id,
            "cluster_split": cluster_split,
            "cluster_size": cluster_size,
            "cluster_rep": cluster_rep,
        })

    result = pd.DataFrame(rows)

    # Cluster cohesion check: queries that share a cluster_id should share in_split.
    cohesion_warnings: List[str] = []
    if has_clusters:
        seen_clusters = result.dropna(subset=["cluster_id"])
        for cid, grp in seen_clusters.groupby("cluster_id"):
            unique_splits = set(grp["in_split"].tolist())
            if len(unique_splits) > 1:
                cohesion_warnings.append(
                    f"  cluster {cid}: queries split across {sorted(unique_splits)}: "
                    f"{grp[['query', 'in_split']].to_dict('records')}"
                )

    # Print summary.
    print("\n=== Per-chain result ===")
    print(result.to_string(index=False))

    print("\n=== Split summary ===")
    counts = result["in_split"].value_counts().to_dict()
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if has_clusters:
        print("\n=== Cluster cohesion check ===")
        if cohesion_warnings:
            print("  WARNING: queries from the same cluster ended up in different splits "
                  "(possible data-leakage signal):")
            for w in cohesion_warnings:
                print(w)
        else:
            print("  PASS — all queries within the same cluster share the same split.")

    if args.out_csv:
        out_path = pathlib.Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"\n[check_chains] wrote {out_path}")


if __name__ == "__main__":
    main()
