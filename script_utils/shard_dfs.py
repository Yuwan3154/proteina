#!/usr/bin/env python3
"""Adaptive UniProt-prefix sharding for d_FS (AFDB) files.

Filename convention: ``AF-<UniProt>-F1-model_v6.pdb`` (or ``...model_v6.pt``).
Bucket = a prefix of the UniProt accession of variable length (2 by default;
deeper where the 2-char bucket would exceed ``--max-per-bucket``). This is
needed because the AFDB UniProt namespace is hugely skewed: ~91 percent of
our 1.05M raw files have prefix ``A0``.

Two artefacts:
- ``shard_manifest.json``: maps each id stem -> bucket dir name, plus a
  ``depth_table`` that records the chosen depth per top-level prefix so other
  tools can re-derive the bucket without scanning the full ID list.
- A bucket function ``id_to_bucket(id_stem, depth_table)`` for use in other
  scripts via ``from proteina.script_utils.shard_dfs import id_to_bucket``.

Usage:
    # Build manifest from effective_ids.txt
    python -m script_utils.shard_dfs build \
        --ids ~/proteina/data/d_FS/effective_ids.txt \
        --manifest ~/proteina/data/d_FS/shard_manifest.json

    # Move raw files into the sharded raw/ tree (dry-run first; mv when --execute)
    python -m script_utils.shard_dfs move-raw \
        --manifest ~/proteina/data/d_FS/shard_manifest.json \
        --src ~/proteina/data/d_FS/raw_unfiltered \
        --dst ~/proteina/data/d_FS/raw \
        --execute
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_AFDB_PREFIX = "AF-"


def uniprot_from_stem(stem: str) -> str:
    """Extract the UniProt accession from a stem like ``AF-Q8WZ42-F1-model_v6``."""
    if not stem.startswith(_AFDB_PREFIX):
        raise ValueError(f"stem {stem!r} does not start with {_AFDB_PREFIX!r}")
    rest = stem[len(_AFDB_PREFIX):]
    # ``Q8WZ42-F1-model_v6`` -> ``Q8WZ42``
    return rest.split("-", 1)[0]


def _build_depth_table(uniprots: List[str], max_per_bucket: int) -> Dict[str, int]:
    """Pick the minimum prefix depth per top-level 2-char bucket such that every
    leaf bucket has <= ``max_per_bucket`` IDs.

    Returns: depth_table[two_char_prefix] = chosen_depth (>=2).
    """
    by_2 = defaultdict(list)
    for u in uniprots:
        if len(u) < 2:
            continue
        by_2[u[:2]].append(u)
    depth_table: Dict[str, int] = {}
    for prefix2, members in by_2.items():
        depth = 2
        # If a single 2-char bucket is over cap, drill deeper.
        # We only consider one global depth per top-level prefix (the deepest
        # needed across its sub-tree). This keeps the manifest small and the
        # tree balanced for the dominant prefix.
        while True:
            sub_counts = defaultdict(int)
            for u in members:
                if len(u) < depth:
                    # IDs shorter than depth go into a 'short' bucket; should be rare
                    sub_counts[u] += 1
                else:
                    sub_counts[u[:depth]] += 1
            if max(sub_counts.values()) <= max_per_bucket:
                depth_table[prefix2] = depth
                break
            depth += 1
            if depth > 8:
                # Safety stop; AFDB UniProt accessions are 6-10 chars.
                depth_table[prefix2] = depth
                break
    return depth_table


def id_to_bucket(stem: str, depth_table: Dict[str, int]) -> str:
    """Return the bucket dir name for an AFDB ID stem given a depth_table."""
    u = uniprot_from_stem(stem)
    if len(u) < 2:
        return "_short"
    prefix2 = u[:2]
    depth = depth_table.get(prefix2, 2)
    return u[:min(len(u), depth)]


def build_manifest(ids_path: Path, manifest_path: Path, max_per_bucket: int) -> None:
    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    uniprots = [uniprot_from_stem(s) for s in ids]
    depth_table = _build_depth_table(uniprots, max_per_bucket)

    bucket_counts: Dict[str, int] = defaultdict(int)
    for s in ids:
        bucket_counts[id_to_bucket(s, depth_table)] += 1

    over_cap = {b: c for b, c in bucket_counts.items() if c > max_per_bucket}
    manifest = {
        "depth_table": depth_table,
        "max_per_bucket": max_per_bucket,
        "n_ids": len(ids),
        "n_buckets": len(bucket_counts),
        "max_bucket_count": max(bucket_counts.values()) if bucket_counts else 0,
        "over_cap_buckets": over_cap,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path} (ids={manifest['n_ids']} buckets={manifest['n_buckets']} "
          f"max_per_bucket={manifest['max_per_bucket']} max_bucket_count={manifest['max_bucket_count']} "
          f"over_cap={len(over_cap)})")
    if over_cap:
        print("WARNING: buckets exceed max_per_bucket (raise depth cap or split):",
              dict(list(over_cap.items())[:5]))


def load_manifest(path: Path) -> Dict[str, int]:
    data = json.loads(path.read_text())
    return data["depth_table"]


def move_raw(manifest_path: Path, src: Path, dst: Path, suffix: str, execute: bool) -> None:
    depth_table = load_manifest(manifest_path)
    src = src.expanduser()
    dst = dst.expanduser()
    moved = 0
    missing = 0
    skipped = 0
    for stem in (line.strip() for line in sys.stdin):
        if not stem:
            continue
        src_path = src / f"{stem}{suffix}"
        if not src_path.exists():
            missing += 1
            continue
        bucket = id_to_bucket(stem, depth_table)
        bucket_dir = dst / bucket
        dst_path = bucket_dir / src_path.name
        if dst_path.exists():
            skipped += 1
            continue
        if execute:
            bucket_dir.mkdir(parents=True, exist_ok=True)
            os.rename(src_path, dst_path)
        moved += 1
        if moved % 50000 == 0:
            print(f"  moved {moved} (missing={missing}, skipped={skipped})", flush=True)
    print(f"DONE: moved={moved} missing={missing} skipped={skipped} execute={execute}")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build shard manifest from effective_ids.txt")
    pb.add_argument("--ids", type=Path, required=True)
    pb.add_argument("--manifest", type=Path, required=True)
    pb.add_argument("--max-per-bucket", type=int, default=1000)

    pm = sub.add_parser("move-raw", help="Move raw files into sharded tree using stems on stdin")
    pm.add_argument("--manifest", type=Path, required=True)
    pm.add_argument("--src", type=Path, required=True)
    pm.add_argument("--dst", type=Path, required=True)
    pm.add_argument("--suffix", default=".pdb")
    pm.add_argument("--execute", action="store_true",
                    help="Actually mv files. Without this, dry-run only.")

    args = p.parse_args()
    if args.cmd == "build":
        build_manifest(args.ids, args.manifest, args.max_per_bucket)
    elif args.cmd == "move-raw":
        move_raw(args.manifest, args.src, args.dst, args.suffix, args.execute)
    return 0


if __name__ == "__main__":
    sys.exit(main())
