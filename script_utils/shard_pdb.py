#!/usr/bin/env python3
"""Middle-two-char (atomworks/PDB-mirror) sharding for PDB files.

PDB ID = 4 chars. The PDB rsync mirror divides by ``pdb_id[1:3]`` (chars at
positions 1 and 2, 0-indexed). For ``1abc``, bucket = ``ab/``. For our
training tree the chain identifier suffix (``_A``) doesn't affect the bucket;
all chains from one PDB ID land in the same bucket.

Some 2-char buckets in our PDB set exceed 1000 files (max observed: ``zk``
with 1535). For those we extend by ``pdb_id[1:4]`` (3 chars) — the maximum
the 4-char PDB ID allows. If 3 chars still exceeds the cap, we fall back to
a hash-based sub-bucket within the over-cap 3-char prefix.

CLI:
    # Build manifest from a list of stems (one per line; chains allowed):
    python -m script_utils.shard_pdb build --stems ~/proteina/data/pdb_train/stems.txt \
        --manifest ~/proteina/data/pdb_train/shard_manifest.json

    # Move raw or processed files into the sharded tree (stems on stdin):
    python -m script_utils.shard_pdb move \
        --manifest ~/proteina/data/pdb_train/shard_manifest.json \
        --src ~/proteina/data/pdb_train/processed \
        --dst /orcd/compute/so3/001/chenxi/pdb_train/processed \
        --suffix .pt --workers 32 --execute
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def pdb_id_from_stem(stem):
    """Extract the 4-char PDB ID from a stem like ``1abc`` or ``1abc_A`` or ``1abc_AB``."""
    base = stem.split("_", 1)[0]
    if len(base) != 4:
        raise ValueError(f"PDB ID must be 4 chars, got {base!r} from stem {stem!r}")
    return base


def _bucket_for(pdb_id, depth_table, hash_subbucket_for):
    """Pure bucket lookup, no I/O. depth_table[pdb_id[1:3]] = 2 or 3.

    For 3-depth buckets that are still over cap, hash_subbucket_for[pdb_id[1:4]]
    gives the hash-bucket prefix length (typically 1 hex char).
    """
    mid2 = pdb_id[1:3]
    depth = depth_table.get(mid2, 2)
    if depth == 2:
        return mid2
    # depth == 3
    mid3 = pdb_id[1:4]
    extra = hash_subbucket_for.get(mid3)
    if extra is None:
        return mid3
    # extra-deep: hash-bucket within mid3
    h = hashlib.md5(pdb_id.encode()).hexdigest()[:extra]
    return f"{mid3}/{h}"


def _build_depth_table(pdb_ids, max_per_bucket):
    """Pick depth (2 or 3) per top-level mid-two bucket. For 3-depth buckets that
    still exceed cap, register them in hash_subbucket_for with N-hex-char depth.
    """
    by_2 = defaultdict(list)
    for p in pdb_ids:
        by_2[p[1:3]].append(p)

    depth_table = {}
    hash_subbucket_for = {}
    for mid2, members in by_2.items():
        if len(members) <= max_per_bucket:
            depth_table[mid2] = 2
            continue
        # Try depth 3
        by_3 = defaultdict(list)
        for p in members:
            by_3[p[1:4]].append(p)
        if max(len(v) for v in by_3.values()) <= max_per_bucket:
            depth_table[mid2] = 3
            continue
        # Some 3-bucket still over cap -> use hash sub-bucket
        depth_table[mid2] = 3
        for mid3, sub_members in by_3.items():
            if len(sub_members) <= max_per_bucket:
                continue
            # Pick smallest hex-prefix that puts each sub-bucket under cap
            for nhex in (1, 2, 3):
                by_h = defaultdict(int)
                for p in sub_members:
                    h = hashlib.md5(p.encode()).hexdigest()[:nhex]
                    by_h[h] += 1
                if max(by_h.values()) <= max_per_bucket:
                    hash_subbucket_for[mid3] = nhex
                    break
            else:
                hash_subbucket_for[mid3] = 3  # last resort
    return depth_table, hash_subbucket_for


def id_to_bucket(stem, manifest):
    """Public bucket function used by move-side callers."""
    pdb_id = pdb_id_from_stem(stem)
    return _bucket_for(pdb_id, manifest["depth_table"], manifest["hash_subbucket_for"])


def build_manifest(stems_path, manifest_path, max_per_bucket):
    stems = [s.strip() for s in Path(stems_path).read_text().splitlines() if s.strip()]
    pdb_ids = sorted({pdb_id_from_stem(s) for s in stems})
    depth_table, hash_subbucket_for = _build_depth_table(pdb_ids, max_per_bucket)
    manifest = {
        "max_per_bucket": max_per_bucket,
        "n_pdb_ids": len(pdb_ids),
        "n_stems": len(stems),
        "depth_table": depth_table,
        "hash_subbucket_for": hash_subbucket_for,
    }
    # Verify
    bucket_counts = defaultdict(int)
    for s in stems:
        bucket_counts[_bucket_for(pdb_id_from_stem(s), depth_table, hash_subbucket_for)] += 1
    manifest["n_buckets"] = len(bucket_counts)
    manifest["max_bucket_count"] = max(bucket_counts.values()) if bucket_counts else 0
    over_cap = {b: c for b, c in bucket_counts.items() if c > max_per_bucket}
    manifest["over_cap_buckets"] = over_cap

    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}: pdb_ids={manifest['n_pdb_ids']} "
          f"stems={manifest['n_stems']} buckets={manifest['n_buckets']} "
          f"max_bucket_count={manifest['max_bucket_count']} "
          f"depth3={sum(1 for v in depth_table.values() if v == 3)} "
          f"hash_sub={len(hash_subbucket_for)} over_cap={len(over_cap)}")


def _move_one(src_path_str, dst_path_str):
    """Cross-FS move = copy + unlink. shutil.move handles both cases."""
    dst_parent = os.path.dirname(dst_path_str)
    os.makedirs(dst_parent, exist_ok=True)
    if os.path.exists(dst_path_str):
        return ("skip", src_path_str)
    shutil.move(src_path_str, dst_path_str)
    return ("ok", src_path_str)


def move(manifest_path, src_dir, dst_dir, suffix, workers, execute):
    manifest = json.loads(Path(manifest_path).read_text())
    src_dir = Path(src_dir).expanduser()
    dst_dir = Path(dst_dir).expanduser()
    payloads = []
    missing = 0
    n_total = 0
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        n_total += 1
        try:
            bucket = id_to_bucket(stem, manifest)
        except ValueError as e:
            missing += 1
            continue
        src = src_dir / name
        dst = dst_dir / bucket / name
        payloads.append((str(src), str(dst)))
    print(f"planned: {len(payloads)} moves (skipped {missing} non-PDB-named, "
          f"total {n_total})", flush=True)
    if not execute:
        for s, d in payloads[:5]:
            print(f"  dry-run: {s} -> {d}")
        print(f"  ... ({len(payloads)} total) [pass --execute to run]")
        return
    ok = 0
    skipped = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_move_one, s, d) for s, d in payloads]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                kind, _ = fut.result()
            except Exception as e:
                kind = "fail"
                failed += 1
                print(f"  FAIL: {e}", flush=True)
                continue
            if kind == "ok":
                ok += 1
            elif kind == "skip":
                skipped += 1
            if i % 20000 == 0:
                print(f"  moved {ok} skipped {skipped} failed {failed} "
                      f"({i}/{len(payloads)})", flush=True)
    print(f"DONE: ok={ok} skipped={skipped} failed={failed}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    pb = sub.add_parser("build")
    pb.add_argument("--stems", required=True,
                    help="File with one stem per line (e.g. 1abc_A or just 1abc).")
    pb.add_argument("--manifest", required=True)
    pb.add_argument("--max-per-bucket", type=int, default=1000)

    pm = sub.add_parser("move")
    pm.add_argument("--manifest", required=True)
    pm.add_argument("--src", required=True)
    pm.add_argument("--dst", required=True)
    pm.add_argument("--suffix", required=True, help="e.g. .pt or .cif.gz")
    pm.add_argument("--workers", type=int, default=32)
    pm.add_argument("--execute", action="store_true")

    args = p.parse_args()
    if args.cmd is None:
        p.print_help()
        return 2
    if args.cmd == "build":
        build_manifest(args.stems, args.manifest, args.max_per_bucket)
    elif args.cmd == "move":
        move(args.manifest, args.src, args.dst, args.suffix, args.workers, args.execute)
    return 0


if __name__ == "__main__":
    sys.exit(main())
