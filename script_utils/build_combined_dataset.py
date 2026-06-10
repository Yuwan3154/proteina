#!/usr/bin/env python3
"""Build a combined PDB+AFDB data_dir for multi-source CAT-balanced training (Phase 2b).

Creates <out-dir>/processed/ as symlinks of every top-level bucket dir from the PDB and
AFDB processed trees (disjoint bucket names) + a merged shard_manifest.json (depth_table =
union, hash_subbucket_for = PDB's). Then verifies one real PDB stem and one real AFDB stem
both resolve to an existing .pt via the same bucket math as PDBDataset._processed_path_for.
Additive + reusable: existing symlinks are skipped (idempotent).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def resolve(stem, processed_dir, manifest):
    # mirrors PDBDataset._processed_path_for (pdb_data.py:669-705)
    fname = stem + ".pt"
    pid = stem.split("_", 1)[0]
    dt = manifest.get("depth_table", {})
    if len(pid) == 4 and pid[1:3] in dt:
        mid2 = pid[1:3]
        depth = dt.get(mid2, 2)
        if depth == 2:
            return processed_dir / mid2 / fname
        mid3 = pid[1:4]
        nhex = manifest.get("hash_subbucket_for", {}).get(mid3)
        if nhex is None:
            return processed_dir / mid3 / fname
        h = hashlib.md5(pid.encode()).hexdigest()[:nhex]
        return processed_dir / mid3 / h / fname
    if stem.startswith("AF-"):
        uniprot = stem[len("AF-"):].split("-", 1)[0]
        if len(uniprot) >= 2:
            depth = dt.get(uniprot[:2], 2)
            bucket = uniprot[: min(len(uniprot), depth)]
            return processed_dir / bucket / fname
    return processed_dir / fname


def first_stem(proc_dir, prefix=None):
    for entry in os.scandir(proc_dir):
        if not entry.is_dir():
            continue
        for f in os.scandir(entry.path):
            if f.name.endswith(".pt") and (prefix is None or f.name.startswith(prefix)):
                return f.name[:-3]
            if f.is_dir():  # depth-3 hash subbucket
                for g in os.scandir(f.path):
                    if g.name.endswith(".pt") and (prefix is None or g.name.startswith(prefix)):
                        return g.name[:-3]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--afdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    pdb, afdb, out = Path(args.pdb_dir), Path(args.afdb_dir), Path(args.out_dir)
    outp = out / "processed"
    outp.mkdir(parents=True, exist_ok=True)

    n_links = 0
    collisions = []
    for src in [pdb / "processed", afdb / "processed"]:
        for entry in os.scandir(src):
            dst = outp / entry.name
            if dst.is_symlink() or dst.exists():
                collisions.append(entry.name)
                continue
            os.symlink(entry.path, dst)
            n_links += 1

    with open(pdb / "shard_manifest.json") as fh:
        mp = json.load(fh)
    with open(afdb / "shard_manifest.json") as fh:
        ma = json.load(fh)
    dtp, dta = mp.get("depth_table", {}), ma.get("depth_table", {})
    overlap = sorted(set(dtp) & set(dta))
    assert not overlap, f"depth_table key collision (PDB vs AFDB): {overlap[:20]}"
    merged = {
        "depth_table": {**dtp, **dta},
        "hash_subbucket_for": {**ma.get("hash_subbucket_for", {}), **mp.get("hash_subbucket_for", {})},
        "n_buckets": mp.get("n_buckets", 0) + ma.get("n_buckets", 0),
    }
    with open(out / "shard_manifest.json", "w") as fh:
        json.dump(merged, fh)

    print(f"[build] symlinked {n_links} bucket dirs; pre-existing(skipped)={len(collisions)} {collisions[:8]}")
    print(f"[build] merged depth_table keys={len(merged['depth_table'])} n_buckets={merged['n_buckets']} overlap={len(overlap)}")

    pdb_stem = first_stem(pdb / "processed")
    afdb_stem = first_stem(afdb / "processed", prefix="AF-")
    for label, stem in [("PDB", pdb_stem), ("AFDB", afdb_stem)]:
        p = resolve(stem, outp, merged)
        ok = p.exists()
        print(f"[verify] {label} stem={stem} -> {p} exists={ok}")
        assert ok, f"{label} resolution FAILED: {p}"
    print("[build] DONE; both-source resolution OK")


if __name__ == "__main__":
    main()
