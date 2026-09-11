"""Build the auth-keyed band index for the assembled pool.

The topology-index builder wants `--templates <tree>:<index_band.npz>`, where the band index carries
per-chain `tm / rewind / length / slot`. The pool is auth-keyed and drawn from three differently-keyed
trees, so its band index is derived from the POOL MANIFEST (auth_id, label_id, source_tree,
source_path) rather than by merging the source indexes independently: every row is looked up under the
name the file actually had in its source tree and re-emitted under the auth id it now has in the pool.

⛔ That ordering matters. Merging the three band indexes and renaming afterwards would silently keep
rows for chains that never made it into the pool, and would reintroduce exactly the kind of
name-based mismatch this whole fix removes. Deriving from the manifest means a band row exists iff
the corresponding npz exists.
⛔ Any pool entry with no band row is REPORTED, not defaulted -- a missing tm/slot cannot be invented.
"""

import argparse
import collections
import pathlib

import numpy as np


def load_band(path):
    z = np.load(path, allow_pickle=True)
    chains = [str(c) for c in z["chains"]]
    return {c: i for i, c in enumerate(chains)}, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="pool_manifest.tsv from assemble_auth_pool.py")
    ap.add_argument("--band", action="append", required=True,
                    help="tree_label=/path/to/index_band.npz (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bands = {}
    for spec in a.band:
        label, path = spec.split("=", 1)
        bands[label] = load_band(path)
        print(f"band index {label}: {len(bands[label][0])} chains from {path}")

    rows, missing = [], []
    counts = collections.Counter()
    with open(a.manifest) as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            auth_id, label_id, tree = f[0], f[1], f[2]
            src_name = pathlib.Path(f[3]).stem if len(f) > 3 else label_id
            idx_map = bands.get(tree)
            if idx_map is None:
                missing.append((auth_id, f"no band index supplied for tree {tree}"))
                counts["no_band_for_tree"] += 1
                continue
            m, z = idx_map
            i = m.get(src_name)
            if i is None:
                missing.append((auth_id, f"{src_name} absent from the {tree} band index"))
                counts["absent_from_band"] += 1
                continue
            rows.append((auth_id, tree, i))
            counts[f"ok_{tree}"] += 1

    if not rows:
        raise SystemExit("no rows resolved -- refusing to write an empty band index")

    out = {"chains": np.array([r[0] for r in rows])}
    for key in ("tm", "rewind", "length", "slot", "min_tm", "max_tm"):
        stacked = []
        for _, tree, i in rows:
            z = bands[tree][1]
            stacked.append(z[key][i] if key in z else None)
        if any(s is None for s in stacked):
            print(f"  key {key!r} absent from at least one source band index -- omitted")
            continue
        out[key] = np.stack(stacked)
    np.savez(a.out, **out)

    print(f"\nwrote {a.out}: {len(rows)} chains, keys {sorted(out)}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24} {v}")
    if missing:
        rep = pathlib.Path(a.out).with_suffix(".missing.tsv")
        with open(rep, "w") as fh:
            fh.write("auth_id\treason\n")
            for cid, why in missing:
                fh.write(f"{cid}\t{why}\n")
        print(f"  {len(missing)} pool entries WITHOUT a band row -> {rep}")
        for cid, why in missing[:5]:
            print(f"    {cid}: {why}")


if __name__ == "__main__":
    main()
