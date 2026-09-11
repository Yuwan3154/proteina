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


BAND_KEYS = ("tm", "rewind", "length", "slot", "min_tm", "max_tm")


def load_band(path):
    """Materialise the arrays ONCE.

    ⛔ Indexing an NpzFile (`z[key][i]`) re-reads and re-decompresses the WHOLE array on every
    access. Doing that per row per key was 69k x 6 = 414k full-array decompressions: it ran for
    minutes and then OOM-killed a 48 GB job. Reading each array once turns the same work into
    ordinary numpy indexing.
    """
    z = np.load(path, allow_pickle=True)
    chains = [str(c) for c in z["chains"]]
    arrays = {k: np.asarray(z[k]) for k in BAND_KEYS if k in z}
    z.close()
    return {c: i for i, c in enumerate(chains)}, arrays


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
    n_by_tree = {t: len(bands[t][0]) for t in bands}
    for key in BAND_KEYS:
        if not all(key in bands[t][1] for t in bands):
            print(f"  key {key!r} absent from at least one source band index -- omitted")
            continue
        # min_tm / max_tm are 0-d band BOUNDS, one per file, not per-chain rows. Carry them through
        # as globals, and check the trees agree: a disagreement would mean the trees were pruned to
        # DIFFERENT bands, which silently changes what "in band" means for part of the pool.
        per_chain = all(bands[t][1][key].ndim >= 1 and bands[t][1][key].shape[0] == n_by_tree[t]
                        for t in bands)
        if per_chain:
            out[key] = np.stack([bands[tree][1][key][i] for _, tree, i in rows])
        else:
            vals = {t: np.asarray(bands[t][1][key]).ravel()[0] for t in bands}
            if len({round(float(v), 6) for v in vals.values()}) != 1:
                raise SystemExit(f"source trees disagree on {key}: {vals} -- they were pruned to "
                                 "different bands, so the pool would not have one definition of "
                                 "'in band'. Refusing to write.")
            out[key] = np.asarray(next(iter(vals.values())))
            print(f"  key {key!r} is a global band bound = {float(next(iter(vals.values()))):.3f} "
                  "(identical across all source trees)")
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
