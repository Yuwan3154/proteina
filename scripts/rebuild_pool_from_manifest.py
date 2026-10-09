"""Recreate the auth-keyed template pool (assemble_auth_pool.py's symlink layout) from its pool_manifest.tsv on another
host, with the source trees moved: every row's source_path is rewritten by --remap OLD=NEW (longest prefix first) and
must exist. Same key -> dest/shard{crc32(key) % 1000:04d}/{key}.npz rule as the assembler. Refuses a non-empty dest.

  python scripts/rebuild_pool_from_manifest.py --manifest pool_manifest.tsv --dest POOL --remap OLD=NEW [--remap ...]
"""

import argparse
import collections
import os
import zlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--remap", action="append", required=True)
    a = ap.parse_args()
    remap = sorted((r.split("=", 1) for r in a.remap), key=lambda kv: -len(kv[0]))
    if os.path.isdir(a.dest) and os.listdir(a.dest):
        raise SystemExit(f"{a.dest} is not empty")
    rows, missing, unmapped = [], [], []
    with open(a.manifest) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        ik, isrc = head.index("auth_id"), head.index("source_path")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            src = f[isrc]
            new = next((n + src[len(o):] for o, n in remap if src.startswith(o.rstrip("/") + "/")), None)
            if new is None:
                unmapped.append(src)
            elif not os.path.exists(new):
                missing.append(new)
            else:
                rows.append((f[ik], new))
    print(f"manifest rows {len(rows) + len(missing) + len(unmapped)}: ok {len(rows)}, missing {len(missing)}, "
          f"unmapped {len(unmapped)}", flush=True)
    if missing or unmapped:
        raise SystemExit(f"refusing to build a partial pool, e.g. {(missing + unmapped)[:5]}")
    counts = collections.Counter()
    for key, new in rows:
        out = os.path.join(a.dest, f"shard{zlib.crc32(key.encode()) % 1000:04d}", f"{key}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.lexists(out):
            counts["already_present"] += 1  # assembler: two label chains resolving to one auth key
            continue
        os.symlink(new, out)
        counts["linked"] += 1
    print(dict(counts), flush=True)


if __name__ == "__main__":
    main()
