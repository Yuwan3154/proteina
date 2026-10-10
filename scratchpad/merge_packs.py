"""Concatenate shard packs written by build_pack.py into ONE pack + index (plan B for the T8 pack when a single-stream
build would exceed its time limit). Every entry is re-verified: md5(final blob slice) == the shard index's md5.
Stems must be disjoint across shards; the merged index is sorted by stem (as build_pack.py writes it).
Usage: python merge_packs.py OUT.pack SHARD1.pack [SHARD2.pack ...]
"""

import hashlib
import os
import shutil
import sys

import numpy as np


def main(out, shards):
    assert not os.path.exists(out), f"{out} exists -- one fresh pack per build"
    idx = [np.load(s + ".idx.npz", allow_pickle=True) for s in shards]
    for s, z in zip(shards, idx):
        size = int(z["offsets"][-1] + z["lengths"][-1]) if len(z["offsets"]) else 0
        assert os.path.getsize(s) == size, f"{s}: {os.path.getsize(s)} bytes vs index end {size}"
    all_stems = np.concatenate([z["stems"] for z in idx])
    assert len(set(all_stems.tolist())) == len(all_stems), "stems repeat across shards"
    stems, offsets, lengths, md5s, base = [], [], [], [], 0
    with open(out + ".partial", "wb") as fo:
        for s, z in zip(shards, idx):
            with open(s, "rb") as fi:
                shutil.copyfileobj(fi, fo, length=64 << 20)
            stems += z["stems"].tolist()
            offsets += (z["offsets"] + base).tolist()
            lengths += z["lengths"].tolist()
            md5s += z["md5"].tolist()
            base += os.path.getsize(s)
            print(f"[merge] {s}: {len(z['stems'])} entries, total {base / 1e9:.2f} GB", flush=True)
        fo.flush()
        os.fsync(fo.fileno())
    bad = 0
    with open(out + ".partial", "rb") as fh:
        for o, l, m in zip(offsets, lengths, md5s):
            fh.seek(o)
            bad += hashlib.md5(fh.read(l)).hexdigest() != m
    assert bad == 0, f"{bad} entries differ from their shard index after the merge"
    order = np.argsort(np.array(stems, dtype=object))
    np.savez(out + ".idx.npz", stems=np.array(stems, dtype=object)[order], offsets=np.array(offsets, np.int64)[order],
             lengths=np.array(lengths, np.int64)[order], md5=np.array(md5s, dtype=object)[order])
    os.replace(out + ".partial", out)
    print(f"[done] {len(stems)} entries from {len(shards)} shards, {base / 1e9:.2f} GB, all re-verified by md5 -> {out}")


if __name__ == "__main__":
    assert len(sys.argv) >= 3, __doc__
    main(sys.argv[1], sys.argv[2:])
