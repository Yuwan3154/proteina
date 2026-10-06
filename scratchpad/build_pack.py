"""Pack processed .pt graphs into one blob + index for PDBDataset's packed-mmap mode (user 2026-10-06: move the tri
training samples off the HDD store to so3/002 SSD as one file instead of ~71k).

Index format = what PDBDataset reads: OUT.idx.npz with stems (str), offsets, lengths (int64). The blob is the raw .pt
bytes concatenated, so torch.load(BytesIO(blob[off:off+len])) == torch.load(source). Every entry is re-verified by
md5(blob slice) == md5(source file) after the write.
Usage: python scratchpad/build_pack.py STEMS.txt DATA_DIR(pdb_train) OUT.pack
"""

import hashlib
import json
import os
import pathlib
import sys

import numpy as np

from proteinfoundation.datasets.pdb_data import _processed_path_sharded

stems_file, data_dir, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
stems = sorted({l.split()[0] for l in open(stems_file) if l.strip()})
man = json.load(open(data_dir / "shard_manifest.json"))
paths = [_processed_path_sharded(data_dir / "processed", s, man) for s in stems]
missing = [s for s, p in zip(stems, paths) if not p.exists()]
assert not missing, f"{len(missing)} stems have no source .pt, e.g. {missing[:10]}"
assert not os.path.exists(out), f"{out} exists -- one fresh pack per build"
offsets, lengths, md5s, off = [], [], [], 0
with open(out + ".partial", "wb") as fh:
    for i, p in enumerate(paths):
        b = p.read_bytes()
        fh.write(b)
        offsets.append(off); lengths.append(len(b)); md5s.append(hashlib.md5(b).hexdigest())
        off += len(b)
        if (i + 1) % 10000 == 0:
            print(f"[write] {i + 1}/{len(paths)} {off / 1e9:.2f} GB", flush=True)
    fh.flush(); os.fsync(fh.fileno())
bad = 0
with open(out + ".partial", "rb") as fh:
    for s, o, l, m in zip(stems, offsets, lengths, md5s):
        fh.seek(o)
        bad += hashlib.md5(fh.read(l)).hexdigest() != m
assert bad == 0, f"{bad} entries differ from their source after the write"
np.savez(out + ".idx.npz", stems=np.array(stems, dtype=object), offsets=np.array(offsets, np.int64),
         lengths=np.array(lengths, np.int64), md5=np.array(md5s, dtype=object))
os.replace(out + ".partial", out)
print(f"[done] {len(stems)} entries, {off / 1e9:.2f} GB, all {len(stems)} re-verified by md5 -> {out}")
