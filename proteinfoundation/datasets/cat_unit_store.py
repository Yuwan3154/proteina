# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""On-disk unit store of the CAT-topology regime (T2 CATH + TED units, plus synthetic templates).

An ENTRY is one structure (a CATH/TED unit: the whole chain when its parent is < 512 residues, else a
domain cut; or one synthetic template). A MEMBER is (entry, CAT, domain): the unit of sampling and of
pairing. A multi-domain whole-chain entry has one member per CAT domain, so it is a template of every CAT
it has a domain in (cut to that domain) and a query for any of them.

Layout (every table is a TSV, every array an .npy so workers memory-map instead of unpickling):
  entries.tsv   entry_id kind(cath|ted|synthetic) src_unit n_res shard row
  members.tsv   member_id entry_id cat split(train|val|test) role(query|template) seg
                seg = half-open POSITION ranges into the entry's residues, "a-b;c-d" (the domain)
  shards/sNNNN/ offsets.npy (int64 [n+1])  aatype.npy (int8 [N])  resnum.npy (int32 [N])
                mask.npy (bool [N, 37])    coords.npy (float32 [N, 37, 3], AF atom37 order = openfold order)
                cbit_off.npy (int64 [n+1], BYTE offsets) + cbits.npy (uint8): the whole-chain Frame2ConFind map
                thresholded at 0.01 and bit-packed row-major (written by precompute_cat_pair_index)

Files per directory are bounded: one directory per shard of <= SHARD_ENTRIES entries.
"""

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SHARD_ENTRIES = 20000
ARRAYS = ("offsets", "aatype", "resnum", "mask", "coords")
CBIT_ARRAYS = ("cbit_off", "cbits")


def parse_seg(seg: str) -> List[Tuple[int, int]]:
    return [(int(a), int(b)) for a, b in (part.split("-") for part in seg.split(";"))]


def format_seg(ranges: List[Tuple[int, int]]) -> str:
    return ";".join(f"{a}-{b}" for a, b in ranges)


def seg_positions(seg: str) -> np.ndarray:
    """Sorted residue positions of a domain."""
    return np.concatenate([np.arange(a, b) for a, b in parse_seg(seg)])


def positions_to_seg(pos: np.ndarray) -> str:
    """Sorted unique positions -> half-open range string."""
    pos = np.unique(pos)
    breaks = np.flatnonzero(np.diff(pos) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(pos) - 1]])
    return format_seg([(int(pos[s]), int(pos[e]) + 1) for s, e in zip(starts, ends)])


def prime_page_cache(root: str, chunk: int = 64 << 20) -> int:
    """Read every shard array once, sequentially, so random member reads hit the page cache (35 GB took 95 s at 368 MB/s on
    SuperCloud; a random-access first touch costs ~50-100 ms per item). Returns the bytes read."""
    n = 0
    for path in sorted(glob.glob(os.path.join(root, "shards", "*", "*.npy"))):
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                n += len(block)
    return n


def write_shard(shard_dir: str, entries: List[Dict]) -> None:
    """entries: dicts with aatype [L] int, resnum [L] int, mask [L, 37] bool, coords [L, 37, 3] float."""
    os.makedirs(shard_dir, exist_ok=True)
    lengths = [int(e["aatype"].shape[0]) for e in entries]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.save(os.path.join(shard_dir, "offsets.npy"), offsets)
    np.save(os.path.join(shard_dir, "aatype.npy"), np.concatenate([e["aatype"] for e in entries]).astype(np.int8))
    np.save(os.path.join(shard_dir, "resnum.npy"), np.concatenate([e["resnum"] for e in entries]).astype(np.int32))
    np.save(os.path.join(shard_dir, "mask.npy"), np.concatenate([e["mask"] for e in entries]).astype(bool))
    np.save(os.path.join(shard_dir, "coords.npy"), np.concatenate([e["coords"] for e in entries]).astype(np.float32))


class UnitStore:
    """Read side. Arrays are memory-mapped lazily per process, so forked DataLoader workers share the page cache."""

    def __init__(self, root: str):
        self.root = root
        self.entries = pd.read_csv(os.path.join(root, "entries.tsv"), sep="\t", dtype={"entry_id": str, "src_unit": str})
        self.members = pd.read_csv(os.path.join(root, "members.tsv"), sep="\t", dtype={"member_id": str, "entry_id": str, "cat": str})
        assert self.entries.entry_id.is_unique, "entries.tsv: duplicate entry_id"
        assert self.members.member_id.is_unique, "members.tsv: duplicate member_id"
        missing = set(self.members.entry_id) - set(self.entries.entry_id)
        assert not missing, f"{len(missing)} members point to entries absent from entries.tsv, e.g. {sorted(missing)[:3]}"
        self._loc = {e: (int(s), int(r), int(n)) for e, s, r, n in zip(self.entries.entry_id, self.entries.shard, self.entries.row, self.entries.n_res)}
        self._arr: Dict[Tuple[int, str], np.ndarray] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arr"] = {}
        return state

    def _array(self, shard: int, name: str) -> np.ndarray:
        key = (shard, name)
        if key not in self._arr:
            self._arr[key] = np.load(os.path.join(self.root, "shards", f"s{shard:04d}", f"{name}.npy"), mmap_mode="r")
        return self._arr[key]

    def has_cbits(self, shard: int) -> bool:
        return os.path.exists(os.path.join(self.root, "shards", f"s{shard:04d}", "cbits.npy"))

    def entry(self, entry_id: str) -> Dict[str, np.ndarray]:
        """aatype, resnum, mask, coords of the whole entry (views into the memory map)."""
        shard, row, n = self._loc[entry_id]
        off = self._array(shard, "offsets")
        a, b = int(off[row]), int(off[row + 1])
        assert b - a == n, f"{entry_id}: stored length {b - a} != entries.tsv n_res {n}"
        return {k: self._array(shard, k)[a:b] for k in ("aatype", "resnum", "mask", "coords")}

    def contact_map(self, entry_id: str) -> np.ndarray:
        """Whole-chain thresholded Frame2ConFind map, float32 [n, n] 0/1."""
        shard, row, n = self._loc[entry_id]
        off = self._array(shard, "cbit_off")
        bits = np.asarray(self._array(shard, "cbits")[int(off[row]):int(off[row + 1])])
        return np.unpackbits(bits, count=n * n).reshape(n, n).astype(np.float32)
