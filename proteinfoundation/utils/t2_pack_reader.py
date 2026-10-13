# Vendored verbatim from Yuwan3154/openfold branch t2-indel-templates @392f129, prune_work/t2_indel/t2_pack_reader.py
# (the pack writer lives there; only TemplatePack is used in proteina). Do not edit here: re-vendor.
"""Readers of the packed T2 template product (design: T2_PACKED_TEMPLATES_PLAN.md; user 10-11: both levels, float32, same pack for rescue chains). numpy only; fork-safe (files are memory-mapped lazily, per process, on first use).
ChainPack  (level 1): the per-chain npz files, byte-identical, concatenated into shards; get_chain(id) -> the npz object (np.load of the slice); read_template(id, i) = indel_pool.read_template on it.
TemplatePack (level 2): template-granular records; get(chain, draw) -> the dict of indel_pool.read_template (+ num_steps, source); draws(chain) -> the stored draw ids; rows(chain) -> index rows.
Pack layout (written by t2_pack_build.py), in <root>/:
  chain.NNN.pack, chain.idx.npz (stems, shard, offsets, lengths, md5)
  tmpl.NNN.bin, tmpl.idx.npz (one row per template: chain, draw, shard, off, nbytes, L, A, L_native, tm_native, tm_template, rewind, num_steps, source, model, arm, ops_len/ops_off into the utf-8 json blob ops_blob; chain_stems / chain_first / chain_count = the rows of each chain)
  metrics_all.csv.gz (every run variant of every chain: main and rescue runs, gate flags, source, rewind, num_steps), MANIFEST.json
A template record = aatype i1[L] | residue_index i4[L] | orig_idx i4[L] | atom_mask u1[L*37] | coords f4[A*3], each array and the record itself starting at a multiple of 8 bytes (see layout()).
"""
import io
import json
import os

import numpy as np


def _al(n):
    return (n + 7) // 8 * 8


def layout(L, A):
    """Byte offsets of the five arrays of a template record and its total size."""
    o_ri = _al(L)
    o_oi = o_ri + _al(4 * L)
    o_am = o_oi + _al(4 * L)
    o_co = o_am + _al(37 * L)
    return dict(aatype=0, residue_index=o_ri, orig_idx=o_oi, atom_mask=o_am, coords=o_co, nbytes=_al(o_co + 12 * A))


class _Shards:
    def __init__(self, root, prefix, suffix):
        self.root, self.prefix, self.suffix, self._mm = root, prefix, suffix, {}

    def get(self, shard):
        if shard not in self._mm:
            self._mm[shard] = np.memmap(os.path.join(self.root, f"{self.prefix}.{shard:03d}.{self.suffix}"), dtype=np.uint8, mode="r")
        return self._mm[shard]


class ChainPack:
    def __init__(self, root):
        z = np.load(os.path.join(root, "chain.idx.npz"), allow_pickle=False)
        self.stems, self.shard, self.offsets, self.lengths, self.md5 = (z[k] for k in ("stems", "shard", "offsets", "lengths", "md5"))
        self.row = {str(s): i for i, s in enumerate(self.stems)}
        self._sh = _Shards(root, "chain", "pack")

    def __contains__(self, cid):
        return cid in self.row

    def raw(self, cid):
        i = self.row[cid]
        o = int(self.offsets[i])
        return self._sh.get(int(self.shard[i]))[o:o + int(self.lengths[i])].tobytes()

    def get_chain(self, cid):
        return np.load(io.BytesIO(self.raw(cid)), allow_pickle=False)

    def read_template(self, cid, i):
        import indel_pool as ip
        return ip.read_template(io.BytesIO(self.raw(cid)), i)


class _Cols:
    """Columns of tmpl.idx.npz loaded on first use (a DataLoader worker pays only for what it reads)."""

    def __init__(self, path):
        self._z, self._c = np.load(path, allow_pickle=False), {}

    def __getitem__(self, k):
        if k not in self._c:
            self._c[k] = self._z[k]
        return self._c[k]


class TemplatePack:
    def __init__(self, root):
        self.col = _Cols(os.path.join(root, "tmpl.idx.npz"))
        self.stems, self.first, self.count = self.col["chain_stems"], self.col["chain_first"], self.col["chain_count"]
        self.chain_row = {str(s): i for i, s in enumerate(self.stems)}
        self._sh = _Shards(root, "tmpl", "bin")

    def __contains__(self, cid):
        return cid in self.chain_row

    def rows(self, cid):
        c = self.chain_row[cid]
        return range(int(self.first[c]), int(self.first[c]) + int(self.count[c]))

    def draws(self, cid):
        return [int(self.col["draw"][r]) for r in self.rows(cid)]

    def _read(self, r):
        c = self.col
        L, A = int(c["L"][r]), int(c["A"][r])
        lay = layout(L, A)
        buf = self._sh.get(int(c["shard"][r]))
        o = int(c["off"][r])

        def arr(name, dtype, n):
            return np.frombuffer(buf[o + lay[name]:o + lay[name] + n * np.dtype(dtype).itemsize], dtype=dtype)

        return dict(aatype=arr("aatype", np.int8, L).copy(), atom_mask=arr("atom_mask", np.uint8, 37 * L).reshape(L, 37).astype(bool), coords=arr("coords", np.float32, 3 * A).reshape(A, 3).copy(),
                    residue_index=arr("residue_index", np.int32, L).copy(), orig_idx=arr("orig_idx", np.int32, L).copy(), arm=str(c["arm"][r]), model=str(c["model"][r]), rewind=int(c["rewind"][r]),
                    draw=int(c["draw"][r]), tm_native=float(c["tm_native"][r]), tm_template=float(c["tm_template"][r]), ops=json.loads(bytes(c["ops_blob"][int(c["ops_off"][r]):int(c["ops_off"][r]) + int(c["ops_len"][r])]).decode()), L_native=int(c["L_native"][r]),
                    num_steps=int(c["num_steps"][r]), source=str(c["source"][r]))

    def get(self, cid, draw):
        for r in self.rows(cid):
            if int(self.col["draw"][r]) == draw:
                return self._read(r)
        raise KeyError((cid, draw))

    def iter_chain(self, cid):
        for r in self.rows(cid):
            yield self._read(r)
