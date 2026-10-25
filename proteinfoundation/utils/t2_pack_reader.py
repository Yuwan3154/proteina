# Vendored verbatim from Yuwan3154/openfold branch t2-indel-templates @b150357, prune_work/t2_indel/t2_pack_reader.py
# (the pack writer lives there; TemplatePack and FinalSet are used in proteina). Do not edit here: re-vendor.
"""Readers of the packed T2 template product (design: T2_PACKED_TEMPLATES_PLAN.md; user 10-11: both levels, float32, same pack for rescue chains). numpy only; fork-safe (files are memory-mapped lazily, per process, on first use).
ChainPack  (level 1): the per-chain npz files, byte-identical, concatenated into shards; get_chain(id) -> the npz object (np.load of the slice); read_template(id, i) = indel_pool.read_template on it. A chain appended from top-up rounds has one npz per round ('base' = the first production): rounds(id), and the optional rnd argument of raw / get_chain / read_template (default: the first = base).
TemplatePack (level 2): template-granular records; get(chain, draw[, rnd]) -> the dict of indel_pool.read_template (+ num_steps, source, round); draws(chain) -> the stored draw ids of all rounds; rows(chain) -> index rows. A template is keyed by (chain, round, draw).
FinalSet: the selection layer final_selection.npz (t2_select_diverse.py) over a pack; iterating yields (chain, the chosen templates as TemplatePack dicts).
Pack layout (written by t2_pack_build.py), in <root>/:
  chain.NNN.pack, chain.idx.npz (stems, shard, offsets, lengths, md5, round; a stem repeats once per round that has an npz of the chain)
  tmpl.NNN.bin, tmpl.idx.npz (one row per template: chain, draw, round, shard, off, nbytes, L, A, L_native, tm_native, tm_template, rewind, num_steps, source, model, arm, ops_len/ops_off into the utf-8 json blob ops_blob; chain_stems / chain_first / chain_count = the rows of each chain)
  metrics_all.csv.gz (every run variant of every chain: main and rescue runs, gate flags, source, rewind, num_steps; appended top-up rounds: source topup-production / topup-rewind100 and round, empty for the first production), MANIFEST.json
  final_selection.npz (optional, next to the pack): chain str[C], first i8[C], count i4[C] (rows of each chain), round str[R], draw i4[R]; the chosen templates of a chain are (round, draw) keys, round 'base' = the first production
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
        self.round = z["round"] if "round" in z.files else np.full(len(self.stems), "base")
        self.row, self.parts = {}, {}
        for i, s in enumerate(self.stems):
            self.row.setdefault(str(s), i)
            self.parts.setdefault(str(s), []).append(i)
        self._sh = _Shards(root, "chain", "pack")

    def __contains__(self, cid):
        return cid in self.row

    def rounds(self, cid):
        return [str(self.round[i]) for i in self.parts[cid]]

    def raw_row(self, i):
        o = int(self.offsets[i])
        return self._sh.get(int(self.shard[i]))[o:o + int(self.lengths[i])].tobytes()

    def part(self, cid, rnd=None):
        rnd = "base" if rnd is None else rnd
        rows = [i for i in self.parts[cid] if self.round[i] == rnd]
        if not rows:
            raise KeyError((cid, rnd, "no npz of this round for the chain", self.rounds(cid)))
        return rows[0]

    def raw(self, cid, rnd=None):
        return self.raw_row(self.part(cid, rnd))

    def get_chain(self, cid, rnd=None):
        return np.load(io.BytesIO(self.raw(cid, rnd)), allow_pickle=False)

    def read_template(self, cid, i, rnd=None):
        import indel_pool as ip
        return ip.read_template(io.BytesIO(self.raw(cid, rnd)), i)


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
        self.has_round = "round" in self.col._z.files
        self._sh = _Shards(root, "tmpl", "bin")

    def __contains__(self, cid):
        return cid in self.chain_row

    def rows(self, cid):
        c = self.chain_row[cid]
        return range(int(self.first[c]), int(self.first[c]) + int(self.count[c]))

    def draws(self, cid):
        return [int(self.col["draw"][r]) for r in self.rows(cid)]

    def round_of(self, r):
        return str(self.col["round"][r]) if self.has_round else "base"

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
                    num_steps=int(c["num_steps"][r]), source=str(c["source"][r]), round=self.round_of(r))

    def get(self, cid, draw, rnd=None):
        for r in self.rows(cid):
            if int(self.col["draw"][r]) == draw and (rnd is None or self.round_of(r) == rnd):
                return self._read(r)
        raise KeyError((cid, rnd, draw))

    def iter_chain(self, cid):
        for r in self.rows(cid):
            yield self._read(r)


class FinalSet:
    """The chosen templates per chain (final_selection.npz) read through a TemplatePack."""

    def __init__(self, root, path=None):
        self.tp = TemplatePack(root)
        z = np.load(path or os.path.join(root, "final_selection.npz"), allow_pickle=False)
        self.stems, self.first, self.count, self.round, self.draw = (z[k] for k in ("chain", "first", "count", "round", "draw"))
        self.row = {str(s): i for i, s in enumerate(self.stems)}

    def __contains__(self, cid):
        return cid in self.row

    def keys(self, cid):
        c = self.row[cid]
        return [(str(self.round[r]), int(self.draw[r])) for r in range(int(self.first[c]), int(self.first[c]) + int(self.count[c]))]

    def templates(self, cid):
        return [self.tp.get(cid, d, rd) for rd, d in self.keys(cid)]

    def __iter__(self):
        for s in self.stems:
            yield str(s), self.templates(str(s))
