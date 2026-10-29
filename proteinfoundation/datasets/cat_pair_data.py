# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""CAT-topology regime: query/template pairs drawn inside a CAT topology (T2 CATH + TED units + synthetic templates).

Sampling unit = MEMBER (entry, CAT, domain), see cat_unit_store. One epoch = one uniformly drawn query member per CAT
(CATUniformSampler / EpochBatchPlanSampler with CAT as the "cluster"), so the loader is fold-balanced. Inside a CAT the
draw is source-stratified (user 2026-10-27, ESMFold2 mixture ratio): CATH with probability cath_draw_prob when the CAT-split has
both CATH and TED query entries, else the available source. The query is the
member's domain, cropped here (no DomainCropTransform: residue_pdb_idx stays un-renumbered, like DomainCrop). Its
reference is a random NON-SELF member of the same (CAT, split) pool -- a different entry, which may be a synthetic
template -- served through TopologyReferenceTransform.assemble_reference from the per-member index built by
utils/precompute_cat_pair_index. The per-residue alignment target (ref_align_target) is the USalign of the query
domain's CA onto the template domain's CA, made online with the reference binary: it is pair-specific, so it cannot be
baked in the way the per-native T8 index bakes it.
"""

import os
import tempfile
import zlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.datasets.cat_unit_store import UnitStore, prime_page_cache, seg_positions
from proteinfoundation.datasets.topology_reference import ALIGN_NONE, MASK_REF_ID, TopologyReferenceTransform
from proteinfoundation.utils.cluster_utils import CATUniformSampler
from proteinfoundation.utils.precompute_synthetic_topology_index import usalign_pairs

AA = "ARNDCQEGHILKMFPSTWYV"  # openfold restype order (the unit store's aatype)
CA_IDX = 1


def write_ca_pdb(ca_xyz: np.ndarray, present: np.ndarray, path: str) -> List[int]:
    """CA-only PDB, byte-identical to precompute_synthetic_topology_index.write_ca_pdb but formatted from python floats in one pass
    (the original spent ~10 ms per call in a per-atom generator); returns written_index -> structure_index."""
    keep = np.flatnonzero(present)
    lines = ["ATOM  %5d  CA  GLY A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n" % (k, k, x, y, z) for k, (x, y, z) in enumerate(ca_xyz[keep].tolist(), 1)]
    with open(path, "w") as fh:
        fh.write("".join(lines))
        fh.write("TER\nEND\n")
    return keep.tolist()


def _tmpdir() -> str:
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


class CATPairReferenceTransform(TopologyReferenceTransform):
    """TopologyReferenceTransform whose template is a same-(CAT, split) non-self member and whose alignment target is an
    online USalign. Everything else (augmentation, caps, standardisation, drop / MASK reference, the missing-reference
    flag and warning) is inherited unchanged, so the model sees the T8 reference in the T8 format.

    Args:
        usalign: path of the USalign binary.
        exclude_own_synthetic: also refuse synthetic templates edited from the query's own unit (a synthetic template of
            the query is a DIFFERENT entry, so by default it is allowed, as in the T8 regime).
        All other arguments: TopologyReferenceTransform (its ``reference_source`` is fixed to "synthetic" so the
        parent's constructor accepts it; template choice and alignment are overridden here).
    """

    def __init__(self, index_path: str, usalign: str, exclude_own_synthetic: bool = False, **kwargs):
        super().__init__(index_path, reference_source="synthetic", **kwargs)
        self.usalign = usalign
        self.exclude_own_synthetic = bool(exclude_own_synthetic)

    def __getstate__(self):
        # a spawn-context DataLoader pickles the dataset into every worker; the preloaded multi-GB index must not travel through
        # shared memory, the worker reloads it lazily. Forked workers inherit it without any pickling.
        state = self.__dict__.copy()
        state.update(_index=None, _id_to_row=None, _generator=None, _generator_pid=None)
        return state

    def _ensure_loaded(self) -> None:
        fresh = self._index is None  # an early return here would skip the base class's per-worker generator re-seed
        super()._ensure_loaded()
        if not fresh:
            return
        for key in ("ca_flat", "elem_flat", "ca_offset", "row_entry", "row_pool", "pool_members_flat", "pool_members_offset", "row_src"):
            assert key in self._index, f"{self.index_path} lacks {key}: not an index of utils/precompute_cat_pair_index"

    def _pick_cat_template(self, row: int) -> int:
        """Uniform draw over the members of row's (CAT, split) pool that belong to another entry; -1 when none."""
        idx = self._index
        p = int(idx["row_pool"][row])
        lo, hi = int(idx["pool_members_offset"][p]), int(idx["pool_members_offset"][p + 1])
        pool = idx["pool_members_flat"][lo:hi].long()
        keep = idx["row_entry"][pool] != idx["row_entry"][row]
        if self.exclude_own_synthetic:
            keep &= idx["row_src"][pool] != idx["row_src"][row]
        cand = pool[keep]
        if cand.numel() == 0:
            return -1
        return int(cand[int(torch.randint(cand.numel(), (1,), generator=self._generator))])

    def _cat_align(self, graph: Data, t_row: int, n_he: int) -> torch.Tensor:
        """Per-query-residue element index of the template (before the helix/strand cap), ALIGN_NONE where unaligned."""
        idx = self._index
        L = int(graph.coords.shape[0])
        a, b = int(idx["ca_offset"][t_row]), int(idx["ca_offset"][t_row + 1])
        t_ca, t_elem = idx["ca_flat"][a:b].numpy(), idx["elem_flat"][a:b].numpy()
        q_ca = graph.coords[:, CA_IDX].numpy()
        q_present = graph.coord_mask[:, CA_IDX].numpy().astype(bool)
        align = np.full((L,), ALIGN_NONE, dtype=np.int64)
        with tempfile.TemporaryDirectory(dir=_tmpdir()) as td:
            pq, pr = os.path.join(td, "q.pdb"), os.path.join(td, "r.pdb")
            qmap = write_ca_pdb(q_ca, q_present, pq)
            rmap = write_ca_pdb(t_ca, np.ones(len(t_ca), dtype=bool), pr)
            assert qmap and rmap, "a structure without a resolved CA reached the aligner"
            pairs, _ = usalign_pairs(pq, pr, len(qmap), len(rmap), binary=self.usalign)
        for qw, rw in pairs:
            align[qmap[qw]] = t_elem[rmap[rw]]
        out = torch.from_numpy(align)
        return torch.where(out < n_he, out, torch.full_like(out, ALIGN_NONE))

    def forward(self, graph: Data) -> Data:
        self._ensure_loaded()
        L = int(graph.coords.shape[0])
        stem = str(graph.protein_id)
        row = self._id_to_row.get(stem)
        pair_seed = getattr(graph, "pair_seed", None)  # fixed validation set: every draw of this item is a function of the member id
        if pair_seed is not None:
            del graph.pair_seed
            running, self._generator = self._generator, torch.Generator().manual_seed(int(pair_seed))
        drop = float(torch.rand(1, generator=self._generator)) < self.drop_prob
        # resolved BEFORE the drop is honoured, as in synthetic mode: topology_missing_ref then reports coverage, not coverage after dropout
        t_row = self._pick_cat_template(row) if row is not None else -1
        missing = t_row < 0 or not self._runs_for(t_row)
        if row is None or drop or missing:
            graph.topology_ref_id = MASK_REF_ID
            graph = self._set_empty(graph, L=L, missing=missing, stem=stem) if missing else self._set_empty(graph, L=L)
        else:
            graph.topology_ref_id = str(self._index["ids"][t_row])
            feats = self._build_reference(t_row, L, augment=True)
            for key, value in feats.items():
                setattr(graph, key, value)
            graph.ref_align_target = self._cat_align(graph, t_row, int(feats["topology_he_tokens"].numel()))
            graph.topology_missing_ref = torch.zeros(1, dtype=torch.long)
        if pair_seed is not None:
            self._generator = running
        return graph


class CATUnitDataset(torch.utils.data.Dataset):
    """Query members of one split. Item i = the member's domain cropped out of its entry, as the graph the T8 transform
    chain expects (openfold atom order, precomputed ``contact_map_confind``). Members whose (CAT, split) pool has no
    other entry are dropped (they cannot form a pair) and counted in the log; synthetic entries are never queries."""

    def __init__(self, store_dir: str, split: str, transform=None, fixed_pair_seed: Optional[int] = None):
        self.store = UnitStore(store_dir)
        self.transform = transform
        self.split = split
        self.fixed_pair_seed = fixed_pair_seed  # not None: the template and every reference draw of an item are fixed by (seed, member id)
        m = self.store.members
        pool = m[m["split"] == split]
        n_entries = pool.groupby("cat")["entry_id"].nunique()
        queries = pool[pool.role == "query"]
        has_partner = queries["cat"].map(n_entries) >= 2
        n_cats = queries[has_partner]["cat"].nunique()
        logger.info(f"CATUnitDataset[{split}]: {len(queries)} query members, {int((~has_partner).sum())} dropped "
                    f"(CAT pool holds a single entry), {n_cats} CATs")
        self.members = queries[has_partner].reset_index(drop=True)
        self.members["kind"] = self.members.entry_id.map(self.store.entries.set_index("entry_id").kind)
        self.lengths = [int(sum(b - a for a, b in self._ranges(s))) for s in self.members.seg]

    @staticmethod
    def _ranges(seg: str):
        return [tuple(int(x) for x in r.split("-")) for r in seg.split(";")]

    def __len__(self) -> int:
        return len(self.members)

    def members_by_cat(self) -> List[List[int]]:
        return [g.index.tolist() for _, g in self.members.groupby("cat")]

    def member_weights_by_cat(self, cath_draw_prob: float) -> List[List[float]]:
        """Draw weights aligned with members_by_cat(): inside a CAT the CATH members share cath_draw_prob and the TED members
        1 - cath_draw_prob (each source uniform over its members); a CAT with one source only draws it with probability 1.
        Only the real sources 'cath' and 'ted' are queries (synthetic entries never are)."""
        out = []
        for _, g in self.members.groupby("cat"):
            is_cath = (g["kind"] == "cath").to_numpy()
            n_c, n_t = int(is_cath.sum()), int((~is_cath).sum())
            assert n_c + n_t == len(g) and set(g["kind"]) <= {"cath", "ted"}, f"unexpected query kinds {set(g['kind'])}"
            p_c = cath_draw_prob if n_c and n_t else float(n_c > 0)
            out.append([p_c / n_c if c else (1.0 - p_c) / n_t for c in is_cath])
        return out

    def __getitem__(self, i: int) -> Data:
        mem = self.members.iloc[i]
        e = self.store.entry(mem.entry_id)
        pos = seg_positions(mem.seg)
        coords = torch.from_numpy(np.array(e["coords"][pos]))
        mask = torch.from_numpy(np.array(e["mask"][pos]))
        aatype = torch.from_numpy(np.array(e["aatype"][pos])).long()
        cm = self.store.contact_map(mem.entry_id)[np.ix_(pos, pos)]
        g = Data()
        g.id = mem.member_id
        g.protein_id = mem.member_id
        g.coords = coords
        g.coord_mask = mask
        g.residue_type = aatype
        g.sequence = "".join(AA[j] for j in aatype.tolist())
        g.residue_pdb_idx = torch.from_numpy(np.array(e["resnum"][pos])).long()
        g.seq_pos = torch.arange(coords.shape[0]).unsqueeze(-1)
        g.contact_map_confind = torch.from_numpy(cm)
        g.cath_code = [mem["cat"]]
        g.database = "cat"
        if self.fixed_pair_seed is not None:
            g.pair_seed = (self.fixed_pair_seed + zlib.crc32(mem.member_id.encode())) % (2**63)
        return self.transform(g) if self.transform else g


class CATPairLightningDataModule(BaseLightningDataModule):
    """Datamodule of the CAT regime. sampling_mode is fixed to "cat-pair"; batches are length-bucketed epoch plans
    (epoch_batch_plan: True) or a plain CATUniformSampler stream."""

    def __init__(self, store_dir: str, cath_draw_prob: float, prime_store: bool = False, **kwargs):
        super().__init__(sampling_mode="cat-pair", **kwargs)
        assert 0.0 <= cath_draw_prob <= 1.0, cath_draw_prob
        self.store_dir = store_dir
        self.data_dir = Path(store_dir)  # train.py names the dataset CSV from datamodule.data_dir before prepare_data
        self.cath_draw_prob = float(cath_draw_prob)
        self.prime_store = bool(prime_store)

    def setup(self, stage=None):
        super().setup(stage)
        # the reference index is loaded HERE, in the main process, so forked DataLoader workers share it (copy-on-write) instead of each
        # paying the file-system latency of a memory-mapped read per item (measured 60 ms of a ~95 ms warm item)
        for t in getattr(self.transform, "transforms", [self.transform]):
            if isinstance(t, CATPairReferenceTransform):
                t._ensure_loaded()
        if self.prime_store:
            logger.info(f"primed the page cache with {prime_page_cache(self.store_dir) / 1e9:.1f} GB of store arrays")

    def _get_dataset(self, split: str):
        fixed = self.cluster_sampler_seed if split == "val" else None
        return CATUnitDataset(self.store_dir, split, transform=self.transform, fixed_pair_seed=fixed)

    def _make_cat_pair_sampler(self, dataset: CATUnitDataset, shard: bool):
        return CATUniformSampler(dataset.members_by_cat(), seed=self.cluster_sampler_seed, shard=shard, fixed=dataset.split == "val",
                                 weights_by_cat=dataset.member_weights_by_cat(self.cath_draw_prob))

    def _cat_plan_members(self, dataset: CATUnitDataset) -> List[List[int]]:
        return dataset.members_by_cat()

    def _cat_plan_weights(self, dataset: CATUnitDataset) -> List[List[float]]:
        return dataset.member_weights_by_cat(self.cath_draw_prob)

    def _dataset_lengths(self, dataset) -> List[int]:
        return dataset.lengths
