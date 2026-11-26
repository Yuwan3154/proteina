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
utils/precompute_cat_pair_index (or on-the-fly from UnitStore in reference_mode="mpnn_chunk9"). The per-residue
alignment target (ref_align_target) is the USalign of the query domain's CA onto the template domain's CA, made online
with the reference binary: it is pair-specific, so it cannot be baked in the way the per-native T8 index bakes it.
"""

import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple
import zlib

from loguru import logger
import numpy as np
import torch
from torch_geometric.data import Data

from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.datasets.cat_unit_store import UnitStore, prime_page_cache, seg_positions
from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.datasets.topology_reference import ALIGN_NONE, MASK_REF_ID, TopologyReferenceTransform
from proteinfoundation.nn.mpnn_topology_encoder import (
    MPNN_ATOM4_INDICES,
    TEMPLATE_SEQ_MASK_TOKEN,
    chunk9_runs,
)
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
    online USalign. Supports both SSE references (`reference_mode="sse"`) and fixed 9-residue ProteinMPNN chunk
    references (`reference_mode="mpnn_chunk9"`), serving template backbone coordinates, sequences (with 25% i.i.d.
    residue masking during training), and domain-cropped chunk9 coarse pair features directly from `UnitStore`.

    Args:
        usalign: path of the USalign binary.
        exclude_own_synthetic: also refuse synthetic templates edited from the query's own unit (a synthetic template of
            the query is a DIFFERENT entry, so by default it is allowed, as in the T8 regime).
        store_dir: optional path to the CAT `UnitStore` directory (defaults to `dirname(index_path)` if `entries.tsv`
            and `members.tsv` exist there). Used in `reference_mode="mpnn_chunk9"` to read template backbone coordinates,
            sequences, residue numbers, and precomputed Frame2ConFind contact maps without rebuilding `cat_pair_index.pt`.
        All other arguments: TopologyReferenceTransform (`reference_source` is fixed to "synthetic" so the parent's
            constructor accepts it; template choice and alignment are overridden here).
    """

    def __init__(
        self,
        index_path: str,
        usalign: str,
        exclude_own_synthetic: bool = False,
        store_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(index_path, reference_source="synthetic", **kwargs)
        self.usalign = usalign
        self.exclude_own_synthetic = bool(exclude_own_synthetic)
        if store_dir is None:
            cand = os.path.dirname(os.path.abspath(index_path))
            if os.path.exists(os.path.join(cand, "entries.tsv")) and os.path.exists(os.path.join(cand, "members.tsv")):
                store_dir = cand
        self.store_dir = store_dir
        self._unit_store: Optional[UnitStore] = None
        self._member_meta: Optional[Dict[str, Tuple[str, str]]] = None
        if self.reference_mode == "mpnn_chunk9" and self.store_dir is not None:
            # Allow loading CAT pool topology from an existing SSE cat_pair_index.pt while building
            # 9-residue chunk features + template coordinates on the fly from UnitStore.
            self._allow_element_type_mismatch = True

    def __getstate__(self):
        # a spawn-context DataLoader pickles the dataset into every worker; the preloaded multi-GB index and UnitStore
        # mmaps must not travel through shared memory, the worker reloads them lazily.
        state = self.__dict__.copy()
        state.update(
            _index=None,
            _id_to_row=None,
            _generator=None,
            _generator_pid=None,
            _unit_store=None,
            _member_meta=None,
        )
        return state

    def _ensure_loaded(self) -> None:
        fresh = self._index is None  # an early return here would skip the base class's per-worker generator re-seed
        super()._ensure_loaded()
        if fresh:
            for key in ("ca_flat", "elem_flat", "ca_offset", "row_entry", "row_pool", "pool_members_flat", "pool_members_offset", "row_src"):
                assert key in self._index, f"{self.index_path} lacks {key}: not an index of utils/precompute_cat_pair_index"
        if self.reference_mode == "mpnn_chunk9" and self._unit_store is None and self.store_dir is not None:
            st = UnitStore(self.store_dir)
            self._unit_store = st
            self._member_meta = {
                str(m): (str(e), str(s))
                for m, e, s in zip(st.members.member_id, st.members.entry_id, st.members.seg)
            }

    def _template_res_tensors_for(
        self, t_row: int, target_T: int, augment: bool
    ) -> Dict[str, torch.Tensor]:
        """Serve template domain backbone coords, mask, sequence (with 25% i.i.d. masking), and un-renumbered residue_idx from UnitStore."""
        if self._unit_store is None or self._member_meta is None:
            return super()._template_res_tensors_for(t_row, target_T=target_T, augment=augment)

        cap_res = max(target_T * self.chunk_size, 1)
        row_id = str(self._index["ids"][t_row]) if t_row < len(self._index["ids"]) else ""
        meta = self._member_meta.get(row_id)
        if meta is None:
            return super()._template_res_tensors_for(t_row, target_T=target_T, augment=augment)

        entry_id, seg = meta
        pos = seg_positions(seg)
        e = self._unit_store.entry(entry_id)
        full = torch.from_numpy(np.array(e["coords"][pos])).float()
        amask = torch.from_numpy(np.array(e["mask"][pos])).bool()
        aatype = torch.from_numpy(np.array(e["aatype"][pos])).long().clamp(0, 20)
        resnum = torch.from_numpy(np.array(e["resnum"][pos])).long()
        res_idx = (resnum - resnum[0]) if resnum.numel() > 0 else torch.zeros(0, dtype=torch.long)

        x4 = full[:cap_res, list(MPNN_ATOM4_INDICES), :]
        m4 = amask[:cap_res, list(MPNN_ATOM4_INDICES)].all(dim=-1).float()
        aatype = aatype[:cap_res]
        res_idx = res_idx[:cap_res]

        if x4.shape[0] == 0:
            x4 = torch.zeros((cap_res, 4, 3), dtype=torch.float32)
            m4 = torch.zeros((cap_res,), dtype=torch.float32)
            aatype = torch.full((cap_res,), TEMPLATE_SEQ_MASK_TOKEN, dtype=torch.long)
            res_idx = torch.arange(cap_res, dtype=torch.long)
        elif augment and self.template_seq_mask_prob > 0.0 and aatype.numel() > 0:
            seq_masked = torch.rand(aatype.shape, generator=self._generator) < self.template_seq_mask_prob
            aatype = torch.where(
                seq_masked,
                torch.full_like(aatype, TEMPLATE_SEQ_MASK_TOKEN),
                aatype,
            )

        return {
            "template_backbone_coords": x4,
            "template_backbone_mask": m4,
            "template_aatype": aatype,
            "template_residue_idx": res_idx,
        }

    def _resolve_row_reference_inputs(self, t_row: int):
        """In mpnn_chunk9 mode, either read precomputed chunk9 rows from the index or build them on the fly from UnitStore."""
        idx = self._index
        if self.reference_mode == "mpnn_chunk9":
            idx_dssp = str(idx.get("dssp_def", ""))
            built_elems = tuple(int(t) for t in idx.get("element_types", ()))
            if idx_dssp == "mpnn_chunk9" and sorted(built_elems) == sorted(self.element_types):
                a, b = int(idx["ca_offset"][t_row]), int(idx["ca_offset"][t_row + 1])
                L_t = max(b - a, 0)
                elem_chunk = torch.arange(L_t, dtype=torch.long) // self.chunk_size
                return self._runs_for(t_row), self._he_contact_for(t_row), self._he_structural_for(t_row, int(idx["he_size"][t_row])), None, elem_chunk

            if self._unit_store is not None and self._member_meta is not None:
                row_id = str(idx["ids"][t_row])
                entry_id, seg = self._member_meta[row_id]
                pos = seg_positions(seg)
                e = self._unit_store.entry(entry_id)
                full = torch.from_numpy(np.array(e["coords"][pos])).float()
                amask = torch.from_numpy(np.array(e["mask"][pos])).float()
                cm_t = torch.from_numpy(self._unit_store.contact_map(entry_id)[np.ix_(pos, pos)]).float()
                L_t = int(full.shape[0])
                runs = chunk9_runs(L_t, chunk_size=self.chunk_size)
                he_contact, keep = sse_contact_reference(cm_t, runs, keep_types=(DSSP_HELIX,))
                structural = sse_structural_pair_features(cm_t, full, amask, runs, keep)
                elem_chunk = torch.arange(L_t, dtype=torch.long) // self.chunk_size
                return runs, he_contact, structural, None, elem_chunk

        return super()._resolve_row_reference_inputs(t_row)

    def _has_valid_template_row(self, t_row: int) -> bool:
        if t_row < 0:
            return False
        if self.reference_mode == "mpnn_chunk9" and self._member_meta is not None:
            row_id = str(self._index["ids"][t_row]) if t_row < len(self._index["ids"]) else ""
            return row_id in self._member_meta
        return bool(self._runs_for(t_row))

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

    def nonself_reference(self, stem: str, length: int, seed: Optional[int] = None):
        """Validation sampling (model_trainer_base): a retrieved template of the member's (CAT, split) pool as (features, reference member id),
        reproducible per (seed, member); None when the member has no partner entry (never a silent fallback to its own topology)."""
        self._ensure_loaded()
        row = self._id_to_row.get(stem)
        if row is None:
            return None
        if seed is not None:
            saved, self._generator = self._generator, torch.Generator().manual_seed((seed + row) % (2**63))
        try:
            t_row = self._pick_cat_template(row)
        finally:
            if seed is not None:
                self._generator = saved
        if not self._has_valid_template_row(t_row):
            return None
        return self._build_reference(t_row, length, augment=False), str(self._index["ids"][t_row])

    def _cat_align(self, graph: Data, t_row: int, n_he: int) -> torch.Tensor:
        """Per-query-residue element index of the template (before the helix/strand or chunk9 cap), ALIGN_NONE where unaligned."""
        idx = self._index
        L = int(graph.coords.shape[0])
        a, b = int(idx["ca_offset"][t_row]), int(idx["ca_offset"][t_row + 1])
        t_ca = idx["ca_flat"][a:b].numpy()
        new_elem = getattr(self, "_last_new_elem_of_res", None)
        if new_elem is not None and int(new_elem.numel()) == (b - a):
            t_elem = new_elem.numpy()
        elif self.reference_mode == "mpnn_chunk9":
            t_elem = np.arange(b - a, dtype=np.int64) // self.chunk_size
        else:
            t_elem = idx["elem_flat"][a:b].numpy()
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
        missing = not self._has_valid_template_row(t_row)
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

    def __init__(self, store_dir: str, split: str, transform=None, fixed_pair_seed: Optional[int] = None, member_ids: Optional[List[str]] = None):
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
        if member_ids is not None:  # an explicit ordered subset (a member may be listed several times, as the fixed validation chain list allows)
            pos = {m: i for i, m in enumerate(self.members.member_id)}
            absent = [m for m in member_ids if m not in pos]
            assert not absent, f"{len(absent)} requested members are not {split} query members with a partner entry, e.g. {absent[:3]}"
            self.members = self.members.iloc[[pos[m] for m in member_ids]].reset_index(drop=True)
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

    def fixed_val_dataset(self, member_ids: List[str]) -> CATUnitDataset:
        """The trainer's fixed validation chain list (validation_sampling.fixed_chain_list) as a dataset of those val members."""
        return CATUnitDataset(self.store_dir, "val", transform=self.transform, fixed_pair_seed=self.cluster_sampler_seed, member_ids=member_ids)

    def _make_cat_pair_sampler(self, dataset: CATUnitDataset, shard: bool):
        return CATUniformSampler(dataset.members_by_cat(), seed=self.cluster_sampler_seed, shard=shard, fixed=dataset.split == "val",
                                 weights_by_cat=dataset.member_weights_by_cat(self.cath_draw_prob))

    def _cat_plan_members(self, dataset: CATUnitDataset) -> List[List[int]]:
        return dataset.members_by_cat()

    def _cat_plan_weights(self, dataset: CATUnitDataset) -> List[List[float]]:
        return dataset.member_weights_by_cat(self.cath_draw_prob)

    def _dataset_lengths(self, dataset) -> List[int]:
        return dataset.lengths
