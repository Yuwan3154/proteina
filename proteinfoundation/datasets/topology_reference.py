# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Attaches a retrieved, augmented topology reference to each training example.

The reference is another chain's fold description, not the query's own: a chain is drawn from the
query's 25%-identity cluster, excluding mates that share its sequence, so the model learns to
realise a topology it is given rather than to copy itself. Chains with no different-sequence mate
(6.66% of the training split, measured) fall back to their own topology.

Everything the transform needs comes from a precomputed flat index, so no second .pt is read per
sample. Element positions are rescaled from the template's length onto the query's, which is what
lets cross-attention relate a topology element to a query residue at all.

reference_source="synthetic" replaces the cluster draw with a partially-diffused variant of the
query's OWN structure (a synthetic template, TM to the native inside a configured range), read
from an index built by utils/precompute_synthetic_topology_index.py. Every chain then has a
non-self reference regardless of cluster size; a chain that still has none falls back to the
unconditional MASK reference and says so loudly (topology_missing_ref = 1 + a warning).
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch_geometric.transforms as T
from loguru import logger
from torch_geometric.data import Data

from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_STRAND,
    MASK_TOKEN,
    N_PAIR_FEATURES,
    PAIR_FEATURE_NAMES,
    SSE_TYPES,
    STRUCTURAL_PAIR_FEATURES,
    SSEAlphabet,
    circuit_topology_features,
    element_positions,
    perturb_runs,
    sse_sequence_gap,
)

# Sentinel written to graph.topology_ref_id when the sample is unconditioned (dropped, no index
# row, or -- under require_nonself -- no non-self template available). Downstream, a sample is
# SELF-referenced iff topology_ref_id == its own protein_id, and unconditioned iff it is this.
MASK_REF_ID = "MASK"

# Value of ref_align_target (a RESIDUE-axis tensor, so it is padded like the residues rather than
# like the topology_* keys) for a query residue that USalign leaves unaligned (or aligned to a
# reference element beyond the helix/strand cap). Also what the dense collate pads with.
ALIGN_NONE = -1


class TopologyReferenceTransform(T.BaseTransform):
    """Adds topology_tokens / topology_pos / topology_he_* to a graph.

    Args:
        index_path: file written by precompute_topology_index.py.
        max_topology_len: element axis is truncated to this; sequences longer than it lose their
            tail rather than silently reshaping the batch.
        max_topology_he_len: same, for the helix/strand axis of the 2D reference.
        sigma_frac: augmentation sigma as a fraction of each element's own length.
        mutate_prob: per-element probability of being perturbed.
        drop_prob: probability of replacing the whole reference with MASK, so the model can also
            run unconditioned (needed for classifier-free guidance at sampling time).
        self_fallback: use the query's own topology when no valid template exists.
        require_nonself: drop to MASK rather than ever returning the query's own
            topology. For the non-self validation arm.
        reference_source: "cluster" (a same-cluster, different-sequence chain -- the original
            scheme) or "synthetic" (a partially-diffused variant of the query's OWN native from a
            synthetic-template index; see utils/precompute_synthetic_topology_index). In synthetic
            mode the index groups each chain with its template rows, and the draw is uniform over
            the rows whose template-vs-native TM lies in ``tm_range``.
        tm_range: (lo, hi), inclusive, synthetic mode only.
        sse_types: DSSP types the token alphabet encodes. (1, 2) = helix+strand only, vocab 44,
            for a model that never reads loop tokens (tri).
        type_mutate_prob: augmentation -- per-element probability of flipping helix <-> strand.
        token_mask_prob: augmentation -- per-element probability of replacing the helix/strand
            token by MASK; the pre-mask token is emitted as ``topology_he_tokens_target`` for a
            masked-token loss. Both rates default to 0 = mechanism skipped, bit-identical output.
    """

    def __init__(
        self,
        index_path: str,
        max_topology_len: int = 128,
        max_topology_he_len: int = 64,
        sigma_frac: float = 0.15,
        mutate_prob: float = 0.3,
        drop_prob: float = 0.0,
        self_fallback: bool = True,
        require_nonself: bool = False,
        exact_max: int = 10,
        bin_step: int = 2,
        catch_all_above: int = 30,
        min_len: int = 1,
        seed: int = 0,
        reference_source: str = "cluster",
        tm_range: Tuple[float, float] = (0.5, 0.9),
        sse_types: Sequence[int] = SSE_TYPES,
        type_mutate_prob: float = 0.0,
        token_mask_prob: float = 0.0,
    ):
        if reference_source not in ("cluster", "synthetic"):
            raise ValueError(f"reference_source must be 'cluster' or 'synthetic', got {reference_source!r}")
        self.index_path = index_path
        self.max_topology_len = max_topology_len
        self.max_topology_he_len = max_topology_he_len
        self.sigma_frac = sigma_frac
        self.mutate_prob = mutate_prob
        self.drop_prob = drop_prob
        self.self_fallback = self_fallback
        # Validation-only: refuse the self-reference entirely, so the arm measures
        # template threading rather than the model's ability to copy the answer.
        self.require_nonself = require_nonself
        self.reference_source = reference_source
        self.tm_range = (float(tm_range[0]), float(tm_range[1]))
        self.type_mutate_prob = float(type_mutate_prob)
        self.token_mask_prob = float(token_mask_prob)
        self.alphabet = SSEAlphabet(
            exact_max=exact_max, bin_step=bin_step, catch_all_above=catch_all_above, min_len=min_len,
            types=tuple(int(t) for t in sse_types),
        )
        self.seed = seed
        self._index = None
        self._id_to_row = None
        self._generator = None
        self.has_pair_features = False
        self._feat_mean = torch.zeros(N_PAIR_FEATURES)
        self._feat_std = torch.ones(N_PAIR_FEATURES)

    # The index is loaded lazily so it is materialised once per process and inherited by forked
    # dataloader workers rather than being deserialised in each of them.
    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        # mmap: the index carries a per-element-pair feature block, so a resident copy in each of
        # the dataloader workers would multiply a multi-gigabyte allocation by num_workers. Memory
        # mapping leaves it in the page cache, shared by every worker.
        self._index = torch.load(
            self.index_path, map_location="cpu", weights_only=False, mmap=True
        )
        self._id_to_row = {s: i for i, s in enumerate(self._index["ids"])}
        # Mixed with torch's per-worker seed: every dataloader worker calls this with the same
        # self.seed, so a bare manual_seed would give all of them one identical stream of template
        # choices, jitter and dropout decisions.
        self._generator = torch.Generator().manual_seed(
            (self.seed + torch.initial_seed()) % (2**63)
        )
        self.has_pair_features = "feat_flat" in self._index
        if self.has_pair_features:
            self._feat_mean = self._index["pair_feature_mean"].float()
            self._feat_std = self._index["pair_feature_std"].float().clamp(min=1e-6)
            # Caught at construction rather than mid-epoch: an index built before the CA-CA
            # channels were dropped carries stats for a wider feature vector.
            if self._feat_mean.numel() != N_PAIR_FEATURES:
                raise ValueError(
                    f"{self.index_path} stores standardisation stats for "
                    f"{self._feat_mean.numel()} pair features but this code defines "
                    f"{N_PAIR_FEATURES} ({', '.join(PAIR_FEATURE_NAMES)}). Rebuild the index with "
                    f"precompute_synthetic_topology_index.py."
                )
        else:
            # An index built before the featurization still drives the contact-only mode: the
            # shape is unchanged, the structural channels read as zero, and standardisation is
            # the identity. The other modes need a rebuilt index to be meaningful.
            self._feat_mean = torch.zeros(N_PAIR_FEATURES)
            self._feat_std = torch.ones(N_PAIR_FEATURES)

    def _runs_for(self, row: int):
        idx = self._index
        a, b = int(idx["runs_offset"][row]), int(idx["runs_offset"][row + 1])
        if b <= a:
            return []
        return [(int(t), int(n)) for t, n in idx["runs_flat"][a:b].tolist()]

    def _he_contact_for(self, row: int) -> torch.Tensor:
        idx = self._index
        a, b = int(idx["he_offset"][row]), int(idx["he_offset"][row + 1])
        size = int(idx["he_size"][row])
        if size <= 0 or b <= a:
            return torch.zeros(0, 0)
        return idx["he_flat"][a:b].reshape(size, size).float()

    def _he_structural_for(self, row: int, size: int) -> torch.Tensor:
        """The [T, T, len(STRUCTURAL_PAIR_FEATURES)] channels only the index can supply."""
        idx = self._index
        n_struct = len(STRUCTURAL_PAIR_FEATURES)
        if not self.has_pair_features or size <= 0:
            return torch.zeros(size, size, n_struct)
        a, b = int(idx["feat_offset"][row]), int(idx["feat_offset"][row + 1])
        # A width mismatch means the index was built with a DIFFERENT feature set (the CA-CA
        # distance channels were dropped 2026-09-10). Returning zeros here would train the model on
        # silently blank structural features, so fail instead and name the rebuild.
        if b - a != size * size * n_struct:
            stored = (b - a) / max(size * size, 1)
            raise ValueError(
                f"index row {row} stores {stored:g} structural channels per element pair but this "
                f"code expects {n_struct} ({', '.join(STRUCTURAL_PAIR_FEATURES)}). The index at "
                f"{self.index_path} predates the feature-set change -- rebuild it with "
                f"precompute_synthetic_topology_index.py."
            )
        return idx["feat_flat"][a:b].reshape(size, size, n_struct).float()

    def _pair_features(
        self, contact: torch.Tensor, structural: torch.Tensor, runs, keep
    ) -> torch.Tensor:
        """Assemble and standardise the [T, T, N_PAIR_FEATURES] reference the model consumes.

        Circuit topology and the sequence gap are rebuilt here rather than read from the index:
        both must describe the reference AS THE MODEL SEES IT, i.e. after truncation to the
        helix/strand cap and after length augmentation.
        """
        circuit = circuit_topology_features(contact)
        gap = sse_sequence_gap(runs, keep)
        feat = torch.cat(
            [contact[..., None], structural, circuit, gap[..., None]], dim=-1
        )
        return (feat - self._feat_mean) / self._feat_std

    def _pick_template(self, row: int) -> int:
        """A same-cluster chain with a different sequence, or the query itself as fallback.

        Synthetic mode: a template row of the query's own group with TM inside ``tm_range``; -1
        when there is none (the caller decides what that means -- never the query itself).
        """
        idx = self._index
        cl = int(idx["cluster_of"][row])
        lo, hi = int(idx["members_offset"][cl]), int(idx["members_offset"][cl + 1])
        members = idx["members_flat"][lo:hi]
        if self.reference_source == "synthetic":
            if members.numel() == 0:
                return -1
            tm = idx["row_tm"][members.long()].float()
            cand = members[(tm >= self.tm_range[0]) & (tm <= self.tm_range[1])]
            if cand.numel() == 0:
                return -1
            j = int(torch.randint(cand.numel(), (1,), generator=self._generator))
            return int(cand[j])
        if members.numel() <= 1:
            return row
        own = idx["seq_hash"][row]
        cand = members[idx["seq_hash"][members.long()] != own]
        if cand.numel() == 0:
            return row
        j = int(torch.randint(cand.numel(), (1,), generator=self._generator))
        return int(cand[j])

    def _align_for(self, t_row: int, length: int, n_he: int) -> torch.Tensor:
        """Per-query-residue reference element index (into the helix/strand axis, before the cap)
        or ALIGN_NONE, from the index's USalign ground truth. All-NONE when the index has none or
        the stored vector was built for a different chain length."""
        idx = self._index
        if "align_offset" not in idx:
            return torch.full((length,), ALIGN_NONE, dtype=torch.long)
        a, b = int(idx["align_offset"][t_row]), int(idx["align_offset"][t_row + 1])
        if b - a != length:
            if b > a:
                logger.warning(
                    f"[topology] alignment vector for row {t_row} has length {b - a}, query has {length} "
                    "-- alignment target dropped for this sample"
                )
            return torch.full((length,), ALIGN_NONE, dtype=torch.long)
        align = idx["align_flat"][a:b].long()
        # elements beyond the helix/strand cap were never shown to the model
        return torch.where(align < n_he, align, torch.full_like(align, ALIGN_NONE))

    def assemble_reference(
        self,
        runs,
        he_contact: torch.Tensor,
        structural: torch.Tensor,
        length: int,
        augment: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """The topology_* tensors the model consumes, from raw run/contact/structural inputs.

        Public because the same assembly has to serve a reference read off a structure file (see
        ``utils/topology_from_structure``) as serves one read out of the index: it carries the
        truncation caps, the alphabet and the standardisation constants, and a second
        implementation of any of those would silently drift from what the model was trained on.

        ``augment=False`` skips the length perturbation AND consumes no RNG, which is what both
        the ground-truth (self-reference) path and the structure-file path need: they must
        describe their input exactly, and they run outside the dataloader where drawing from
        ``self._generator`` would be meaningless.
        """
        # The standardisation constants and the caps both live behind the lazy load, and an
        # external caller has no reason to know that.
        self._ensure_loaded()
        keep = [i for i, (t, _) in enumerate(runs) if t in (DSSP_HELIX, DSSP_STRAND)]
        if he_contact.shape[0] != len(keep):
            he_contact = torch.zeros(len(keep), len(keep))
            structural = torch.zeros(len(keep), len(keep), len(STRUCTURAL_PAIR_FEATURES))

        length_aug = self.mutate_prob > 0.0 and self.sigma_frac > 0.0
        if augment and (length_aug or self.type_mutate_prob > 0.0):
            runs = perturb_runs(
                runs,
                self.sigma_frac,
                self.mutate_prob if length_aug else 0.0,
                self._generator,
                min_len=self.alphabet.min_len,
                type_mutate_prob=self.type_mutate_prob,
            )
            # a type flip only swaps helix <-> strand, so `keep` and every per-element tensor
            # (contacts, structural features, alignment target) stay aligned with the runs

        # The 1D token axis carries exactly the runs the alphabet encodes: with the default
        # 3-type alphabet that is every run; with a helix+strand alphabet the loops are gone and
        # the positions must be filtered the same way or tokens and positions would misalign.
        tok_keep = [
            i for i, (t, n) in enumerate(runs) if t in self.alphabet.types and n >= self.alphabet.min_len
        ]
        tokens = torch.tensor([self.alphabet.token(*runs[i]) for i in tok_keep], dtype=torch.long)
        pos_all = element_positions(runs, target_len=length)
        # Un-rescaled midpoints: each element's own-chain residue index, origin 0, exactly like
        # the query's own indexing. A model that LEFT-ALIGNS query and reference (rather than
        # stretching the reference onto the query length) needs these, not the rescaled ones.
        pos_raw_all = element_positions(runs, target_len=None)
        pos = pos_all[tok_keep] if len(tok_keep) else torch.zeros(0, dtype=torch.float32)
        pos_raw = pos_raw_all[tok_keep] if len(tok_keep) else torch.zeros(0, dtype=torch.float32)
        he_tokens = torch.tensor(
            [self.alphabet.token(*runs[i]) for i in keep], dtype=torch.long
        )
        he_pos = pos_all[keep] if len(keep) else torch.zeros(0, dtype=torch.float32)

        he_pos_raw = pos_raw_all[keep] if len(keep) else torch.zeros(0, dtype=torch.float32)
        tokens = tokens[: self.max_topology_len]
        pos = pos[: self.max_topology_len]
        pos_raw = pos_raw[: self.max_topology_len]
        k = min(len(keep), self.max_topology_he_len)
        he_tokens, he_pos = he_tokens[:k], he_pos[:k]
        he_pos_raw = he_pos_raw[:k]
        he_contact = he_contact[:k, :k]
        he_feat = self._pair_features(he_contact, structural[:k, :k], runs, keep[:k])

        # Token masking (BERT-style, token identity only; positions and pair features stay): the
        # pre-mask token is the target for the masked-token loss, 0 (= PAD) where not masked.
        he_target = torch.zeros_like(he_tokens)
        if augment and self.token_mask_prob > 0.0 and he_tokens.numel():
            masked = torch.rand(he_tokens.shape, generator=self._generator) < self.token_mask_prob
            he_target = torch.where(masked, he_tokens, he_target)
            he_tokens = torch.where(masked, torch.full_like(he_tokens, MASK_TOKEN), he_tokens)

        return {
            "topology_tokens": tokens if tokens.numel() else torch.full((1,), MASK_TOKEN, dtype=torch.long),
            "topology_pos": pos if pos.numel() else torch.zeros(1, dtype=torch.float32),
            "topology_he_tokens": (
                he_tokens if he_tokens.numel() else torch.full((1,), MASK_TOKEN, dtype=torch.long)
            ),
            "topology_he_pos": he_pos if he_pos.numel() else torch.zeros(1, dtype=torch.float32),
            "topology_pos_raw": pos_raw if pos_raw.numel() else torch.zeros(1, dtype=torch.float32),
            "topology_he_pos_raw": (
                he_pos_raw if he_pos_raw.numel() else torch.zeros(1, dtype=torch.float32)
            ),
            "topology_he_contact": he_contact if he_contact.numel() else torch.zeros(1, 1),
            "topology_he_feat": he_feat if he_feat.numel() else torch.zeros(1, 1, N_PAIR_FEATURES),
            "topology_he_tokens_target": (
                he_target if he_target.numel() else torch.zeros(1, dtype=torch.long)
            ),
        }

    def _build_reference(self, t_row: int, length: int, augment: bool) -> Dict[str, torch.Tensor]:
        """``assemble_reference`` fed from index row ``t_row``."""
        runs = self._runs_for(t_row)
        he_contact = self._he_contact_for(t_row)
        # The stored map was built from the unperturbed runs, so its axis must stay aligned with
        # them even if augmentation changes element lengths (which never changes their count).
        structural = self._he_structural_for(t_row, he_contact.shape[0])
        return self.assemble_reference(runs, he_contact, structural, length, augment)

    def self_reference(self, stem: str, length: int) -> Optional[Dict[str, torch.Tensor]]:
        """The chain's OWN topology, unaugmented and never dropped.

        Used by validation sampling to condition on the correct answer, which measures whether the
        model can realise a topology it is given. Returns None for a chain the index does not
        cover, so the caller can fall back rather than condition on something invented.
        """
        self._ensure_loaded()
        row = self._id_to_row.get(stem)
        if row is None:
            return None
        return self._build_reference(row, length, augment=False)

    def nonself_reference(self, stem: str, length: int, seed: Optional[int] = None):
        """A RETRIEVED same-cluster, different-sequence topology. The realistic task.

        ⭐ Counterpart to self_reference for the validation SAMPLING path. self_reference conditions
        on the correct answer, so the headline
        `validation_sampling/contact_precision_at_L_*` is a CEILING, not a measurement: every one of
        the fixed validation chains gets its own topology. This returns a genuine template instead,
        which is what a test-time user actually has.

        Returns (features, reference_stem). reference_stem is returned rather than logged internally
        so the caller can record WHICH template was used -- without it the number cannot be
        stratified by reference quality afterwards, which is the whole point of measuring it.

        ⛔ Returns None when the chain has no different-sequence mate, rather than silently falling
        back to self. A silent fallback would put ceiling samples back into the arm that exists to
        exclude them, and the arm would quietly measure the thing it was built to avoid.

        seed makes the draw reproducible across sweep points, so a step-count sweep compares the
        SAME (query, reference) pairs at every step count instead of re-rolling the template and
        confounding the comparison.
        """
        self._ensure_loaded()
        row = self._id_to_row.get(stem)
        if row is None:
            return None
        if seed is not None:
            gen_saved, self._generator = self._generator, torch.Generator().manual_seed(
                (seed + row) % (2**63)
            )
        try:
            t_row = self._pick_template(row)
        finally:
            if seed is not None:
                self._generator = gen_saved
        if t_row < 0 or t_row == row or not self._runs_for(t_row):
            return None
        return self._build_reference(t_row, length, augment=False), str(self._index["ids"][t_row])

    def forward(self, graph: Data) -> Data:
        self._ensure_loaded()
        L = int(graph.coords.shape[0])
        stem = str(getattr(graph, "protein_id", getattr(graph, "id", "")))
        row = self._id_to_row.get(stem)

        drop = float(torch.rand(1, generator=self._generator)) < self.drop_prob
        if row is None or drop:
            graph.topology_ref_id = MASK_REF_ID
            if row is None and self.reference_source == "synthetic":
                return self._set_empty(graph, missing=True, stem=stem)
            return self._set_empty(graph, L=L)

        t_row = self._pick_template(row)  # cluster: `row` itself when no valid template; synthetic: -1
        if self.reference_source == "synthetic":
            if t_row < 0 or not self._runs_for(t_row):
                graph.topology_ref_id = MASK_REF_ID
                return self._set_empty(graph, missing=True, stem=stem)
            graph.topology_ref_id = str(self._index["ids"][t_row])
            feats = self._build_reference(t_row, L, augment=True)
            for key, value in feats.items():
                setattr(graph, key, value)
            n_he = int(feats["topology_he_tokens"].numel())
            graph.ref_align_target = self._align_for(t_row, L, n_he)
            graph.topology_missing_ref = torch.zeros(1, dtype=torch.long)
            return graph
        # ⭐ require_nonself: for a validation arm that measures the REALISTIC task (thread a
        # template that is not the answer). Without it, self-fallback silently hands back the
        # query's own topology and the metric becomes a ceiling rather than a measurement.
        if self.require_nonself and t_row == row:
            graph.topology_ref_id = MASK_REF_ID
            return self._set_empty(graph, L=L)
        if t_row == row and not self.self_fallback:
            graph.topology_ref_id = MASK_REF_ID
            return self._set_empty(graph, L=L)
        if not self._runs_for(t_row):
            t_row = row
        if self.require_nonself and t_row == row:
            graph.topology_ref_id = MASK_REF_ID
            return self._set_empty(graph, L=L)

        # ⛔ Set on EVERY exit path, not just this one. dense_padded_collate INTERSECTS keys across
        # the samples in a batch, so a key present on only some samples is silently DROPPED from the
        # batch entirely -- it would not misalign, it would vanish, and the stratification would
        # come back empty with no error.
        graph.topology_ref_id = str(self._index["ids"][t_row])
        feats = self._build_reference(t_row, L, augment=True)
        for key, value in feats.items():
            setattr(graph, key, value)
        # cluster mode carries no alignment ground truth unless the index was built with one
        graph.ref_align_target = self._align_for(t_row, L, int(feats["topology_he_tokens"].numel()))
        graph.topology_missing_ref = torch.zeros(1, dtype=torch.long)
        return graph

    def _set_empty(self, graph: Data, L: Optional[int] = None, missing: bool = False,
                   stem: str = "") -> Data:
        """The no-reference case: a single MASK element, which the model treats as unconditioned.

        ``missing=True`` is the synthetic-mode failure the run is NOT supposed to hit (a chain with
        no template in range): it is logged loudly per occurrence and flagged on the graph so the
        trainer can report its rate -- silence here would hide a data-coverage bug as "training".
        """
        if L is None:
            L = int(graph.coords.shape[0])
        if missing:
            logger.warning(
                f"[topology] NO synthetic reference for {stem!r} (TM range {self.tm_range}) -- "
                "falling back to UNCONDITIONAL. This is not expected; check template coverage."
            )
        graph.topology_tokens = torch.full((1,), MASK_TOKEN, dtype=torch.long)
        graph.topology_pos = torch.zeros(1, dtype=torch.float32)
        graph.topology_he_tokens = torch.full((1,), MASK_TOKEN, dtype=torch.long)
        graph.topology_he_pos = torch.zeros(1, dtype=torch.float32)
        graph.topology_pos_raw = torch.zeros(1, dtype=torch.float32)
        graph.topology_he_pos_raw = torch.zeros(1, dtype=torch.float32)
        graph.topology_he_contact = torch.zeros(1, 1)
        graph.topology_he_feat = torch.zeros(1, 1, N_PAIR_FEATURES)
        graph.topology_he_tokens_target = torch.zeros(1, dtype=torch.long)
        graph.ref_align_target = torch.full((L,), ALIGN_NONE, dtype=torch.long)
        graph.topology_missing_ref = torch.tensor([1 if missing else 0], dtype=torch.long)
        return graph

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source={self.reference_source}, tm_range={self.tm_range}, "
            f"max_topology_len={self.max_topology_len}, sigma_frac={self.sigma_frac}, "
            f"mutate_prob={self.mutate_prob}, type_mutate_prob={self.type_mutate_prob}, "
            f"token_mask_prob={self.token_mask_prob}, drop_prob={self.drop_prob}, "
            f"vocab={self.alphabet.vocab_size})"
        )
