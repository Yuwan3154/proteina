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
"""

from typing import Dict, Optional

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_STRAND,
    MASK_TOKEN,
    N_PAIR_FEATURES,
    STRUCTURAL_PAIR_FEATURES,
    SSEAlphabet,
    circuit_topology_features,
    element_positions,
    perturb_runs,
    sse_sequence_gap,
)


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
        exact_max: int = 10,
        bin_step: int = 2,
        catch_all_above: int = 30,
        min_len: int = 1,
        seed: int = 0,
    ):
        self.index_path = index_path
        self.max_topology_len = max_topology_len
        self.max_topology_he_len = max_topology_he_len
        self.sigma_frac = sigma_frac
        self.mutate_prob = mutate_prob
        self.drop_prob = drop_prob
        self.self_fallback = self_fallback
        self.alphabet = SSEAlphabet(
            exact_max=exact_max, bin_step=bin_step, catch_all_above=catch_all_above, min_len=min_len
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
        """The [T, T, 3] channels that only the index can supply (see sse_topology)."""
        idx = self._index
        n_struct = len(STRUCTURAL_PAIR_FEATURES)
        if not self.has_pair_features or size <= 0:
            return torch.zeros(size, size, n_struct)
        a, b = int(idx["feat_offset"][row]), int(idx["feat_offset"][row + 1])
        if b - a != size * size * n_struct:
            return torch.zeros(size, size, n_struct)
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
        """A same-cluster chain with a different sequence, or the query itself as fallback."""
        idx = self._index
        cl = int(idx["cluster_of"][row])
        lo, hi = int(idx["members_offset"][cl]), int(idx["members_offset"][cl + 1])
        members = idx["members_flat"][lo:hi]
        if members.numel() <= 1:
            return row
        own = idx["seq_hash"][row]
        cand = members[idx["seq_hash"][members.long()] != own]
        if cand.numel() == 0:
            return row
        j = int(torch.randint(cand.numel(), (1,), generator=self._generator))
        return int(cand[j])

    def _build_reference(self, t_row: int, length: int, augment: bool) -> Dict[str, torch.Tensor]:
        """The six tensors the model consumes, for index row ``t_row`` rescaled onto a
        ``length``-residue query.

        ``augment=False`` skips the length perturbation AND consumes no RNG, which is what the
        ground-truth (self-reference) path needs: it must describe the query exactly, and it runs
        outside the dataloader where drawing from ``self._generator`` would be meaningless.
        """
        runs = self._runs_for(t_row)
        he_contact = self._he_contact_for(t_row)
        keep = [i for i, (t, _) in enumerate(runs) if t in (DSSP_HELIX, DSSP_STRAND)]
        # The stored map was built from the unperturbed runs, so its axis must stay aligned with
        # them even if augmentation changes element lengths (which never changes their count).
        structural = self._he_structural_for(t_row, he_contact.shape[0])
        if he_contact.shape[0] != len(keep):
            he_contact = torch.zeros(len(keep), len(keep))
            structural = torch.zeros(len(keep), len(keep), len(STRUCTURAL_PAIR_FEATURES))

        if augment and self.mutate_prob > 0.0 and self.sigma_frac > 0.0:
            runs = perturb_runs(
                runs,
                self.sigma_frac,
                self.mutate_prob,
                self._generator,
                min_len=self.alphabet.min_len,
            )

        tokens = torch.tensor(self.alphabet.runs_to_tokens(runs), dtype=torch.long)
        pos = element_positions(runs, target_len=length)
        he_tokens = torch.tensor(
            [self.alphabet.token(*runs[i]) for i in keep], dtype=torch.long
        )
        he_pos = pos[keep] if len(keep) else torch.zeros(0, dtype=torch.float32)

        tokens = tokens[: self.max_topology_len]
        pos = pos[: self.max_topology_len]
        k = min(len(keep), self.max_topology_he_len)
        he_tokens, he_pos = he_tokens[:k], he_pos[:k]
        he_contact = he_contact[:k, :k]
        he_feat = self._pair_features(he_contact, structural[:k, :k], runs, keep[:k])

        return {
            "topology_tokens": tokens if tokens.numel() else torch.full((1,), MASK_TOKEN, dtype=torch.long),
            "topology_pos": pos if pos.numel() else torch.zeros(1, dtype=torch.float32),
            "topology_he_tokens": (
                he_tokens if he_tokens.numel() else torch.full((1,), MASK_TOKEN, dtype=torch.long)
            ),
            "topology_he_pos": he_pos if he_pos.numel() else torch.zeros(1, dtype=torch.float32),
            "topology_he_contact": he_contact if he_contact.numel() else torch.zeros(1, 1),
            "topology_he_feat": he_feat if he_feat.numel() else torch.zeros(1, 1, N_PAIR_FEATURES),
        }

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

    def forward(self, graph: Data) -> Data:
        self._ensure_loaded()
        L = int(graph.coords.shape[0])
        stem = str(getattr(graph, "protein_id", getattr(graph, "id", "")))
        row = self._id_to_row.get(stem)

        drop = float(torch.rand(1, generator=self._generator)) < self.drop_prob
        if row is None or drop:
            return self._set_empty(graph)

        t_row = self._pick_template(row)  # returns `row` itself when no valid template exists
        if t_row == row and not self.self_fallback:
            return self._set_empty(graph)
        if not self._runs_for(t_row):
            t_row = row

        for key, value in self._build_reference(t_row, L, augment=True).items():
            setattr(graph, key, value)
        return graph

    def _set_empty(self, graph: Data) -> Data:
        """The no-reference case: a single MASK element, which the model treats as unconditioned."""
        graph.topology_tokens = torch.full((1,), MASK_TOKEN, dtype=torch.long)
        graph.topology_pos = torch.zeros(1, dtype=torch.float32)
        graph.topology_he_tokens = torch.full((1,), MASK_TOKEN, dtype=torch.long)
        graph.topology_he_pos = torch.zeros(1, dtype=torch.float32)
        graph.topology_he_contact = torch.zeros(1, 1)
        graph.topology_he_feat = torch.zeros(1, 1, N_PAIR_FEATURES)
        return graph

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_topology_len={self.max_topology_len}, "
            f"sigma_frac={self.sigma_frac}, mutate_prob={self.mutate_prob}, "
            f"drop_prob={self.drop_prob})"
        )
