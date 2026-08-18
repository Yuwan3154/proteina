# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""A 2D-only contact-map model: triangle multiplicative updates over a joint query+topology grid.

Deliberately the opposite design to ContactMapHierSiT, which mixes a 1D single track with a
pooled 2D hierarchy and cross-attends to the topology reference. Here there is no 1D track at
all, no attention of any kind, and no hierarchy: everything is one pair representation, updated
only by triangle multiplication plus the AF2 Evoformer transition.

The topology reference is not cross-attended -- it is CONCATENATED onto the token axis. With L
query residues and T reference elements the pair representation is (L+T) x (L+T), whose four
blocks are query-query, query-reference, reference-query and reference-reference. Triangle
multiplication then propagates between the query and the reference for free, through the same
mechanism it uses within each: a query pair (i, j) is updated by paths through reference elements
and vice versa.

Positions on that joint axis are LEFT-ALIGNED: query residue i sits at i, and reference element e
sits at its own-chain midpoint index (``topology_he_pos_raw``), both counted from 0. The
reference is NOT stretched onto the query's length -- an element a third of the way along a
short template stays at its own residue index, not at a third of the query.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinfoundation.datasets.sse_topology import MASK_TOKEN as TOPOLOGY_MASK_TOKEN
from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES, PAIR_FEATURE_NAMES
from proteinfoundation.openfold_stub.model.pair_transition import PairTransition
from proteinfoundation.openfold_stub.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from proteinfoundation.nn.contact_map_hier import PAIR_FEATURE_MODES, STRUCTURAL_PAIR_FEATURES

# Which region of the joint grid a cell belongs to. The model cannot infer this from the features
# alone (a zeroed query-reference cell looks like a zeroed query-query cell), and the four regions
# mean different things, so it is given explicitly.
BLOCK_QQ, BLOCK_QT, BLOCK_TQ, BLOCK_TT = 0, 1, 2, 3
N_BLOCK_TYPES = 4


class TimestepEmbedding(nn.Module):
    """Sinusoidal time embedding -> MLP, the usual diffusion conditioning stem."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1))
        )
        ang = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class TriBlock(nn.Module):
    """One reasoning step: outgoing + incoming triangle multiplication, then the AF2 transition.

    No triangle attention. Triangle multiplication is the part that moves information along
    i -> k -> j paths, which is what a contact map needs, and it is O(N^3 c) rather than
    attention's O(N^3 c) plus an N^2 x N attention matrix -- on an (L+T)^2 grid the memory
    difference is what decides whether the model fits at all.
    """

    def __init__(self, dim: int, tri_hidden: int, transition_n: int, dim_cond: int):
        super().__init__()
        self.tri_out = TriangleMultiplicationOutgoing(c_z=dim, c_hidden=tri_hidden)
        self.tri_in = TriangleMultiplicationIncoming(c_z=dim, c_hidden=tri_hidden)
        self.transition = PairTransition(c_z=dim, n=transition_n)
        # adaLN-Zero style gating on the time embedding: the block starts as identity, so depth
        # can be added without destabilising early training.
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(dim_cond, 3 * dim))
        nn.init.zeros_(self.cond[1].weight)
        nn.init.zeros_(self.cond[1].bias)

    def forward(self, z, pair_mask, cond):
        g_out, g_in, g_tr = self.cond(cond)[:, None, None, :].chunk(3, dim=-1)
        m = pair_mask[..., None]
        z = z + g_out.tanh() * self.tri_out(z, mask=pair_mask) * m
        z = z + g_in.tanh() * self.tri_in(z, mask=pair_mask) * m
        z = z + g_tr.tanh() * self.transition(z, mask=pair_mask) * m
        return z * m


class ContactMapTriSiT(nn.Module):
    """2D-only, triangle-multiplication contact-map denoiser over a joint query+topology grid."""

    def __init__(self, **kwargs):
        super().__init__()
        self.dim = int(kwargs["pair_dim"])
        self.tri_hidden = int(kwargs.get("tri_hidden", self.dim))
        self.n_blocks = int(kwargs["n_blocks"])
        self.transition_n = int(kwargs.get("transition_n", 4))  # AF2 Evoformer expansion
        self.dim_cond = int(kwargs.get("dim_cond", 128))
        self.max_topology_he_len = int(kwargs.get("max_topology_he_len", 64))
        self.max_rel_pos = int(kwargs.get("max_rel_pos", 64))
        self.topology_vocab_size = int(kwargs.get("topology_vocab_size", 65))
        self.pair_ref_features = kwargs.get("pair_ref_features", "both")
        if self.pair_ref_features not in PAIR_FEATURE_MODES:
            raise ValueError(
                f"pair_ref_features must be one of {sorted(PAIR_FEATURE_MODES)}, "
                f"got {self.pair_ref_features!r}"
            )
        self.pair_feat_idx = _pair_feature_indices(self.pair_ref_features)

        # Contract with the trainer, same as the other contact models.
        self.contact_map_mode = True
        self.contact_map_input_dim = int(kwargs.get("contact_map_input_dim", 1))
        self.non_contact_value = int(kwargs.get("non_contact_value", 0))
        self.predict_coords = False

        self.seq_emb = nn.Embedding(int(kwargs.get("n_residue_types", 22)), self.dim)
        self.topo_emb = nn.Embedding(
            self.topology_vocab_size, self.dim, padding_idx=0
        )
        self.block_type_emb = nn.Embedding(N_BLOCK_TYPES, self.dim)
        self.rel_pos_emb = nn.Embedding(2 * self.max_rel_pos + 2, self.dim)
        self.time_emb = TimestepEmbedding(self.dim_cond)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.dim_cond, self.dim_cond), nn.SiLU(),
            nn.Linear(self.dim_cond, self.dim_cond),
        )
        # Cell inputs: noised map, self-conditioning map, and the reference's own pair features.
        self.cell_in = nn.Linear(2 + len(self.pair_feat_idx), self.dim)

        self.blocks = nn.ModuleList(
            TriBlock(self.dim, self.tri_hidden, self.transition_n, self.dim_cond)
            for _ in range(self.n_blocks)
        )
        self.out_norm = nn.LayerNorm(self.dim)
        self.out = nn.Linear(self.dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, batch: Dict, force_compile: bool = False) -> Dict:
        cm_t = batch["contact_map_t"]
        B, L = cm_t.shape[0], cm_t.shape[1]
        device, dtype = cm_t.device, cm_t.dtype
        mask = batch["mask"]

        he_tokens = batch.get("topology_he_tokens")
        if he_tokens is None:
            he_tokens = torch.full((B, 1), TOPOLOGY_MASK_TOKEN, dtype=torch.long, device=device)
            he_pos = torch.zeros(B, 1, device=device)
            he_feat = torch.zeros(B, 1, 1, N_PAIR_FEATURES, device=device)
        else:
            he_pos = batch["topology_he_pos_raw"].float()
            he_feat = batch["topology_he_feat"].float()
        T = he_tokens.shape[1]
        he_valid = he_tokens > 0

        # ── joint token axis: [query residues | topology elements] ────────────────────────────
        tok_mask = torch.cat([mask.bool(), he_valid], dim=1)  # [B, N]
        pair_mask = (tok_mask[:, :, None] & tok_mask[:, None, :]).to(dtype)
        N = L + T

        # Left-aligned positions: query residue i at i, element e at its OWN-chain midpoint.
        q_pos = torch.arange(L, device=device, dtype=torch.float32)[None].expand(B, L)
        pos = torch.cat([q_pos, he_pos], dim=1)  # [B, N]
        rel = (pos[:, :, None] - pos[:, None, :]).round().long()
        rel = rel.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        z = self.rel_pos_emb(rel)

        # Region identity, so a zeroed query-reference cell is not confused with a query-query one.
        is_t = torch.zeros(N, dtype=torch.long, device=device)
        is_t[L:] = 1
        block_id = is_t[:, None] * 2 + is_t[None, :]  # qq=0, qt=1, tq=2, tt=3
        z = z + self.block_type_emb(block_id)[None]

        # Sequence identity enters as a 2D outer sum -- there is no 1D track to put it on.
        rtype = batch.get("residue_type")
        if rtype is not None:
            e = self.seq_emb(rtype.long().clamp(min=0))
            e = F.pad(e, (0, 0, 0, T))
            z = z + e[:, :, None, :] + e[:, None, :, :]
        te = self.topo_emb(he_tokens.clamp(min=0)) * he_valid[..., None]
        te = F.pad(te, (0, 0, L, 0))
        z = z + te[:, :, None, :] + te[:, None, :, :]

        # Per-cell scalar inputs, placed into their own blocks and zero elsewhere.
        cm_sc = batch.get("contact_map_sc")
        if cm_sc is None:
            cm_sc = torch.zeros_like(cm_t)
        cells = z.new_zeros(B, N, N, 2 + len(self.pair_feat_idx))
        cells[:, :L, :L, 0] = cm_t
        cells[:, :L, :L, 1] = cm_sc
        cells[:, L:, L:, 2:] = he_feat[..., self.pair_feat_idx].to(z.dtype)
        z = (z + self.cell_in(cells)) * pair_mask[..., None]

        cond = self.cond_mlp(self.time_emb(batch["t"]))
        for blk in self.blocks:
            z = blk(z, pair_mask, cond)

        logits = self.out(self.out_norm(z))[..., 0]
        logits = logits[:, :L, :L]
        logits = 0.5 * (logits + logits.transpose(1, 2))  # a contact map is symmetric by definition
        return {
            "contact_map_logits": logits,
            "contact_map_pred": torch.sigmoid(logits),
        }


def _pair_feature_indices(mode: str):
    """Which of the reference's element-pair channels this mode consumes."""
    names = list(PAIR_FEATURE_NAMES)
    circuit = [n for n in names if n.startswith("circuit_")]
    proximity = [n for n in STRUCTURAL_PAIR_FEATURES if n != "contact_frac"] + ["seq_gap", "contact_frac"]
    if mode == "contact":
        keep = ["contact_max"]
    elif mode == "circuit":
        keep = ["contact_max"] + circuit
    elif mode == "proximity":
        keep = ["contact_max"] + [n for n in proximity if n in names]
    else:
        keep = names
    return [names.index(n) for n in keep]
