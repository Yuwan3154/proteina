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
from proteinfoundation.datasets.sse_topology import (
    N_PAIR_FEATURES,
    PAIR_FEATURE_MODES,
    PAIR_FEATURE_NAMES,
)
from proteinfoundation.nn.ot_alignment import OTAlignmentHead
from proteinfoundation.openfold_stub.model.pair_transition import PairTransition
from proteinfoundation.openfold_stub.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

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
        # AF2 convention, chosen and kept whole: the ONLY zero-init in each residual branch is
        # OpenFold's own output projection (`init="final"`), which is already inside tri_out,
        # tri_in and transition. Nothing multiplies their outputs.
        #
        # Time conditioning therefore modulates each sub-module's INPUT (FiLM scale+shift),
        # never its output. Zero-initialised, so the modulation starts as the identity rather
        # than as zero -- a multiplier of 0 on an output that is already 0 is what froze this
        # trunk in the first place.
        #
        # The AF3 alternative would have been to keep an output gate and make it sigmoid(-2.0)
        # = 0.119 like AdaptiveLayerNormOutputScale, and to remove OpenFold's zero-init from
        # under it. Either discipline is fine; mixing the two is not, and AF2's is the one that
        # comes for free with the OpenFold modules this block is built from.
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim_cond, 6 * dim))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    @staticmethod
    def _film(x, scale, shift):
        return x * (1.0 + scale) + shift

    def forward(self, z, pair_mask, cond):
        p = self.mod(cond)[:, None, None, :].chunk(6, dim=-1)
        m = pair_mask[..., None]
        z = z + self.tri_out(self._film(z, p[0], p[1]), mask=pair_mask) * m
        z = z + self.tri_in(self._film(z, p[2], p[3]), mask=pair_mask) * m
        z = z + self.transition(self._film(z, p[4], p[5]), mask=pair_mask) * m
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
        # None, not False, and the distinction is load-bearing: model_trainer_base tests
        # `predict_coords is False` (identity) in two places, and a False here made TRI ALONE flip
        # predict_from_dist on as soon as a distogram existed, landing in
        # _predict_structure_from_distogram whose two backends both `raise NotImplementedError`.
        # ContactMapHierSiT and ContactMapDiT already use None; this makes the three arms agree.
        # Structure-from-distogram remains reachable, but only when a config asks for it via
        # predict_structure_from_distogram, instead of switching itself on for one arm.
        self.predict_coords = None

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
        # Off unless a config asks for it, so the baseline arm stays bit-identical.
        ot_cfg = dict(kwargs.get("ot_align") or {})
        self.ot_align = None
        if ot_cfg.pop("enabled", False):
            self.ot_align = OTAlignmentHead(dim=self.dim, **ot_cfg)

        # Distogram head. `num_buckets_predict_pair` is the key protein_transformer already uses for
        # this, and it MUST equal loss.num_dist_buckets -- proteina.py asserts the two match. Absent
        # => no head and no `pair_logits`, so the auxiliary distogram loss stays inactive and the
        # baseline arm is unchanged.
        self.num_buckets_predict_pair = kwargs.get("num_buckets_predict_pair", None)
        self.dist_head = None
        if self.num_buckets_predict_pair is not None:
            self.dist_head = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, int(self.num_buckets_predict_pair)),
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
        q_feat = None
        if rtype is not None:
            e = self.seq_emb(rtype.long().clamp(min=0))
            q_feat = e  # the only per-token query features tri has; the OT head reads them
            e = F.pad(e, (0, 0, 0, T))
            z = z + e[:, :, None, :] + e[:, None, :, :]
        te = self.topo_emb(he_tokens.clamp(min=0)) * he_valid[..., None]
        r_feat = te
        te = F.pad(te, (0, 0, L, 0))
        z = z + te[:, :, None, :] + te[:, None, :, :]

        # Per-cell scalar inputs, placed into their own blocks and zero elsewhere.
        cm_sc = batch.get("contact_map_sc")
        has_sc = cm_sc is not None
        if cm_sc is None:
            cm_sc = torch.zeros_like(cm_t)
        cells = z.new_zeros(B, N, N, 2 + len(self.pair_feat_idx))
        cells[:, :L, :L, 0] = cm_t
        cells[:, :L, :L, 1] = cm_sc
        cells[:, L:, L:, 2:] = he_feat[..., self.pair_feat_idx].to(z.dtype)
        z = (z + self.cell_in(cells)) * pair_mask[..., None]

        if self.ot_align is not None:
            # Built from the self-conditioned map where available: the topology reference only
            # helps at t < 0.5, which is exactly where c_t is noisiest.
            inj = self.ot_align(
                q_feat=q_feat if q_feat is not None else z.new_zeros(B, L, self.dim),
                r_feat=r_feat,
                c_q=(cm_sc if has_sc else cm_t).to(z.dtype),
                c_r=he_feat[..., 0].to(z.dtype),
                q_mask=mask.bool(),
                r_mask=he_valid,
                he_pos=he_pos,
            )
            z[:, :L, L:] = z[:, :L, L:] + inj
            z[:, L:, :L] = z[:, L:, :L] + inj.transpose(1, 2)
            z = z * pair_mask[..., None]

        cond = self.cond_mlp(self.time_emb(batch["t"]))
        for blk in self.blocks:
            z = blk(z, pair_mask, cond)

        logits = self.out(self.out_norm(z))[..., 0]
        logits = logits[:, :L, :L]
        logits = 0.5 * (logits + logits.transpose(1, 2))  # a contact map is symmetric by definition
        # Padded cells carry a learned constant otherwise (LayerNorm of the masked-to-zero trunk
        # output is the bias, not zero), which ContactMapHierSiT does not have. Loss and metrics
        # both mask, so this changes no number -- it removes an asymmetry that made the two arms'
        # padding render differently and be read as a modelling difference.
        q_valid = mask.bool()
        q_pair = (q_valid[:, :, None] & q_valid[:, None, :])
        logits = logits * q_pair.to(logits.dtype)
        out = {
            "contact_map_logits": logits,
            "contact_map_pred": torch.sigmoid(logits),
        }

        if self.dist_head is not None:
            # Query block only: the trunk's grid is (L+T) wide, but the distogram target is the
            # query's own CA-CA distances, so the reference rows/columns must not be fed to a loss
            # that will index them as query residues.
            pair_logits = self.dist_head(z[:, :L, :L])
            # A distance matrix is symmetric, matching how the contact logits above are handled.
            pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))
            out["pair_logits"] = pair_logits * q_pair[..., None].to(pair_logits.dtype)

        return out


def _pair_feature_indices(mode: str):
    """Which of the reference's element-pair channels this mode consumes.

    Uses the shared mapping so both architectures are fed the SAME channels for
    the same `pair_ref_features` setting -- a second definition here would let the arms drift
    apart silently and make the comparison meaningless.
    """
    names = list(PAIR_FEATURE_NAMES)
    return [names.index(n) for n in PAIR_FEATURE_MODES[mode]]
