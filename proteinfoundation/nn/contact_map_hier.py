# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""ContactMapHierSiT: two-level hierarchical contact map diffusion backbone.

Successor to ContactMapSiT (contact_map_dit.py), which pooled the L x L map into
l x l blocks exactly once, ran all-by-all attention over the block tokens, and
decoded each block independently with no full-resolution skip.

    Phase 1 - Protein encoder (unchanged, shared with ContactMapSiT)
      sequence + pair representations from FeatureFactory, N_enc pair-biased MHSA
      blocks interleaved with PairReprUpdate (no triangle multiplication).

    Level 0 (cells, L x L)
      cell = Linear([contact_map_t, contact_map_sc, pair_rep]) -> d_local
      block_featurizer="local_attn": n_local_layers of within-block self-attention
      with 2D RoPE over the (row, col) position inside the block.
      block_featurizer="conv": no within-block attention, so the level-1 pooling
      projection below acts directly on the embedded cells and is exactly a
      strided Conv2d(kernel=stride=block_size) over them - the v1 patchify, with
      the cell embedding as a rank-d_local bottleneck on its input channels.

    Level 1 (blocks, P1 x P1, P1 = L / block_size)
      block token = Linear(flatten(block_size^2 cells)) -> d_block, plus a pooled
      pair_rep context term.

    Level 2 (super-blocks, P2 x P2, P2 = P1 / super_factor)
      super token = Linear(flatten(super_factor^2 blocks)) -> d_super, plus a
      pooled pair_rep context term.
      n_global_layers of all-by-all attention with a Swin-style joint 2D relative
      position bias (learned per-head table over clipped (drow, dcol)).

    Decoder
      super -> block: Linear(d_super -> super_factor^2 * d_block), reshaped onto
      the block grid, CONCATENATED with the pre-global block tokens (skip),
      projected back to d_block, then n_block_layers of all-by-all block attention
      with its own relative position bias.
      block -> cell: Linear(d_block -> block_size^2 * d_local), reshaped onto the
      cell grid, CONCATENATED with the cell features and the full-resolution
      pair_rep (skip), projected to a single logit per (i, j).

Memory note: the relative position bias is materialised as [heads, N, N] per
attention layer (N = P1^2 or P2^2), so block-level attention cost grows as
(L / block_size)^4. This is the dominant term at small block sizes and large L.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinfoundation.nn.alphafold3_pytorch_utils.modules import (
    AdaptiveLayerNorm,
    AdaptiveLayerNormOutputScale,
    Transition,
)
from proteinfoundation.datasets.sse_topology import (
    CIRCUIT_PAIR_FEATURES,
    N_PAIR_FEATURES,
    PAIR_FEATURE_NAMES,
    PROXIMITY_PAIR_FEATURES,
    pair_feature_indices,
)
from proteinfoundation.datasets.sse_topology import MASK_TOKEN as TOPOLOGY_MASK_TOKEN
from proteinfoundation.datasets.sse_topology import PAD_TOKEN as TOPOLOGY_PAD_TOKEN
from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.protein_transformer import (
    MultiheadAttnAndTransition,
    PairReprBuilder,
    PairReprUpdate,
    TransitionADALN,
)

# Which channels of the 2D SSE reference the pair track sees. "contact" is the original
# behaviour (which elements touch, nothing more); the other three add the descriptions of HOW they
# touch that sse_topology computes, and exist so the two can be compared empirically.
PAIR_FEATURE_MODES: Dict[str, Tuple[str, ...]] = {
    "contact": ("contact_max",),
    "circuit": ("contact_max",) + CIRCUIT_PAIR_FEATURES,
    "proximity": ("contact_max",) + PROXIMITY_PAIR_FEATURES,
    "both": PAIR_FEATURE_NAMES,
}

# Cache of relative-position index grids, keyed by (P, max_offset, device).
# Shared across layers and levels since the indices depend only on the geometry.
_REL_INDEX_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}

# Cache of 2D RoPE (cos, sin) tables, keyed by (block_size, head_dim, device, dtype).
_ROPE_CACHE: Dict[Tuple[int, int, str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def flat_rel_index(P: int, max_offset: int, device: torch.device) -> torch.Tensor:
    """Flattened index into a (2*max_offset+1)^2 relative position table.

    Args:
        P: Grid size (tokens are the P**2 cells of a P x P grid, row-major).
        max_offset: Offsets are clipped to +/- this value, Swin-style.
        device: Target device.

    Returns:
        Long tensor of shape [P**2, P**2] indexing a flattened (drow, dcol) table.
    """
    key = (P, max_offset, str(device))
    if key not in _REL_INDEX_CACHE:
        rows = torch.arange(P, device=device).repeat_interleave(P)  # [P**2]
        cols = torch.arange(P, device=device).repeat(P)  # [P**2]
        d_row = (rows[:, None] - rows[None, :]).clamp(-max_offset, max_offset) + max_offset
        d_col = (cols[:, None] - cols[None, :]).clamp(-max_offset, max_offset) + max_offset
        _REL_INDEX_CACHE[key] = d_row * (2 * max_offset + 1) + d_col
    return _REL_INDEX_CACHE[key]


def rope_2d_tables(
    block_size: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for 2D RoPE over the cells of one block.

    The head dimension is split in half: the first half is rotated by the row
    position, the second half by the column position.

    Args:
        block_size: Block side length l; tokens are the l**2 cells, row-major.
        head_dim: Per-head dimension, must be divisible by 4.
        device: Target device.
        dtype: Target dtype.

    Returns:
        (cos, sin), each of shape [l**2, head_dim].
    """
    key = (block_size, head_dim, str(device), dtype)
    if key not in _ROPE_CACHE:
        if head_dim % 4 != 0:
            raise ValueError(f"2D RoPE needs head_dim divisible by 4, got {head_dim}")
        n_freq = head_dim // 4
        inv_freq = base ** (-torch.arange(n_freq, device=device, dtype=torch.float32) / n_freq)
        pos = torch.arange(block_size, device=device, dtype=torch.float32)
        rows = pos.repeat_interleave(block_size)  # [l**2]
        cols = pos.repeat(block_size)  # [l**2]
        ang = torch.cat([rows[:, None] * inv_freq, cols[:, None] * inv_freq], dim=-1)  # [l**2, hd/2]
        cos = ang.cos().repeat_interleave(2, dim=-1).to(dtype)
        sin = ang.sin().repeat_interleave(2, dim=-1).to(dtype)
        _ROPE_CACHE[key] = (cos, sin)
    return _ROPE_CACHE[key]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to [*, n, head_dim] given [n, head_dim] cos/sin tables."""
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
    return x * cos + rotated * sin


class RelPosBias2D(nn.Module):
    """Swin-style joint 2D relative position bias.

    A learned per-head scalar indexed by the (drow, dcol) offset between two grid
    positions, clipped to +/- max_offset. Unlike a factorised (drow-table plus
    dcol-table) bias this can express joint offset structure, which is the point
    of reasoning over 2D block positions rather than 1D sequence separation.
    """

    def __init__(self, nheads: int, max_offset: int):
        super().__init__()
        self.max_offset = int(max_offset)
        n = (2 * self.max_offset + 1) ** 2
        self.table = nn.Parameter(torch.zeros(nheads, n))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, P: int) -> torch.Tensor:
        """Returns the bias for a P x P grid, shape [nheads, P**2, P**2]."""
        idx = flat_rel_index(P, self.max_offset, self.table.device)
        return self.table[:, idx]


class BiasedMHSA(nn.Module):
    """Multi-head self-attention accepting an additive bias and/or 2D RoPE.

    torch.nn.MultiheadAttention (used by the v1 SiT blocks) has no route for a
    per-head positional bias, hence this small SDPA-based implementation.
    """

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        if dim % nheads != 0:
            raise ValueError(f"dim {dim} not divisible by nheads {nheads}")
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
            mask: [B, N] bool, True for valid tokens.
            bias: Optional [nheads, N, N] additive attention bias.
            rope: Optional (cos, sin) tables of shape [N, head_dim].

        Returns:
            [B, N, dim]
        """
        B, N, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, self.nheads, self.head_dim).transpose(1, 2) for t in (q, k, v))

        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # A fully-masked query row (a block lying entirely in the padding region)
        # softmaxes to NaN on some SDPA backends, so invalid queries are allowed to
        # attend freely and their output is zeroed on the way out instead.
        query_invalid = ~mask[:, None, :, None]  # [B, 1, N, 1]
        if bias is None:
            attn_mask = mask[:, None, None, :] | query_invalid  # [B, 1, N, N] bool
        else:
            neg = torch.finfo(q.dtype).min
            m = torch.where(mask[:, None, None, :], 0.0, neg).to(q.dtype)  # [B, 1, 1, N]
            m = m.masked_fill(query_invalid, 0.0)  # [B, 1, N, N]
            # Relative-position tables are shared across the batch ([nheads, N, N]); a bias
            # derived from the pair representation is per-sample ([B, nheads, N, N]).
            b = bias if bias.dim() == 4 else bias[None]
            attn_mask = m + b.to(q.dtype)  # [B, nheads, N, N]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out) * mask[..., None]


class BiasedSiTBlock(nn.Module):
    """adaLN-Zero attention + MLP block with an optional relative position bias.

    Same structure as contact_map_dit.SiTBlock (adaLN in, adaLN-Zero out scaling,
    residual around each sublayer) but routed through BiasedMHSA so that block-
    and super-block-level attention can carry a relative position bias, and so
    within-block attention can carry 2D RoPE.
    """

    def __init__(self, dim: int, nheads: int, dim_cond: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.attn = BiasedMHSA(dim, nheads)
        self.scale_output = AdaptiveLayerNormOutputScale(dim=dim, dim_cond=dim_cond)
        self.mlp = TransitionADALN(dim=dim, dim_cond=dim_cond, expansion_factor=mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
            cond: [B, 1, dim_cond] or [B, N, dim_cond]; broadcast over tokens.
            mask: [B, N] bool
            bias: Optional [nheads, N, N]
            rope: Optional (cos, sin), each [N, head_dim]

        Returns:
            [B, N, dim]
        """
        h = self.adaln(x, cond, mask)
        h = self.attn(h, mask, bias=bias, rope=rope)
        x = x + self.scale_output(h, cond, mask)
        x = x + self.mlp(x, cond, mask)
        return x * mask[..., None]


def rope_1d_from_positions(
    pos: torch.Tensor, head_dim: int, base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin tables at arbitrary (possibly fractional) positions.

    Positions are floats rather than indices because topology-element positions are rescaled onto
    the query grid, which rarely lands on integers.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = base ** (-torch.arange(0, half, device=pos.device, dtype=torch.float32) / half)
    ang = pos.float().unsqueeze(-1) * inv_freq  # [..., half]
    # apply_rope pairs channels as (0,1), (2,3), ... so the angle table must be interleaved to
    # match; a half-split layout would not be a rotation and would break relative-offset behaviour.
    ang = ang.repeat_interleave(2, dim=-1)
    return ang.cos(), ang.sin()


class CrossAttention(nn.Module):
    """Query tokens attend to a topology reference of unrelated length, with relative positions.

    Query index and topology-element index are different coordinate systems, so a bare cross
    attention would have no notion of "this element sits near this residue". Each element carries a
    residue-space midpoint rescaled onto the query grid, and RoPE is applied to queries at their own
    index and to keys at those rescaled positions -- the mixed-resolution RoPE alignment trick, which
    makes the attention logit depend on the query-key offset in query-residue units.
    """

    def __init__(self, dim_q: int, dim_kv: int, nheads: int, use_rope: bool = True):
        super().__init__()
        if dim_q % nheads != 0:
            raise ValueError(f"dim_q={dim_q} not divisible by nheads={nheads}")
        self.nheads = nheads
        self.head_dim = dim_q // nheads
        self.use_rope = use_rope
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.to_q = nn.Linear(dim_q, dim_q, bias=False)
        self.to_k = nn.Linear(dim_kv, dim_q, bias=False)
        self.to_v = nn.Linear(dim_kv, dim_q, bias=False)
        self.proj = nn.Linear(dim_q, dim_q, bias=False)
        nn.init.zeros_(self.proj.weight)  # zero-init gate, matching the adaLN-Zero blocks

    def forward(
        self,
        x: torch.Tensor,               # [B, N, dim_q]
        kv: torch.Tensor,              # [B, T, dim_kv]
        kv_mask: torch.Tensor,         # [B, T] bool
        q_pos: Optional[torch.Tensor] = None,   # [N] query grid positions
        kv_pos: Optional[torch.Tensor] = None,  # [B, T] key positions on the query grid
    ) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.to_q(self.norm_q(x))
        h = self.norm_kv(kv)
        k, v = self.to_k(h), self.to_v(h)
        q = q.reshape(B, N, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.nheads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.nheads, self.head_dim).transpose(1, 2)

        if self.use_rope and q_pos is not None and kv_pos is not None:
            cq, sq = rope_1d_from_positions(q_pos, self.head_dim)
            q = apply_rope(q, cq[None, None].to(q.dtype), sq[None, None].to(q.dtype))
            ck, sk = rope_1d_from_positions(kv_pos, self.head_dim)
            k = apply_rope(k, ck[:, None].to(k.dtype), sk[:, None].to(k.dtype))

        # A reference that is entirely padding (no template at all) would softmax to NaN, so
        # such rows are allowed to attend freely and zeroed on the way out instead.
        empty = ~kv_mask.any(dim=-1)  # [B]
        attn_mask = kv_mask[:, None, None, :] | empty[:, None, None, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.proj(out) * (~empty)[:, None, None].to(out.dtype)


class TopologyPairReference(nn.Module):
    """Embeds the T x T SSE-by-SSE reference into cross-attention keys/values.

    Only helices and strands index this reference: loops are flexible linkers whose contacts carry
    no topological constraint, and including them would square a four-times-longer axis.

    Each element pair contributes the two element embeddings plus a selectable block of pair
    descriptors -- see PAIR_FEATURE_MODES for what each choice supplies.
    """

    def __init__(self, dim_token: int, dim_out: int, n_pair_features: int = 1):
        super().__init__()
        self.n_pair_features = n_pair_features
        self.proj = nn.Linear(2 * dim_token + n_pair_features, dim_out)

    def forward(self, tok_embed: torch.Tensor, pair_feat: torch.Tensor) -> torch.Tensor:
        """[B, T, d], [B, T, T, F] -> [B, T*T, dim_out]"""
        B, T, _ = tok_embed.shape
        a = tok_embed[:, :, None, :].expand(B, T, T, tok_embed.shape[-1])
        b = tok_embed[:, None, :, :].expand(B, T, T, tok_embed.shape[-1])
        feat = torch.cat([a, b, pair_feat], dim=-1)
        return self.proj(feat).reshape(B, T * T, -1)


class CrossAttention2D(nn.Module):
    """2D grid queries attend to the flattened SSE-pair reference with a joint 2D relative bias.

    A pair query at grid position (u, v) relates to reference pair (s, t) through the joint offset
    (u - p_s, v - p_t), where p is the element position rescaled onto the query grid. That is the
    same Swin-style joint 2D bias used by the pooled self-attention levels, just evaluated across
    two different grids instead of one.
    """

    def __init__(self, dim_q: int, dim_kv: int, nheads: int, max_offset: int):
        super().__init__()
        if dim_q % nheads != 0:
            raise ValueError(f"dim_q={dim_q} not divisible by nheads={nheads}")
        self.nheads = nheads
        self.head_dim = dim_q // nheads
        self.max_offset = max_offset
        self.span = 2 * max_offset + 1
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.to_q = nn.Linear(dim_q, dim_q, bias=False)
        self.to_k = nn.Linear(dim_kv, dim_q, bias=False)
        self.to_v = nn.Linear(dim_kv, dim_q, bias=False)
        self.proj = nn.Linear(dim_q, dim_q, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.bias_table = nn.Parameter(torch.zeros(self.span * self.span, nheads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def _bias(self, P: int, kv_pos: torch.Tensor) -> torch.Tensor:
        """[B, T] element positions on the query grid -> [B, nheads, P*P, T*T]."""
        B, T = kv_pos.shape
        grid = torch.arange(P, device=kv_pos.device, dtype=torch.float32)
        d = grid[:, None] - kv_pos[:, None, :]  # [B, P, T]
        d = d.round().long().clamp(-self.max_offset, self.max_offset) + self.max_offset
        # row offset varies with the query row and the reference's first element; column with the
        # second -- the outer sum over the two axes is the joint 2D offset index.
        idx = d[:, :, None, :, None] * self.span + d[:, None, :, None, :]  # [B, P, P, T, T]
        idx = idx.reshape(B, P * P, T * T)
        bias = self.bias_table[idx]  # [B, P*P, T*T, nheads]
        return bias.permute(0, 3, 1, 2)

    def forward(
        self,
        x: torch.Tensor,        # [B, P*P, dim_q]
        kv: torch.Tensor,       # [B, T*T, dim_kv]
        kv_mask: torch.Tensor,  # [B, T*T] bool
        P: int,
        kv_pos: torch.Tensor,   # [B, T] positions on the query grid
    ) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.to_q(self.norm_q(x))
        h = self.norm_kv(kv)
        k, v = self.to_k(h), self.to_v(h)
        q = q.reshape(B, N, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.nheads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.nheads, self.head_dim).transpose(1, 2)

        empty = ~kv_mask.any(dim=-1)
        neg = torch.finfo(q.dtype).min
        m = torch.where(kv_mask, 0.0, neg).to(q.dtype)[:, None, None, :]
        m = m.masked_fill(empty[:, None, None, None], 0.0)
        attn_mask = m + self._bias(P, kv_pos).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.proj(out) * (~empty)[:, None, None].to(out.dtype)


class OuterProductMean(nn.Module):
    """AlphaFold-style single -> pair projection: outer product of two low-rank projections."""

    def __init__(self, dim_single: int, dim_pair: int, dim_hidden: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim_single)
        self.to_a = nn.Linear(dim_single, dim_hidden, bias=False)
        self.to_b = nn.Linear(dim_single, dim_hidden, bias=False)
        self.proj = nn.Linear(dim_hidden * dim_hidden, dim_pair)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """[B, N, d_single] -> [B, N, N, dim_pair]"""
        h = self.norm(s)
        a, b = self.to_a(h), self.to_b(h)
        outer = torch.einsum("bic,bjd->bijcd", a, b)
        return self.proj(outer.flatten(start_dim=-2))


class PairToBias(nn.Module):
    """2D pair grid -> additive per-head attention bias for the 1D track at the same level."""

    def __init__(self, dim_pair: int, nheads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_pair)
        self.proj = nn.Linear(dim_pair, nheads, bias=False)

    def forward(self, pair_grid: torch.Tensor) -> torch.Tensor:
        """[B, N, N, dim_pair] -> [B, nheads, N, N]"""
        return self.proj(self.norm(pair_grid)).permute(0, 3, 1, 2)


class ContactMapHierSiT(nn.Module):
    """Two-level hierarchical contact map backbone (see module docstring).

    Implements the same model.nn interface as ProteinTransformerAF3 and
    ContactMapSiT, so it is selected purely by `nn_class` in the nn config.

    Required batch keys:
        contact_map_t   [B, L, L]       noisy contact map
        mask            [B, L] bool
        t               [B]             flow-matching timestep in [0, 1]
        residue_type    [B, L] long     amino-acid indices (0-20)
        contact_map_sc  [B, L, L]       self-conditioning contact map (optional)
        cath_code_indices               fold conditioning (optional)

    Output dict keys:
        contact_map_logits  [B, L, L]
        contact_map_pred    [B, L, L]   sigmoid (or tanh) of the logits
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Phase 1 hyper-parameters (shared with ContactMapSiT)
        self.token_dim = int(kwargs["token_dim"])
        self.pair_repr_dim = int(kwargs["pair_repr_dim"])
        self.dim_cond = int(kwargs["dim_cond"])
        self.n_enc_layers = int(kwargs["n_enc_layers"])
        self.enc_pair_update_every_n = int(kwargs["enc_pair_update_every_n"])
        nheads_enc = int(kwargs["nheads_enc"])
        use_qkln = bool(kwargs.get("use_qkln", True))

        # Hierarchy geometry
        self.block_size = int(kwargs["block_size"])
        self.super_factor = int(kwargs["super_factor"])
        self.pad_period = self.block_size * self.super_factor
        self.max_seq_len = int(kwargs["max_seq_len"])

        # Level 0 (cells)
        self.block_featurizer = str(kwargs["block_featurizer"])
        if self.block_featurizer not in ("local_attn", "conv"):
            raise ValueError(
                f"block_featurizer must be 'local_attn' or 'conv', got {self.block_featurizer!r}"
            )
        self.d_local = int(kwargs["d_local"])
        self.n_local_layers = int(kwargs["n_local_layers"])
        nheads_local = int(kwargs["nheads_local"])

        # Levels 1/2 and the attention stacks
        self.d_block = int(kwargs["d_block"])
        self.d_super = int(kwargs["d_super"])
        nheads_attn = int(kwargs["nheads_dit"])
        self.n_global_layers = int(kwargs["n_global_layers"])
        self.n_block_layers = int(kwargs["n_block_layers"])
        self.d_cond_dit = int(kwargs.get("d_cond_dit", self.dim_cond))
        mlp_ratio = float(kwargs.get("mlp_ratio", 4.0))

        # Optional 1D single track cross-attending to a reference topology. Absent from a
        # config -> False -> the model behaves exactly as the 2D-only architecture.
        self.topology_cond = bool(kwargs.get("topology_cond", False))
        self.topology_vocab_size = int(kwargs.get("topology_vocab_size", 0))
        self.max_topology_len = int(kwargs.get("max_topology_len", 0))
        self.d_topo = int(kwargs.get("d_topo", 0))
        self.d_single = int(kwargs.get("d_single", 0))
        self.n_topo_layers = int(kwargs.get("n_topo_layers", 0))
        nheads_topo = int(kwargs.get("nheads_topo", 0))
        self.topology_reinject = bool(kwargs.get("topology_reinject", True))
        self.pair_ref_features = str(kwargs.get("pair_ref_features", "contact"))
        if self.pair_ref_features not in PAIR_FEATURE_MODES:
            raise ValueError(
                f"pair_ref_features must be one of {sorted(PAIR_FEATURE_MODES)}, "
                f"got {self.pair_ref_features!r}"
            )
        self.pair_feat_idx = pair_feature_indices(PAIR_FEATURE_MODES[self.pair_ref_features])
        if self.topology_cond:
            missing = [
                k
                for k, v in (
                    ("topology_vocab_size", self.topology_vocab_size),
                    ("max_topology_len", self.max_topology_len),
                    ("d_topo", self.d_topo),
                    ("d_single", self.d_single),
                    ("n_topo_layers", self.n_topo_layers),
                    ("nheads_topo", nheads_topo),
                )
                if v <= 0
            ]
            if missing:
                raise ValueError(f"topology_cond=True requires {missing} to be set > 0")

        # Attributes read by Proteina
        self.contact_map_mode = True
        self.predict_coords = None
        self.predict_dssp = False
        self.non_contact_value = int(kwargs.get("non_contact_value", 0))
        if self.non_contact_value not in (0, -1):
            raise ValueError(f"non_contact_value must be 0 or -1, got {self.non_contact_value}")

        _ff_skip = {"feature_embedding_mode", "individual_feat_ln"}
        _feat_kwargs = {k: v for k, v in kwargs.items() if k not in _ff_skip}

        # ── Phase 1: protein encoder ──────────────────────────────────────────
        feats_init_seq = list(kwargs.get("feats_init_seq", ["res_seq_pdb_idx"]))
        self.init_repr_factory = FeatureFactory(
            feats=feats_init_seq,
            dim_feats_out=self.token_dim,
            use_ln_out=False,
            mode="seq",
            use_residue_type_emb=bool(kwargs.get("residue_type_emb_init_seq", False)),
            use_ext_lig_emb=bool(kwargs.get("ext_lig_emb_init_seq", False)),
            feature_embedding_mode=kwargs.get("feature_embedding_mode", "concat"),
            individual_feat_ln=bool(kwargs.get("individual_feat_ln", True)),
            **_feat_kwargs,
        )

        feats_cond_seq = list(kwargs.get("feats_cond_seq", ["time_emb", "fold_emb"]))
        self.cond_factory = FeatureFactory(
            feats=feats_cond_seq,
            dim_feats_out=self.dim_cond,
            use_ln_out=False,
            mode="seq",
            use_residue_type_emb=bool(kwargs.get("residue_type_emb_cond_seq", False)),
            use_ext_lig_emb=bool(kwargs.get("ext_lig_emb_cond_seq", False)),
            feature_embedding_mode=kwargs.get("feature_embedding_mode", "concat"),
            individual_feat_ln=bool(kwargs.get("individual_feat_ln", True)),
            **_feat_kwargs,
        )
        self.transition_c_1 = Transition(self.dim_cond, expansion_factor=2)
        self.transition_c_2 = Transition(self.dim_cond, expansion_factor=2)

        feats_pair_repr = list(kwargs.get("feats_pair_repr", ["rel_seq_sep", "contact_map_sc"]))
        feats_pair_cond = list(kwargs.get("feats_pair_cond", ["time_emb"]))
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=feats_pair_repr,
            feats_cond=feats_pair_cond,
            dim_feats_out=self.pair_repr_dim,
            dim_cond_pair=self.dim_cond,
            **_feat_kwargs,
        )

        contact_map_input_dim = int(kwargs.get("contact_map_input_dim", 1))
        self.linear_contact_embed = nn.Linear(contact_map_input_dim, self.pair_repr_dim, bias=False)

        self.encoder_blocks = nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=nheads_enc,
                    dim_cond=self.dim_cond,
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=use_qkln,
                )
                for _ in range(self.n_enc_layers)
            ]
        )
        n_pair_updates = max(1, self.n_enc_layers // self.enc_pair_update_every_n)
        self.pair_updates = nn.ModuleList(
            [
                PairReprUpdate(
                    token_dim=self.token_dim,
                    pair_dim=self.pair_repr_dim,
                    expansion_factor_transition=2,
                    use_tri_mult=False,
                )
                for _ in range(n_pair_updates)
            ]
        )

        self.cond_to_dit = nn.Linear(self.dim_cond, self.d_cond_dit, bias=True)

        # ── Level 0: cell embedding + optional within-block attention ─────────
        # 2 channels (noisy map, self-conditioning map) plus the full-resolution
        # pair representation, so every cell sees the encoder's output directly.
        self.cell_embed = nn.Linear(2 + self.pair_repr_dim, self.d_local, bias=False)
        self.local_blocks = nn.ModuleList(
            [
                BiasedSiTBlock(self.d_local, nheads_local, self.d_cond_dit, mlp_ratio)
                for _ in range(self.n_local_layers if self.block_featurizer == "local_attn" else 0)
            ]
        )
        self.local_head_dim = self.d_local // nheads_local

        # ── Level 1: cells -> blocks ──────────────────────────────────────────
        self.pool_cell_to_block = nn.Linear(self.block_size**2 * self.d_local, self.d_block)
        self.pair_ctx_block = nn.Linear(self.pair_repr_dim, self.d_block, bias=False)

        # ── Level 2: blocks -> super-blocks ───────────────────────────────────
        self.pool_block_to_super = nn.Linear(self.super_factor**2 * self.d_block, self.d_super)
        self.pair_ctx_super = nn.Linear(self.pair_repr_dim, self.d_super, bias=False)

        # Relative position bias tables. The clip radius covers the full grid at
        # max_seq_len, so nothing inside the training length range is clipped.
        # Both radii come from the PADDED length: max_seq_len is rounded up to a
        # multiple of pad_period, which can add one more block per side (e.g.
        # max_seq_len=512, block_size=12, super_factor=4 pads to 528 -> 44 blocks,
        # so offsets reach 43, not the 42 that ceil(512/12)-1 would allow for).
        n_super_max = math.ceil(self.max_seq_len / self.pad_period)
        max_off_super = max(1, n_super_max - 1)
        max_off_block = max(1, n_super_max * self.super_factor - 1)
        self.rel_pos_super = RelPosBias2D(nheads_attn, max_off_super)
        self.rel_pos_block = RelPosBias2D(nheads_attn, max_off_block)

        self.global_blocks = nn.ModuleList(
            [
                BiasedSiTBlock(self.d_super, nheads_attn, self.d_cond_dit, mlp_ratio)
                for _ in range(self.n_global_layers)
            ]
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.unpool_super_to_block = nn.Linear(self.d_super, self.super_factor**2 * self.d_block)
        self.merge_block_skip = nn.Linear(2 * self.d_block, self.d_block)
        self.block_blocks = nn.ModuleList(
            [
                BiasedSiTBlock(self.d_block, nheads_attn, self.d_cond_dit, mlp_ratio)
                for _ in range(self.n_block_layers)
            ]
        )
        self.unpool_block_to_cell = nn.Linear(self.d_block, self.block_size**2 * self.d_local)
        self.output_norm = nn.LayerNorm(2 * self.d_local + self.pair_repr_dim)
        self.output_head = nn.Linear(2 * self.d_local + self.pair_repr_dim, 1, bias=True)

        # ── Optional 1D single track conditioned on a reference topology ──────
        # Absent -> the model is exactly the 2D-only architecture above.
        if self.topology_cond:
            self.topology_embed = nn.Embedding(
                self.topology_vocab_size, self.d_topo, padding_idx=TOPOLOGY_PAD_TOKEN
            )
            self.topology_pos = nn.Embedding(self.max_topology_len, self.d_topo)
            self.single_in = nn.Linear(self.token_dim, self.d_single)
            self.single_self_blocks = nn.ModuleList(
                [
                    BiasedSiTBlock(self.d_single, nheads_topo, self.d_cond_dit, mlp_ratio)
                    for _ in range(self.n_topo_layers)
                ]
            )
            self.single_cross_blocks = nn.ModuleList(
                [
                    CrossAttention(self.d_single, self.d_topo, nheads_topo)
                    for _ in range(self.n_topo_layers)
                ]
            )
            # 1D mirrors the 2D pooling exactly: flatten `factor` consecutive positions and
            # project, so the single track has the same token count as the pair grid side
            # length at every level.
            self.pool_single_to_block = nn.Linear(self.block_size * self.d_single, self.d_block)
            self.pool_single_to_super = nn.Linear(self.super_factor * self.d_block, self.d_super)
            self.unpool_single_super_to_block = nn.Linear(
                self.d_super, self.super_factor * self.d_block
            )
            self.merge_single_block_skip = nn.Linear(2 * self.d_block, self.d_block)
            self.unpool_single_block_to_cell = nn.Linear(
                self.d_block, self.block_size * self.d_single
            )
            self.merge_single_cell_skip = nn.Linear(2 * self.d_single, self.d_single)

            # Per-level AF3-style coupling, one set per 2D layer at that level.
            self.single_super_blocks = nn.ModuleList(
                [
                    BiasedSiTBlock(self.d_super, nheads_attn, self.d_cond_dit, mlp_ratio)
                    for _ in range(self.n_global_layers)
                ]
            )
            self.opm_super = nn.ModuleList(
                [OuterProductMean(self.d_super, self.d_super) for _ in range(self.n_global_layers)]
            )
            self.pair_to_bias_super = nn.ModuleList(
                [PairToBias(self.d_super, nheads_attn) for _ in range(self.n_global_layers)]
            )
            self.single_block_blocks = nn.ModuleList(
                [
                    BiasedSiTBlock(self.d_block, nheads_attn, self.d_cond_dit, mlp_ratio)
                    for _ in range(self.n_block_layers)
                ]
            )
            self.opm_block = nn.ModuleList(
                [OuterProductMean(self.d_block, self.d_block) for _ in range(self.n_block_layers)]
            )
            self.pair_to_bias_block = nn.ModuleList(
                [PairToBias(self.d_block, nheads_attn) for _ in range(self.n_block_layers)]
            )
            # 2D SSE-by-SSE contact reference, cross-attended by the pair track at both pooled
            # levels. Offsets are clipped at the padded block-grid width, which is the widest
            # query grid that ever attends to it.
            self.topology_pair_ref = TopologyPairReference(
                self.d_topo, self.d_topo, len(self.pair_feat_idx)
            )
            self.cross2d_super = CrossAttention2D(
                self.d_super, self.d_topo, nheads_attn, max_off_super
            )
            self.cross2d_block = CrossAttention2D(
                self.d_block, self.d_topo, nheads_attn, max_off_block
            )
            if self.topology_reinject:
                self.reinject_super = CrossAttention(self.d_super, self.d_topo, nheads_attn)
                self.reinject_block = CrossAttention(self.d_block, self.d_topo, nheads_attn)
            # The single track reaches the output head alongside the 2D cell features, so a
            # residue-level signal can influence its own row/column of the map.
            self.single_to_cell = nn.Linear(self.d_single, self.d_local, bias=False)

    # ── Grid helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _grid_to_tokens(x: torch.Tensor, factor: int) -> torch.Tensor:
        """[B, P*f, P*f, D] -> [B, P**2, f**2 * D], grouping each f x f tile."""
        B, N, _, D = x.shape
        P = N // factor
        x = x.reshape(B, P, factor, P, factor, D).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, P * P, factor * factor * D)

    @staticmethod
    def _tokens_to_grid(x: torch.Tensor, P: int, factor: int, D: int) -> torch.Tensor:
        """[B, P**2, f**2 * D] -> [B, P*f, P*f, D], inverse of _grid_to_tokens."""
        B = x.shape[0]
        x = x.reshape(B, P, P, factor, factor, D).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, P * factor, P * factor, D)

    # ── Forward ───────────────────────────────────────────────────────────────

    def _ensure_topology_keys(self, batch: Dict, B: int, device) -> Dict:
        """Supply an unconditioned topology reference when the batch carries none.

        Sampling generates from the length dataset, so those batches never pass through
        TopologyReferenceTransform and have no topology_* keys at all. The substitute is exactly
        what the transform emits for a dropped sample -- a single MASK element -- which is a
        regime the model trains in (drop_prob), not an improvised input. It also mirrors how the
        trainer handles CATH on the same path, where it builds all-null fold indices rather than
        conditioning on a fold it does not have.
        """
        if batch.get("topology_tokens") is not None:
            return batch
        batch = dict(batch)
        mask_tok = torch.full((B, 1), TOPOLOGY_MASK_TOKEN, dtype=torch.long, device=device)
        zeros = torch.zeros(B, 1, dtype=torch.float32, device=device)
        batch["topology_tokens"] = mask_tok
        batch["topology_pos"] = zeros
        batch["topology_he_tokens"] = mask_tok.clone()
        batch["topology_he_pos"] = zeros.clone()
        batch["topology_he_contact"] = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
        batch["topology_he_feat"] = torch.zeros(
            B, 1, 1, N_PAIR_FEATURES, dtype=torch.float32, device=device
        )
        return batch

    def forward(self, batch: Dict, force_compile: bool = False) -> Dict:
        """Run the hierarchy over a batch.

        Args:
            batch: dataloader batch (see the module docstring for required keys).
            force_compile: accepted and ignored. model_trainer_base passes it to every nn on the
                sampling path (`eval_nn(batch, force_compile=...)`); only ProteinTransformerAF3
                uses it to pick a compile path, but a forward that refuses the kwarg raises
                TypeError the first time validation sampling runs.
        """
        contact_map_t = batch["contact_map_t"]  # [B, L, L]
        if contact_map_t.dim() != 3:
            # Multi-channel maps (e.g. the discrete-diffusion configs) would silently
            # mis-broadcast through the cell embedding below.
            raise ValueError(
                f"ContactMapHierSiT expects contact_map_t [B, L, L], got {tuple(contact_map_t.shape)}"
            )
        mask = batch["mask"]  # [B, L] bool
        B, L = mask.shape
        device, dtype = contact_map_t.device, contact_map_t.dtype

        contact_map_sc = batch.get("contact_map_sc", torch.zeros_like(contact_map_t))
        pair_mask = mask[:, :, None] & mask[:, None, :]  # [B, L, L]

        # Feature factories read x_t purely for its (B, N) shape in contact map mode.
        if "x_t" not in batch:
            batch = dict(batch)
            batch["x_t"] = torch.zeros(B, L, 3, device=device, dtype=dtype)

        # ── Phase 1: protein encoder ──────────────────────────────────────────
        seq = self.init_repr_factory(batch) * mask[..., None]
        c = self.cond_factory(batch)
        c = self.transition_c_1(c, mask)
        c = self.transition_c_2(c, mask)

        pair_rep = self.pair_repr_builder(batch)
        contact_embed = self.linear_contact_embed(contact_map_t.unsqueeze(-1))
        pair_rep = (pair_rep + contact_embed) * pair_mask[..., None].to(dtype)

        pair_update_idx = 0
        for i, enc_block in enumerate(self.encoder_blocks):
            seq = enc_block(seq, pair_rep, c, mask)
            if (i + 1) % self.enc_pair_update_every_n == 0 and pair_update_idx < len(self.pair_updates):
                pair_rep = self.pair_updates[pair_update_idx](seq, pair_rep, mask)
                pair_update_idx += 1
        while pair_update_idx < len(self.pair_updates):
            pair_rep = self.pair_updates[pair_update_idx](seq, pair_rep, mask)
            pair_update_idx += 1

        # Global conditioning vector, kept with a singleton token axis so adaLN
        # broadcasts instead of materialising a per-token copy at cell resolution.
        cond = self.cond_to_dit(c.mean(dim=1))[:, None, :]  # [B, 1, d_cond_dit]

        # ── Padding to a multiple of block_size * super_factor ────────────────
        pad_L = (self.pad_period - L % self.pad_period) % self.pad_period
        L_pad = L + pad_L
        mask_pad = F.pad(mask, (0, pad_L)) if pad_L > 0 else mask  # [B, L_pad]
        if pad_L > 0:
            contact_map_t = F.pad(contact_map_t, (0, pad_L, 0, pad_L))
            contact_map_sc = F.pad(contact_map_sc, (0, pad_L, 0, pad_L))
            pair_rep = F.pad(pair_rep, (0, 0, 0, pad_L, 0, pad_L))
            pair_mask = F.pad(pair_mask, (0, pad_L, 0, pad_L))

        P1 = L_pad // self.block_size
        P2 = P1 // self.super_factor
        cell_mask_f = pair_mask.to(dtype)

        # ── Optional 1D single track, conditioned on a reference topology ─────
        single = topo = topo_mask = ref_kv = ref_mask = None
        topo_pos = he_pos_block = he_pos_super = None
        if self.topology_cond:
            batch = self._ensure_topology_keys(batch, B, device)
            tokens = batch["topology_tokens"]  # [B, T]
            # Validity is `> 0`, not `!= PAD`: the dense collate pads integer tensors with
            # NON_FLOAT_PADDING_VALUE (-1), not with TOPOLOGY_PAD_TOKEN (0), so testing against
            # the pad token alone would treat -1 as a real element and index the embedding at -1.
            # MASK_TOKEN (1) stays valid -- it is the explicit "no topology" element.
            topo_mask = tokens > 0
            tokens = tokens.clamp(min=0)
            T = tokens.shape[1]
            pos_idx = torch.arange(T, device=device).clamp(max=self.max_topology_len - 1)
            topo = self.topology_embed(tokens) + self.topology_pos(pos_idx)[None]
            topo = topo * topo_mask[..., None].to(dtype)
            # Element positions arrive already rescaled onto the query residue grid, so a query
            # residue and a topology element can be compared directly.
            topo_pos = batch["topology_pos"].to(torch.float32)  # [B, T]

            single = self.single_in(seq)
            for self_blk, cross_blk in zip(self.single_self_blocks, self.single_cross_blocks):
                single = self_blk(single, cond, mask, bias=None)
                single = single + cross_blk(
                    single,
                    topo,
                    topo_mask,
                    q_pos=torch.arange(L, device=device, dtype=torch.float32),
                    kv_pos=topo_pos,
                )
            single = single * mask[..., None].to(dtype)
            if pad_L > 0:
                single = F.pad(single, (0, 0, 0, pad_L))

            he_tokens = batch["topology_he_tokens"]  # [B, T_he]
            he_valid = he_tokens > 0
            he_embed = self.topology_embed(he_tokens.clamp(min=0)) * he_valid[..., None].to(dtype)
            # An index built before the pair featurization only supplies the contact channel, so
            # the mode that needs nothing else keeps working against it.
            if "topology_he_feat" in batch:
                pair_feat = batch["topology_he_feat"].to(dtype)[..., self.pair_feat_idx]
            elif self.pair_ref_features == "contact":
                pair_feat = batch["topology_he_contact"].to(dtype)[..., None]
            else:
                raise KeyError(
                    f"pair_ref_features={self.pair_ref_features!r} needs topology_he_feat, which "
                    "TopologyReferenceTransform only emits from a featurized topology index"
                )
            ref_kv = self.topology_pair_ref(he_embed, pair_feat)
            ref_mask = (he_valid[:, :, None] & he_valid[:, None, :]).reshape(B, -1)
            # The same element positions expressed on each pooled grid.
            he_pos = batch["topology_he_pos"].to(torch.float32)  # [B, T_he], residue units
            he_pos_block = he_pos / self.block_size
            he_pos_super = he_pos / self.pad_period

        # ── Level 0: cell features ────────────────────────────────────────────
        cell_in = torch.cat(
            [contact_map_t[..., None], contact_map_sc[..., None], pair_rep], dim=-1
        )  # [B, L_pad, L_pad, 2 + pair_repr_dim]
        cells = self.cell_embed(cell_in) * cell_mask_f[..., None]  # [B, L_pad, L_pad, d_local]

        if self.block_featurizer == "local_attn":
            l = self.block_size
            n_blocks = P1 * P1
            # One attention problem per block: [B * P1**2, l**2, d_local]
            x = self._grid_to_tokens(cells, l).reshape(B * n_blocks, l * l, self.d_local)
            m = self._grid_to_tokens(pair_mask[..., None], l).reshape(B * n_blocks, l * l) > 0
            cond_local = cond.expand(B, n_blocks, self.d_cond_dit).reshape(B * n_blocks, 1, self.d_cond_dit)
            rope = rope_2d_tables(l, self.local_head_dim, device, x.dtype)
            for blk in self.local_blocks:
                x = blk(x, cond_local, m, bias=None, rope=rope)
            cells = self._tokens_to_grid(
                x.reshape(B, n_blocks, l * l * self.d_local), P1, l, self.d_local
            )
            cells = cells * cell_mask_f[..., None]

        # ── Level 1: cells -> blocks ──────────────────────────────────────────
        # Strided Conv2d(kernel=stride=block_size) over the cell embedding,
        # expressed as flatten + project so both featurizer paths share it.
        block_tok = self.pool_cell_to_block(self._grid_to_tokens(cells, self.block_size))
        pair_pool_block = F.avg_pool2d(pair_rep.permute(0, 3, 1, 2), self.block_size)
        block_tok = block_tok + self.pair_ctx_block(
            pair_pool_block.permute(0, 2, 3, 1).reshape(B, P1 * P1, self.pair_repr_dim)
        )
        block_mask = (
            F.avg_pool2d(cell_mask_f[:, None], self.block_size).reshape(B, P1 * P1) > 0
        )  # [B, P1**2]
        block_tok = block_tok * block_mask[..., None]

        # ── Level 2: blocks -> super-blocks ───────────────────────────────────
        block_grid = block_tok.reshape(B, P1, P1, self.d_block)
        super_tok = self.pool_block_to_super(self._grid_to_tokens(block_grid, self.super_factor))
        pair_pool_super = F.avg_pool2d(pair_rep.permute(0, 3, 1, 2), self.pad_period)
        super_tok = super_tok + self.pair_ctx_super(
            pair_pool_super.permute(0, 2, 3, 1).reshape(B, P2 * P2, self.pair_repr_dim)
        )
        super_mask = (
            F.avg_pool2d(block_mask.reshape(B, 1, P1, P1).to(dtype), self.super_factor).reshape(
                B, P2 * P2
            )
            > 0
        )
        super_tok = super_tok * super_mask[..., None]

        # ── Pool the single track alongside, level by level ───────────────────
        if self.topology_cond:
            single_block = self.pool_single_to_block(
                single.reshape(B, P1, self.block_size * self.d_single)
            )
            single_mask_block = mask_pad.reshape(B, P1, self.block_size).any(dim=-1)
            single_block = single_block * single_mask_block[..., None]
            single_super = self.pool_single_to_super(
                single_block.reshape(B, P2, self.super_factor * self.d_block)
            )
            single_mask_super = single_mask_block.reshape(B, P2, self.super_factor).any(dim=-1)
            single_super = single_super * single_mask_super[..., None]

        # ── Global all-by-all attention over super-block tokens ───────────────
        bias_super = self.rel_pos_super(P2)
        if not self.topology_cond:
            for blk in self.global_blocks:
                super_tok = blk(super_tok, cond, super_mask, bias=bias_super)
        else:
            super_grid = super_tok.reshape(B, P2, P2, self.d_super)
            if self.topology_reinject:
                single_super = single_super + self.reinject_super(
                    single_super, topo, topo_mask,
                    q_pos=torch.arange(P2, device=device, dtype=torch.float32),
                    kv_pos=topo_pos / self.pad_period,
                )
            for i, blk in enumerate(self.global_blocks):
                # AF3-style coupling: pair biases the single attention, then the single feeds
                # back into the pair through an outer product mean.
                single_super = self.single_super_blocks[i](
                    single_super, cond, single_mask_super,
                    bias=self.pair_to_bias_super[i](super_grid),
                )
                super_grid = super_grid + self.opm_super[i](single_super)
                super_tok = super_grid.reshape(B, P2 * P2, self.d_super)
                super_tok = super_tok + self.cross2d_super(
                    super_tok, ref_kv, ref_mask, P2, he_pos_super
                )
                super_tok = blk(super_tok, cond, super_mask, bias=bias_super)
                super_grid = super_tok.reshape(B, P2, P2, self.d_super)
            super_tok = super_grid.reshape(B, P2 * P2, self.d_super)

        # ── Decoder: super -> blocks, with skip ───────────────────────────────
        up = self._tokens_to_grid(
            self.unpool_super_to_block(super_tok), P2, self.super_factor, self.d_block
        ).reshape(B, P1 * P1, self.d_block)
        block_tok = self.merge_block_skip(torch.cat([up, block_tok], dim=-1)) * block_mask[..., None]

        bias_block = self.rel_pos_block(P1)
        if not self.topology_cond:
            for blk in self.block_blocks:
                block_tok = blk(block_tok, cond, block_mask, bias=bias_block)
        else:
            # The single track unpools and merges its own skip, mirroring the pair track exactly.
            single_up = self.unpool_single_super_to_block(single_super).reshape(B, P1, self.d_block)
            single_block = self.merge_single_block_skip(
                torch.cat([single_up, single_block], dim=-1)
            ) * single_mask_block[..., None]
            if self.topology_reinject:
                single_block = single_block + self.reinject_block(
                    single_block, topo, topo_mask,
                    q_pos=torch.arange(P1, device=device, dtype=torch.float32),
                    kv_pos=topo_pos / self.block_size,
                )
            block_grid = block_tok.reshape(B, P1, P1, self.d_block)
            for i, blk in enumerate(self.block_blocks):
                single_block = self.single_block_blocks[i](
                    single_block, cond, single_mask_block,
                    bias=self.pair_to_bias_block[i](block_grid),
                )
                block_grid = block_grid + self.opm_block[i](single_block)
                block_tok = block_grid.reshape(B, P1 * P1, self.d_block)
                block_tok = block_tok + self.cross2d_block(
                    block_tok, ref_kv, ref_mask, P1, he_pos_block
                )
                block_tok = blk(block_tok, cond, block_mask, bias=bias_block)
                block_grid = block_tok.reshape(B, P1, P1, self.d_block)
            block_tok = block_grid.reshape(B, P1 * P1, self.d_block)

        # ── Decoder: blocks -> cells, with skip ───────────────────────────────
        cell_up = self._tokens_to_grid(
            self.unpool_block_to_cell(block_tok), P1, self.block_size, self.d_local
        )  # [B, L_pad, L_pad, d_local]

        if self.topology_cond:
            # Unpool the single track back to residue resolution with its own skip, then let it
            # reach the map as an outer sum over rows and columns -- residue i's state informs its
            # entire row and column, which is where a 1D track can help a 2D output.
            single_cell = self.unpool_single_block_to_cell(block_tok.reshape(B, P1, P1, self.d_block).mean(dim=2))
            single_cell = single_cell.reshape(B, L_pad, self.d_single)
            single_cell = self.merge_single_cell_skip(
                torch.cat([single_cell, single], dim=-1)
            ) * mask_pad[..., None].to(dtype)
            s_cell = self.single_to_cell(single_cell)  # [B, L_pad, d_local]
            cell_up = cell_up + s_cell[:, :, None, :] + s_cell[:, None, :, :]

        out = torch.cat([cell_up, cells, pair_rep], dim=-1)
        logits = self.output_head(self.output_norm(out)).squeeze(-1)  # [B, L_pad, L_pad]

        if pad_L > 0:
            logits = logits[:, :L, :L]

        logits = (logits + logits.transpose(-1, -2)) / 2.0
        logits = logits * (mask[:, :, None] & mask[:, None, :]).to(dtype)

        contact_map_pred = (
            torch.sigmoid(logits) if self.non_contact_value == 0 else torch.tanh(logits)
        )
        return {"contact_map_logits": logits, "contact_map_pred": contact_map_pred}
