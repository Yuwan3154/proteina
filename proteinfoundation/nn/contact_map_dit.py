# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
ContactMapSiT: Hybrid Protein Encoder + SiT (DiT-style) architecture for
contact map diffusion. Completely separate from ProteinTransformerAF3 —
no existing files are modified.

Architecture:
  Phase 1 — Protein Encoder (existing modules, use_tri_mult=False)
    - FeatureFactory builds sequence and pair representations from residue
      type, time, and fold conditioning
    - N_enc MultiheadAttnAndTransition blocks (pair-biased MHSA)
    - Interleaved PairReprUpdate blocks (outer-product style, no triangle mult)
  Bridge
    - Mean-pool conditioning vector c → global adaLN conditioning for DiT
    - Avg-pool pair_rep over 5×5 blocks → per-patch local context tokens
  Phase 2 — SiT on contact map patches
    - Conv2d(2, d_patch, k=5, s=5) patchifies [contact_map_t, contact_map_sc]
    - N_dit SiTBlocks: MultiHeadAttentionADALN + TransitionADALN (reused)
    - Linear(d_patch, 25) → reshape → symmetrize → sigmoid
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinfoundation.nn.alphafold3_pytorch_utils.modules import Transition
from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.protein_transformer import (
    MultiHeadAttentionADALN,
    MultiheadAttnAndTransition,
    PairReprBuilder,
    PairReprUpdate,
    TransitionADALN,
)


def sinusoidal_2d_pos_emb(P: int, d: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal 2D positional embedding for a P×P patch grid.

    Uses the first d//2 dims for row position and the remaining d//2 dims for
    column position. Computed at runtime so it generalises to any protein length
    without storing a learnable parameter table.

    Args:
        P: Grid size (number of patches per side).
        d: Embedding dimension (must be even).
        device: Target device.

    Returns:
        Tensor of shape [P², d].
    """
    half_d = d // 2
    # Row and column indices for every patch in the flattened P² sequence
    row_idx = torch.arange(P, device=device).unsqueeze(1).expand(P, P).reshape(-1).float()
    col_idx = torch.arange(P, device=device).unsqueeze(0).expand(P, P).reshape(-1).float()

    # Frequency terms: shape [half_d // 2]
    n_freqs = half_d // 2
    div_term = 10000.0 ** (torch.arange(0, n_freqs, device=device).float() / max(n_freqs, 1))

    row_emb = torch.zeros(P * P, half_d, device=device)
    row_emb[:, 0::2] = torch.sin(row_idx.unsqueeze(1) / div_term.unsqueeze(0))
    row_emb[:, 1::2] = torch.cos(row_idx.unsqueeze(1) / div_term.unsqueeze(0))

    col_emb = torch.zeros(P * P, half_d, device=device)
    col_emb[:, 0::2] = torch.sin(col_idx.unsqueeze(1) / div_term.unsqueeze(0))
    col_emb[:, 1::2] = torch.cos(col_idx.unsqueeze(1) / div_term.unsqueeze(0))

    return torch.cat([row_emb, col_emb], dim=-1)  # [P², d]


class SiTBlock(nn.Module):
    """Single SiT / DiT block: adaLN-Zero MHSA + adaLN-Zero MLP.

    Reuses MultiHeadAttentionADALN and TransitionADALN directly from
    protein_transformer.py.  adaLN-Zero output scaling is already built into
    AdaptiveLayerNormOutputScale (zero-initialised gate → near-identity at
    init → stable training).
    """

    def __init__(self, d_patch: int, nheads: int, d_cond: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mha = MultiHeadAttentionADALN(d_patch, nheads, d_cond)
        self.mlp = TransitionADALN(dim=d_patch, dim_cond=d_cond, expansion_factor=mlp_ratio)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, P², d_patch]
            cond: [B, P², d_cond]
            mask: [B, P²] bool

        Returns:
            [B, P², d_patch]
        """
        x = x + self.mha(x, cond, mask)
        x = x + self.mlp(x, cond, mask)
        return x * mask[..., None]


class ContactMapSiT(nn.Module):
    """Hybrid Protein Encoder + SiT architecture for contact map diffusion.

    Implements the same model.nn interface as ProteinTransformerAF3 so it can
    be dropped in via a config change without touching any existing code.

    Required batch keys:
        contact_map_t   [B, L, L]       noisy contact map
        mask            [B, L] bool
        t               [B]             flow-matching timestep ∈ [0, 1]
        residue_type    [B, L] long     amino-acid indices (0–20)
        contact_map_sc  [B, L, L]       self-conditioning contact map (optional)
        cath_code_indices               fold conditioning (optional)

    Output dict keys (same as ProteinTransformerAF3 in contact_map_mode):
        contact_map_logits  [B, L, L]   raw logits
        contact_map_pred    [B, L, L]   sigmoid (or tanh) of logits
    """

    def __init__(self, **kwargs):
        super().__init__()

        # ── Phase 1 hyper-parameters ─────────────────────────────────────────
        self.token_dim = int(kwargs["token_dim"])
        self.pair_repr_dim = int(kwargs["pair_repr_dim"])
        self.dim_cond = int(kwargs["dim_cond"])
        self.n_enc_layers = int(kwargs.get("n_enc_layers", 4))
        self.enc_pair_update_every_n = int(kwargs.get("enc_pair_update_every_n", 2))
        nheads_enc = int(kwargs.get("nheads_enc", 8))
        use_qkln = bool(kwargs.get("use_qkln", True))

        # ── Phase 2 hyper-parameters ─────────────────────────────────────────
        self.d_patch = int(kwargs["d_patch"])
        self.n_dit_layers = int(kwargs.get("n_dit_layers", 12))
        nheads_dit = int(kwargs.get("nheads_dit", 8))
        self.d_cond_dit = int(kwargs.get("d_cond_dit", self.dim_cond))
        mlp_ratio = float(kwargs.get("mlp_ratio", 4.0))
        self.patch_size = int(kwargs.get("patch_size", 5))

        # ── Required attributes read by Proteina ─────────────────────────────
        self.contact_map_mode = True
        self.predict_coords = None
        self.predict_dssp = False
        self.non_contact_value = int(kwargs.get("non_contact_value", 0))
        if self.non_contact_value not in (0, -1):
            raise ValueError(f"non_contact_value must be 0 or -1, got {self.non_contact_value}")

        # ── Feature factory kwargs (passed through) ───────────────────────────
        # Strip keys that FeatureFactory doesn't expect to avoid conflicts
        _ff_skip = {"feature_embedding_mode", "individual_feat_ln"}
        _feat_kwargs = {k: v for k, v in kwargs.items() if k not in _ff_skip}

        # ── Phase 1: Protein Encoder ──────────────────────────────────────────

        # Initial sequence representation (residue identity, positional encoding)
        feats_init_seq = list(kwargs.get("feats_init_seq", ["res_seq_pdb_idx"]))
        self.init_repr_factory = FeatureFactory(
            feats=feats_init_seq,
            dim_feats_out=self.token_dim,
            use_ln_out=False,
            mode="seq",
            use_residue_type_emb=bool(kwargs.get("residue_type_emb_init_seq", False)),
            feature_embedding_mode=kwargs.get("feature_embedding_mode", "concat"),
            individual_feat_ln=bool(kwargs.get("individual_feat_ln", True)),
            **_feat_kwargs,
        )

        # Conditioning variables (time, fold, optional sequence)
        feats_cond_seq = list(kwargs.get("feats_cond_seq", ["time_emb", "fold_emb"]))
        self.cond_factory = FeatureFactory(
            feats=feats_cond_seq,
            dim_feats_out=self.dim_cond,
            use_ln_out=False,
            mode="seq",
            use_residue_type_emb=bool(kwargs.get("residue_type_emb_cond_seq", False)),
            feature_embedding_mode=kwargs.get("feature_embedding_mode", "concat"),
            individual_feat_ln=bool(kwargs.get("individual_feat_ln", True)),
            **_feat_kwargs,
        )
        self.transition_c_1 = Transition(self.dim_cond, expansion_factor=2)
        self.transition_c_2 = Transition(self.dim_cond, expansion_factor=2)

        # Initial pair representation (sequence separation + contact_map_sc)
        feats_pair_repr = list(kwargs.get("feats_pair_repr", ["rel_seq_sep", "contact_map_sc"]))
        feats_pair_cond = list(kwargs.get("feats_pair_cond", ["time_emb"]))
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=feats_pair_repr,
            feats_cond=feats_pair_cond,
            dim_feats_out=self.pair_repr_dim,
            dim_cond_pair=self.dim_cond,
            **_feat_kwargs,
        )

        # Embed noisy contact map directly into pair representation space
        contact_map_input_dim = int(kwargs.get("contact_map_input_dim", 1))
        self.linear_contact_embed = nn.Linear(contact_map_input_dim, self.pair_repr_dim, bias=False)

        # N_enc encoder blocks (pair-biased MHSA + adaLN transition, no triangle mult)
        self.encoder_blocks = nn.ModuleList([
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
        ])

        # Pair update blocks (outer-product style, no triangle multiplicative updates)
        n_pair_updates = max(1, self.n_enc_layers // self.enc_pair_update_every_n)
        self.pair_updates = nn.ModuleList([
            PairReprUpdate(
                token_dim=self.token_dim,
                pair_dim=self.pair_repr_dim,
                expansion_factor_transition=2,
                use_tri_mult=False,
            )
            for _ in range(n_pair_updates)
        ])

        # ── Bridge ────────────────────────────────────────────────────────────

        # Project pooled pair representation to patch token space
        self.pair_to_patch = nn.Linear(self.pair_repr_dim, self.d_patch, bias=False)

        # Project global conditioning (mean of c) to DiT conditioning dimension
        self.cond_to_dit = nn.Linear(self.dim_cond, self.d_cond_dit, bias=True)

        # ── Phase 2: SiT ──────────────────────────────────────────────────────

        # Patchify: two input channels (contact_map_t + contact_map_sc)
        self.patch_embed = nn.Conv2d(
            in_channels=2,
            out_channels=self.d_patch,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # SiT blocks (full MHSA, adaLN-Zero — no pair bias needed here)
        self.dit_blocks = nn.ModuleList([
            SiTBlock(self.d_patch, nheads_dit, self.d_cond_dit, mlp_ratio)
            for _ in range(self.n_dit_layers)
        ])

        # Output head: unpatchify to [B, P², patch_size²]
        self.output_norm = nn.LayerNorm(self.d_patch)
        self.output_head = nn.Linear(self.d_patch, self.patch_size * self.patch_size, bias=True)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict:
        """
        Args:
            batch: dict with keys as documented in class docstring.

        Returns:
            dict with contact_map_logits [B,L,L] and contact_map_pred [B,L,L].
        """
        contact_map_t = batch["contact_map_t"]   # [B, L, L]
        mask = batch["mask"]                      # [B, L] bool
        B, L = mask.shape
        device = contact_map_t.device
        dtype = contact_map_t.dtype

        contact_map_sc = batch.get(
            "contact_map_sc", torch.zeros_like(contact_map_t)
        )  # [B, L, L]

        pair_mask = mask[:, :, None] & mask[:, None, :]  # [B, L, L] bool

        # Feature factories that check for "x_t" just to get (B, N) dimensions.
        # Inject a dummy zero tensor so existing feature code works unchanged.
        if "x_t" not in batch:
            batch = dict(batch)
            batch["x_t"] = torch.zeros(B, L, 3, device=device, dtype=dtype)

        # ── Phase 1: Protein Encoder ──────────────────────────────────────────

        # Initial sequence representation
        seq = self.init_repr_factory(batch)         # [B, L, token_dim]
        seq = seq * mask[..., None]

        # Conditioning variables c (time + fold + optional sequence type)
        c = self.cond_factory(batch)                # [B, L, dim_cond]
        c = self.transition_c_1(c, mask)
        c = self.transition_c_2(c, mask)

        # Initial pair representation (from rel_seq_sep + contact_map_sc features)
        pair_rep = self.pair_repr_builder(batch)    # [B, L, L, pair_repr_dim]

        # Embed noisy contact map and add to pair representation
        contact_embed = self.linear_contact_embed(
            contact_map_t.unsqueeze(-1)
        )  # [B, L, L, pair_repr_dim]
        pair_rep = (pair_rep + contact_embed) * pair_mask[..., None].to(dtype)

        # Encoder blocks interleaved with pair updates
        pair_update_idx = 0
        for i, enc_block in enumerate(self.encoder_blocks):
            seq = enc_block(seq, pair_rep, c, mask)
            if (i + 1) % self.enc_pair_update_every_n == 0 and pair_update_idx < len(self.pair_updates):
                pair_rep = self.pair_updates[pair_update_idx](seq, pair_rep, mask)
                pair_update_idx += 1

        # Run any remaining pair updates (if n_enc_layers not divisible)
        while pair_update_idx < len(self.pair_updates):
            pair_rep = self.pair_updates[pair_update_idx](seq, pair_rep, mask)
            pair_update_idx += 1

        # ── Bridge ────────────────────────────────────────────────────────────

        # Pad L to nearest multiple of patch_size
        pad_L = (self.patch_size - L % self.patch_size) % self.patch_size
        L_pad = L + pad_L

        # Pad contact maps and pair_rep
        if pad_L > 0:
            contact_map_t_pad = F.pad(contact_map_t, (0, pad_L, 0, pad_L))
            contact_map_sc_pad = F.pad(contact_map_sc, (0, pad_L, 0, pad_L))
            pair_rep_pad = F.pad(pair_rep, (0, 0, 0, pad_L, 0, pad_L))
            pair_mask_float_pad = F.pad(pair_mask.to(dtype), (0, pad_L, 0, pad_L))
        else:
            contact_map_t_pad = contact_map_t
            contact_map_sc_pad = contact_map_sc
            pair_rep_pad = pair_rep
            pair_mask_float_pad = pair_mask.to(dtype)

        P = L_pad // self.patch_size  # number of patches per side

        # Pool pair_rep to patch resolution and project to d_patch
        # pair_rep_pad: [B, L_pad, L_pad, pair_repr_dim] → need [B, pair_repr_dim, L_pad, L_pad]
        pair_4d = pair_rep_pad.permute(0, 3, 1, 2)                   # [B, pair_repr_dim, L_pad, L_pad]
        pair_pool = F.avg_pool2d(pair_4d, self.patch_size)            # [B, pair_repr_dim, P, P]
        pair_pool = pair_pool.permute(0, 2, 3, 1).reshape(B, P * P, self.pair_repr_dim)
        pair_ctx = self.pair_to_patch(pair_pool)                      # [B, P², d_patch]

        # Global conditioning: mean of per-residue c → project to d_cond_dit
        c_global = c.mean(dim=1)                                      # [B, dim_cond]
        cond_global = self.cond_to_dit(c_global)                      # [B, d_cond_dit]
        cond = cond_global.unsqueeze(1).expand(B, P * P, self.d_cond_dit)  # [B, P², d_cond_dit]

        # Patch mask from pooled pair mask
        patch_mask_2d = F.avg_pool2d(
            pair_mask_float_pad.unsqueeze(1), self.patch_size
        ).squeeze(1)                                                   # [B, P, P]
        patch_mask = (patch_mask_2d > 0).reshape(B, P * P)            # [B, P²] bool

        # ── Phase 2: SiT ──────────────────────────────────────────────────────

        # Patchify via Conv2d: stack two input channels
        contact_input = torch.stack(
            [contact_map_t_pad, contact_map_sc_pad], dim=1
        )  # [B, 2, L_pad, L_pad]
        tokens = self.patch_embed(contact_input)                       # [B, d_patch, P, P]
        tokens = tokens.permute(0, 2, 3, 1).reshape(B, P * P, self.d_patch)  # [B, P², d_patch]

        # Add pair context (local structural conditioning) and 2D positional embedding
        tokens = tokens + pair_ctx
        tokens = tokens + sinusoidal_2d_pos_emb(P, self.d_patch, device).unsqueeze(0)

        # Apply patch mask
        tokens = tokens * patch_mask[..., None]

        # N_dit SiT blocks
        for block in self.dit_blocks:
            tokens = tokens + block(tokens, cond, patch_mask)

        # ── Output head: unpatchify ───────────────────────────────────────────
        tokens = self.output_norm(tokens)                              # [B, P², d_patch]
        patch_preds = self.output_head(tokens)                         # [B, P², patch_size²]

        # Reshape to spatial [B, L_pad, L_pad]
        ps = self.patch_size
        patch_preds = patch_preds.reshape(B, P, P, ps, ps)
        # permute so spatial dims come together: (B, P, ps, P, ps) → (B, P*ps, P*ps)
        contact_pred = patch_preds.permute(0, 1, 3, 2, 4).reshape(B, L_pad, L_pad)

        # Crop back to original length
        if pad_L > 0:
            contact_pred = contact_pred[:, :L, :L]

        # Enforce symmetry and apply pair mask
        contact_pred = (contact_pred + contact_pred.transpose(-1, -2)) / 2.0
        contact_pred = contact_pred * pair_mask.to(dtype)

        # Normalise to probability / score
        if self.non_contact_value == 0:
            contact_map_pred = torch.sigmoid(contact_pred)
        else:
            contact_map_pred = torch.tanh(contact_pred)

        return {
            "contact_map_logits": contact_pred,
            "contact_map_pred": contact_map_pred,
        }
