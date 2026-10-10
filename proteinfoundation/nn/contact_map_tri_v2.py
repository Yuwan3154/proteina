# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""T8 tri (ConFind tri v2, user 2026-10-09): two stages of triangle-multiplication blocks.

Stage A runs on the joint (L+T)^2 grid of query residues and topology-reference elements, exactly as
ContactMapTriSiT does (left-aligned positions, region embedding, the reference's element-pair features in
its own block). The alignment and masked-token heads read the grid at the END of stage A, just before the
template is dropped (user: "The QT alignment loss will be applied at the middle right before the
transition", answer (b)). Stage B then runs on the query block alone, L^2, and the contact map is read
from its output.

Other differences from ContactMapTriSiT, each a user decision of 2026-10-09:
  * pair width 128, tri hidden 128, 4 + 4 blocks;
  * SwiGLU transition with hidden width 512 (AF3 SI Algorithm 11 form: two bias-free input projections,
    swish(a) * b, bias-free output projection; OpenFold init: LeCun in, zero 'final' out);
  * up to 96 reference elements, loops included as elements (one loop token), plus a per-element feature
    vector (``topology_he_elem_feat``, the element length) added to the element rows and columns.
Triangle updates run through one fused projection (the five input linears of AF2 Alg. 11/12 as one matmul,
as in the ProteinEBMalign pair TRM); parameters and state_dict keys are the OpenFold modules', so the
fused path is checked against the unfused one.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinfoundation.datasets.sse_topology import MASK_TOKEN as TOPOLOGY_MASK_TOKEN
from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES, PAIR_FEATURE_MODES
from proteinfoundation.nn.contact_map_tri import (
    N_BLOCK_TYPES,
    ContactMapTriSiT,
    TimestepEmbedding,
    _pair_feature_indices,
)
from proteinfoundation.openfold_stub.model.primitives import LayerNorm, Linear
from proteinfoundation.openfold_stub.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


def _fused_trimul(self, z, mask):
    """TriangleMultiplicativeUpdate.forward with linear_{a_p, a_g, b_p, b_g, g} as one matmul. Same math."""
    c = self.c_hidden
    mask = mask.unsqueeze(-1)
    z = self.layer_norm_in(z)
    lins = (self.linear_a_p, self.linear_a_g, self.linear_b_p, self.linear_b_g, self.linear_g)
    w = torch.cat([l.weight for l in lins])
    b = torch.cat([l.bias for l in lins])
    ap, ag, bp, bg, g = F.linear(z, w, b).split([c, c, c, c, self.c_z], dim=-1)
    a = ap * torch.sigmoid(ag) * mask
    bb = bp * torch.sigmoid(bg) * mask
    x = self._combine_projections(a, bb)
    x = self.linear_z(self.layer_norm_out(x))
    return x * torch.sigmoid(g)


class TriMulOutgoingFused(TriangleMultiplicationOutgoing):
    forward = _fused_trimul


class TriMulIncomingFused(TriangleMultiplicationIncoming):
    forward = _fused_trimul


class SwiGLUTransition(nn.Module):
    """AF3 SI Algorithm 11 transition: LN -> (a, b) = bias-free projections -> swish(a) * b -> bias-free out."""

    def __init__(self, c_z: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.layer_norm = LayerNorm(c_z)
        self.linear_a = Linear(c_z, hidden, bias=False, init="default")
        self.linear_b = Linear(c_z, hidden, bias=False, init="default")
        self.linear_out = Linear(hidden, c_z, bias=False, init="final")

    def forward(self, z, mask):
        z = self.layer_norm(z)
        a, b = F.linear(z, torch.cat([self.linear_a.weight, self.linear_b.weight])).split(self.hidden, dim=-1)
        return self.linear_out(F.silu(a) * b) * mask.unsqueeze(-1)


class TriBlockV2(nn.Module):
    """ContactMapTriSiT's TriBlock (FiLM of each sub-module's input by time, zero-init) with a SwiGLU transition."""

    def __init__(self, dim: int, tri_hidden: int, transition_hidden: int, dim_cond: int):
        super().__init__()
        self.tri_out = TriMulOutgoingFused(c_z=dim, c_hidden=tri_hidden)
        self.tri_in = TriMulIncomingFused(c_z=dim, c_hidden=tri_hidden)
        self.transition = SwiGLUTransition(dim, transition_hidden)
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


class ContactMapTriV2(nn.Module):
    """Stage A: n_blocks_ref blocks on (L+T)^2; heads; stage B: n_blocks_query blocks on L^2; contact map."""

    def __init__(self, **kwargs):
        super().__init__()
        self.dim = int(kwargs["pair_dim"])
        self.tri_hidden = int(kwargs["tri_hidden"])
        self.n_blocks_ref = int(kwargs["n_blocks_ref"])
        self.n_blocks_query = int(kwargs["n_blocks_query"])
        self.transition_hidden = int(kwargs["transition_hidden"])
        self.dim_cond = int(kwargs["dim_cond"])
        self.max_topology_he_len = int(kwargs["max_topology_he_len"])
        self.max_rel_pos = int(kwargs["max_rel_pos"])
        self.topology_vocab_size = int(kwargs["topology_vocab_size"])
        self.n_elem_features = int(kwargs["n_elem_features"])
        self.pair_ref_features = kwargs.get("pair_ref_features", "both")
        if self.pair_ref_features not in PAIR_FEATURE_MODES:
            raise ValueError(f"pair_ref_features must be one of {sorted(PAIR_FEATURE_MODES)}, got {self.pair_ref_features!r}")
        self.pair_feat_idx = _pair_feature_indices(self.pair_ref_features)

        self.contact_map_mode = True
        self.contact_map_input_dim = int(kwargs.get("contact_map_input_dim", 1))
        self.non_contact_value = int(kwargs.get("non_contact_value", 0))
        self.predict_coords = None  # see ContactMapTriSiT: None, not False

        self.seq_emb = nn.Embedding(int(kwargs.get("n_residue_types", 22)), self.dim)
        self.topo_emb = nn.Embedding(self.topology_vocab_size, self.dim, padding_idx=0)
        self.elem_in = nn.Linear(self.n_elem_features, self.dim)
        self.block_type_emb = nn.Embedding(N_BLOCK_TYPES, self.dim)
        self.rel_pos_emb = nn.Embedding(2 * self.max_rel_pos + 2, self.dim)
        self.time_emb = TimestepEmbedding(self.dim_cond)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.dim_cond, self.dim_cond), nn.SiLU(), nn.Linear(self.dim_cond, self.dim_cond),
        )
        self.cell_in = nn.Linear(2 + len(self.pair_feat_idx), self.dim)

        blk_args = (self.dim, self.tri_hidden, self.transition_hidden, self.dim_cond)
        self.blocks_ref = nn.ModuleList(TriBlockV2(*blk_args) for _ in range(self.n_blocks_ref))
        self.blocks_query = nn.ModuleList(TriBlockV2(*blk_args) for _ in range(self.n_blocks_query))

        self.align_head = self.align_none = self.mlm_head = None
        if dict(kwargs.get("align_head") or {}).get("enabled", False):
            self.align_head = nn.Linear(self.dim, 1)
            self.align_none = nn.Linear(self.dim, 1)
        if dict(kwargs.get("mlm_head") or {}).get("enabled", False):
            self.mlm_head = nn.Linear(self.dim, self.topology_vocab_size)
        # only with a head to read it: an unconsumed LayerNorm gets no gradient and DDP aborts
        self.mid_norm = nn.LayerNorm(self.dim) if (self.align_head is not None or self.mlm_head is not None) else None

        self.out_norm = nn.LayerNorm(self.dim)
        self.out = nn.Linear(self.dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.configure_compile(kwargs)

    configure_compile = ContactMapTriSiT.configure_compile
    _bucket = staticmethod(ContactMapTriSiT._bucket)
    _unpad_out = staticmethod(ContactMapTriSiT._unpad_out)

    def _pad_batch(self, batch: Dict, Lp: int, Tp: int) -> Dict:
        out = ContactMapTriSiT._pad_batch(self, batch, Lp, Tp)
        ef = batch.get("topology_he_elem_feat")
        if ef is not None and Tp > ef.shape[1]:
            out["topology_he_elem_feat"] = F.pad(ef, (0, 0, 0, Tp - ef.shape[1]))
        return out

    def _forward_impl_sc(self, batch: Dict) -> Dict:
        return self._forward_impl(batch)

    def forward(self, batch: Dict, force_compile: bool = False) -> Dict:
        L = batch["contact_map_t"].shape[1]
        tok = batch.get("topology_he_tokens")
        T = tok.shape[1] if tok is not None else 1
        Lp, Tp = self._bucket(L, self.pad_len_buckets), self._bucket(T, self.pad_topo_buckets)
        padded = (Lp, Tp) != (L, T)
        if padded:
            batch = self._pad_batch(batch, Lp, Tp if tok is not None else T)
        if torch.is_grad_enabled():
            fn = self._forward_impl
            if self.use_torch_compile:
                if self._compiled_train is None:
                    self._compiled_train = torch.compile(self._forward_impl, dynamic=False, mode=self.compile_mode_train)
                fn = self._compiled_train
        else:
            fn = self._forward_impl
            if self.use_torch_compile_sc or force_compile:
                if self._compiled_eval is None:
                    self._compiled_eval = torch.compile(self._forward_impl_sc, dynamic=False, mode=self.compile_mode_eval)
                fn = self._compiled_eval
        out = fn(batch)
        return self._unpad_out(out, L, T) if padded else out

    def _forward_impl(self, batch: Dict) -> Dict:
        cm_t = batch["contact_map_t"]
        B, L = cm_t.shape[0], cm_t.shape[1]
        device, dtype = cm_t.device, cm_t.dtype
        mask = batch["mask"]

        he_tokens = batch.get("topology_he_tokens")
        if he_tokens is None:
            he_tokens = torch.full((B, 1), TOPOLOGY_MASK_TOKEN, dtype=torch.long, device=device)
            he_pos = torch.zeros(B, 1, device=device)
            he_feat = torch.zeros(B, 1, 1, N_PAIR_FEATURES, device=device)
            elem_feat = torch.zeros(B, 1, self.n_elem_features, device=device)
        else:
            he_pos = batch["topology_he_pos_raw"].float()
            he_feat = batch["topology_he_feat"].float()
            elem_feat = batch["topology_he_elem_feat"].float()
        T = he_tokens.shape[1]
        assert elem_feat.shape == (B, T, self.n_elem_features), (tuple(elem_feat.shape), (B, T, self.n_elem_features))
        he_valid = he_tokens > 0

        tok_mask = torch.cat([mask.bool(), he_valid], dim=1)
        pair_mask = (tok_mask[:, :, None] & tok_mask[:, None, :]).to(dtype)
        N = L + T

        q_pos = torch.arange(L, device=device, dtype=torch.float32)[None].expand(B, L)
        pos = torch.cat([q_pos, he_pos], dim=1)
        rel = (pos[:, :, None] - pos[:, None, :]).round().long()
        rel = rel.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        z = self.rel_pos_emb(rel)

        is_t = torch.zeros(N, dtype=torch.long, device=device)
        is_t[L:] = 1
        z = z + self.block_type_emb(is_t[:, None] * 2 + is_t[None, :])[None]

        rtype = batch.get("residue_type")
        if rtype is not None:
            e = self.seq_emb(rtype.long().clamp(min=0))
            seq_dropped = batch.get("seq_dropped")
            if seq_dropped is not None:
                e = e * (1.0 - seq_dropped.reshape(B, -1)[:, :1, None].to(e.dtype))
            e = F.pad(e, (0, 0, 0, T))
            z = z + e[:, :, None, :] + e[:, None, :, :]
        te = (self.topo_emb(he_tokens.clamp(min=0)) + self.elem_in(elem_feat.to(z.dtype))) * he_valid[..., None]
        te = F.pad(te, (0, 0, L, 0))
        z = z + te[:, :, None, :] + te[:, None, :, :]

        cm_sc = batch.get("contact_map_sc")
        if cm_sc is None:
            cm_sc = torch.zeros_like(cm_t)
        cells = z.new_zeros(B, N, N, 2 + len(self.pair_feat_idx))
        cells[:, :L, :L, 0] = cm_t
        cells[:, :L, :L, 1] = cm_sc
        cells[:, L:, L:, 2:] = he_feat[..., self.pair_feat_idx].to(z.dtype)
        z = (z + self.cell_in(cells)) * pair_mask[..., None]

        cond = self.cond_mlp(self.time_emb(batch["t"]))
        for blk in self.blocks_ref:
            z = blk(z, pair_mask, cond)

        q_valid = mask.bool()
        out = {}
        zm = self.mid_norm(z) if self.mid_norm is not None else None
        if self.align_head is not None:
            qt_valid = q_valid[:, :, None] & he_valid[:, None, :]
            out["align_logits"] = self.align_head(zm[:, :L, L:])[..., 0] * qt_valid.to(zm.dtype)
            ar = torch.arange(L, device=device)
            out["align_none_logits"] = self.align_none(zm[:, ar, ar])[..., 0] * q_valid.to(zm.dtype)
        if self.mlm_head is not None:
            at = torch.arange(L, N, device=device)
            out["mlm_logits"] = self.mlm_head(zm[:, at, at])

        q_pair = (q_valid[:, :, None] & q_valid[:, None, :])
        qmask = q_pair.to(dtype)
        zq = z[:, :L, :L] * qmask[..., None]
        for blk in self.blocks_query:
            zq = blk(zq, qmask, cond)

        logits = self.out(self.out_norm(zq))[..., 0]
        logits = 0.5 * (logits + logits.transpose(1, 2))
        logits = logits * q_pair.to(logits.dtype)
        out["contact_map_logits"] = logits
        out["contact_map_pred"] = torch.sigmoid(logits)
        return out
