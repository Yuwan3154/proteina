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
    swish(a) * b, bias-free output projection; AF3 transition_block init: 'relu' (He) in, 'default' out when
    conditioned);
  * up to 96 reference elements, loops included as elements (one loop token), plus a per-element feature
    vector (``topology_he_elem_feat``, the element length) added to the element rows and columns.
Triangle updates run through one fused projection (the five input linears of AF2 Alg. 11/12 as one matmul,
as in the ProteinEBMalign pair TRM); parameters and state_dict keys are the OpenFold modules', so the
fused path is checked against the unfused one.

AlphaFold conventions for everything the user did not specify (user 2026-10-10: "FOLLOW ALPHAFOLD VALUES"):
  * embeddings (relative position with AF2's 2k+1 bins, residue type, template tokens, region) = one-hot -> Linear with
    LeCun init and zero bias (AF2 InputEmbedder), computed as an exact table lookup; per-token features enter the pair
    as a_i + b_j with two separate linears (AF2 InputEmbedder left/right_single); cell input LeCun / zero bias;
  * heads read the pair representation without a LayerNorm through a 'final' (zero) init Linear with bias; the contact
    logits are symmetrised as l + l^T (AF2 / OpenFold DistogramHead and MaskedMSAHead, heads.py);
  * diffusion-time conditioning as in the AF3 code (diffusion_head.py / diffusion_transformer.py): fixed Fourier
    constants (noise_level_embeddings.py) -> LayerNorm without offset -> LinearNoBias -> 2 x unconditioned transition
    n=2; each sub-module's input LayerNorm is adaptive_layernorm (zero-init scale/bias linears) and its output goes
    through adaptive_zero_init (output Linear default init, gate Linear zero weights, bias -2); SwiGLU input
    projections use the 'relu' (He) init. The time t in [0, 1] goes into the Fourier embedding directly (AF3 feeds
    1/4 log(sigma / sigma_data) of its EDM noise level; this model's noise variable is t).
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinfoundation.datasets.sse_topology import MASK_TOKEN as TOPOLOGY_MASK_TOKEN
from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES, PAIR_FEATURE_MODES
from proteinfoundation.nn.af3_fourier_constants import AF3_FOURIER_BIAS, AF3_FOURIER_WEIGHT
from proteinfoundation.nn.contact_map_tri import (
    N_BLOCK_TYPES,
    ContactMapTriSiT,
    _pair_feature_indices,
)
from proteinfoundation.openfold_stub.model.primitives import LayerNorm, Linear, lecun_normal_init_
from proteinfoundation.openfold_stub.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


def _fused_trimul(self, z, mask):
    """TriangleMultiplicativeUpdate.forward with linear_{a_p, a_g, b_p, b_g, g} as one matmul. Same math."""
    return _fused_trimul_core(self, self.layer_norm_in(z), mask)


def _fused_trimul_core(self, z, mask):
    """The update after the input LayerNorm (z already normalised; TriBlockV2 normalises with AdaLN instead)."""
    c = self.c_hidden
    mask = mask.unsqueeze(-1)
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
    core = _fused_trimul_core


class TriMulIncomingFused(TriangleMultiplicationIncoming):
    forward = _fused_trimul
    core = _fused_trimul_core


class SwiGLUTransition(nn.Module):
    """AF3 SI Algorithm 11 transition: LN -> (a, b) = bias-free projections -> swish(a) * b -> bias-free out.
    Inits from AF3 transition_block: ffw_transition1 'relu'; output 'final' when unconditioned, 'default' when
    conditioned (adaptive_zero_init)."""

    def __init__(self, c_z: int, hidden: int, out_init: str = "final"):
        super().__init__()
        self.hidden = hidden
        self.layer_norm = LayerNorm(c_z)
        self.linear_a = Linear(c_z, hidden, bias=False, init="relu")
        self.linear_b = Linear(c_z, hidden, bias=False, init="relu")
        self.linear_out = Linear(hidden, c_z, bias=False, init=out_init)

    def forward(self, z, mask):
        return self.core(self.layer_norm(z), mask)

    def core(self, z, mask):
        a, b = F.linear(z, torch.cat([self.linear_a.weight, self.linear_b.weight])).split(self.hidden, dim=-1)
        return self.linear_out(F.silu(a) * b) * mask.unsqueeze(-1)


class OneHotLinear(nn.Module):
    """AF2 one-hot -> Linear (LeCun weights, zero bias) as an exact table lookup (no one-hot tensor is built)."""

    def __init__(self, n: int, dim: int):
        super().__init__()
        self.linear = Linear(n, dim, init="default")

    def forward(self, idx):
        return F.embedding(idx, self.linear.weight.t()) + self.linear.bias


class FourierEmbedding(nn.Module):
    """AF3 noise_embeddings: cos(2 pi (t w + b)) with AF3's fixed 256-dim w, b (non-persistent, so not in ckpts)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.tensor(AF3_FOURIER_WEIGHT, dtype=torch.float32), persistent=False)
        self.register_buffer("b", torch.tensor(AF3_FOURIER_BIAS, dtype=torch.float32), persistent=False)

    def forward(self, t):
        return torch.cos(2.0 * math.pi * (t.float()[:, None] * self.w + self.b))


class CondTransition(nn.Module):
    """AF3 unconditioned transition_block on the conditioning vector (n = 2, diffusion_head.py), residual."""

    def __init__(self, c: int, n: int):
        super().__init__()
        self.t = SwiGLUTransition(c, n * c)

    def forward(self, s):
        return s + self.t(s[:, None, None, :], s.new_ones(s.shape[0], 1, 1))[:, 0, 0, :]


class AdaLN(nn.Module):
    """AF3 adaptive_layernorm: a = sigmoid(Linear(LN_scale(s))) * LN(a) + LinearNoBias(LN_scale(s)), linears zero-init."""

    def __init__(self, dim: int, dim_cond: int):
        super().__init__()
        self.ln_a = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_s = nn.LayerNorm(dim_cond, elementwise_affine=True, bias=False)
        self.linear_s = Linear(dim_cond, dim, init="final")
        self.linear_nobias_s = Linear(dim_cond, dim, bias=False, init="final")

    def forward(self, a, s):
        s = self.ln_s(s)
        return torch.sigmoid(self.linear_s(s))[:, None, None, :] * self.ln_a(a) + self.linear_nobias_s(s)[:, None, None, :]


def _gate(dim_cond: int, dim: int) -> Linear:
    """AF3 adaptive_zero_init gate: sigmoid(Linear(s)), zero weights, bias init -2."""
    g = Linear(dim_cond, dim, init="final")
    with torch.no_grad():
        g.bias.fill_(-2.0)
    return g


class TriBlockV2(nn.Module):
    """Outgoing + incoming fused triangle multiplication + SwiGLU transition, each conditioned on time as in AF3:
    AdaLN in place of the sub-module's input LayerNorm, adaptive_zero_init on its output (the output Linear gets the
    default init, as AF3 gives a conditioned output projection; AF3 has no conditioned trimul, so its linear_z is
    treated as that projection)."""

    def __init__(self, dim: int, tri_hidden: int, transition_hidden: int, dim_cond: int):
        super().__init__()
        self.tri_out = TriMulOutgoingFused(c_z=dim, c_hidden=tri_hidden)
        self.tri_in = TriMulIncomingFused(c_z=dim, c_hidden=tri_hidden)
        self.transition = SwiGLUTransition(dim, transition_hidden, out_init="default")
        # AdaLN replaces each module's own input LayerNorm, so those parameters go (no unused parameters for DDP)
        self.tri_out.layer_norm_in = nn.Identity()
        self.tri_in.layer_norm_in = nn.Identity()
        for tri in (self.tri_out, self.tri_in):
            with torch.no_grad():
                lecun_normal_init_(tri.linear_z.weight)
                tri.linear_z.bias.zero_()
        self.transition.layer_norm = nn.Identity()
        self.adaln = nn.ModuleList(AdaLN(dim, dim_cond) for _ in range(3))
        self.gate = nn.ModuleList(_gate(dim_cond, dim) for _ in range(3))

    def forward(self, z, pair_mask, cond):
        m = pair_mask[..., None]
        for k, mod in enumerate((self.tri_out, self.tri_in, self.transition)):
            g = torch.sigmoid(self.gate[k](cond))[:, None, None, :]
            z = z + g * mod.core(self.adaln[k](z, cond), pair_mask) * m
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

        n_rt = int(kwargs.get("n_residue_types", 22))
        self.seq_emb_i, self.seq_emb_j = OneHotLinear(n_rt, self.dim), OneHotLinear(n_rt, self.dim)
        # padded elements are masked in forward
        self.topo_emb_i = OneHotLinear(self.topology_vocab_size, self.dim)
        self.topo_emb_j = OneHotLinear(self.topology_vocab_size, self.dim)
        self.elem_in_i = Linear(self.n_elem_features, self.dim, init="default")  # AF2 LeCun-normal init
        self.elem_in_j = Linear(self.n_elem_features, self.dim, init="default")
        self.block_type_emb = OneHotLinear(N_BLOCK_TYPES, self.dim)
        self.rel_pos_emb = OneHotLinear(2 * self.max_rel_pos + 1, self.dim)  # AF2 relpos: 2k + 1 bins
        self.fourier = FourierEmbedding()
        self.fourier_ln = nn.LayerNorm(256, bias=False)  # AF3 noise_embedding_initial_norm: create_offset=False
        self.cond_in = Linear(256, self.dim_cond, bias=False, init="default")
        self.cond_transitions = nn.ModuleList(CondTransition(self.dim_cond, 2) for _ in range(2))
        self.cell_in = Linear(2 + len(self.pair_feat_idx), self.dim, init="default")

        blk_args = (self.dim, self.tri_hidden, self.transition_hidden, self.dim_cond)
        self.blocks_ref = nn.ModuleList(TriBlockV2(*blk_args) for _ in range(self.n_blocks_ref))
        self.blocks_query = nn.ModuleList(TriBlockV2(*blk_args) for _ in range(self.n_blocks_query))

        self.align_head = self.align_none = self.mlm_head = None
        if dict(kwargs.get("align_head") or {}).get("enabled", False):
            self.align_head = Linear(self.dim, 1, init="final")  # AF2 heads: 'final' (zero) init
            self.align_none = Linear(self.dim, 1, init="final")
        if dict(kwargs.get("mlm_head") or {}).get("enabled", False):
            self.mlm_head = Linear(self.dim, self.topology_vocab_size, init="final")
        self.out = Linear(self.dim, 1, init="final")  # AF2/OpenFold DistogramHead: final init, bias, no LayerNorm
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
            r = rtype.long().clamp(min=0)
            keep = 1.0
            seq_dropped = batch.get("seq_dropped")
            if seq_dropped is not None:
                keep = 1.0 - seq_dropped.reshape(B, -1)[:, :1, None].to(z.dtype)
            ei = F.pad(self.seq_emb_i(r) * keep, (0, 0, 0, T))
            ej = F.pad(self.seq_emb_j(r) * keep, (0, 0, 0, T))
            z = z + ei[:, :, None, :] + ej[:, None, :, :]
        tk, ef, hv = he_tokens.clamp(min=0), elem_feat.to(z.dtype), he_valid[..., None]
        ti = F.pad((self.topo_emb_i(tk) + self.elem_in_i(ef)) * hv, (0, 0, L, 0))
        tj = F.pad((self.topo_emb_j(tk) + self.elem_in_j(ef)) * hv, (0, 0, L, 0))
        z = z + ti[:, :, None, :] + tj[:, None, :, :]

        cm_sc = batch.get("contact_map_sc")
        if cm_sc is None:
            cm_sc = torch.zeros_like(cm_t)
        cells = z.new_zeros(B, N, N, 2 + len(self.pair_feat_idx))
        cells[:, :L, :L, 0] = cm_t
        cells[:, :L, :L, 1] = cm_sc
        cells[:, L:, L:, 2:] = he_feat[..., self.pair_feat_idx].to(z.dtype)
        z = (z + self.cell_in(cells)) * pair_mask[..., None]

        cond = self.cond_in(self.fourier_ln(self.fourier(batch["t"])))
        for tr in self.cond_transitions:
            cond = tr(cond)
        for blk in self.blocks_ref:
            z = blk(z, pair_mask, cond)

        q_valid = mask.bool()
        out = {}
        if self.align_head is not None:
            qt_valid = q_valid[:, :, None] & he_valid[:, None, :]
            out["align_logits"] = self.align_head(z[:, :L, L:])[..., 0] * qt_valid.to(z.dtype)
            ar = torch.arange(L, device=device)
            out["align_none_logits"] = self.align_none(z[:, ar, ar])[..., 0] * q_valid.to(z.dtype)
        if self.mlm_head is not None:
            at = torch.arange(L, N, device=device)
            out["mlm_logits"] = self.mlm_head(z[:, at, at])

        q_pair = (q_valid[:, :, None] & q_valid[:, None, :])
        qmask = q_pair.to(dtype)
        zq = z[:, :L, :L] * qmask[..., None]
        for blk in self.blocks_query:
            zq = blk(zq, qmask, cond)

        logits = self.out(zq)[..., 0]
        logits = logits + logits.transpose(1, 2)
        logits = logits * q_pair.to(logits.dtype)
        out["contact_map_logits"] = logits
        out["contact_map_pred"] = torch.sigmoid(logits)
        return out
