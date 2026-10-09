"""1D template model with fixed-window compression (user directives 2026-10-08, 2026-10-09).

Regime: proteina CA flow matching (predict clean CA). Layout:
  1D pair-biased attention over residues (ProteinTransformerAF3 blocks) -> SSE head (DSSP CE, aux only)
  -> compress by LOCAL ATTENTION over windows of 9 residues, each overlapping the next by 4 (stride 5), right-padded
     (user 2026-10-09; replaces the SSE-run segmentation): one token per window, a K x K window pair rep, concatenated
     with the T template topology elements on one (K+T)^2 grid (as in ContactMapTriSiT), updated ONLY by the tri's TriBlocks
  -> decompress 2D -> 1D (one of four designs, `decompress`); a residue in two windows averages the two windows' outputs
  -> 1D pair-biased attention -> CA head, outer-product distogram head.
The four decompressors (research/k_pair2single.md §5) all write h = s_pre + o with zero-initialised outputs:
  pair_row_xattn  residue query (skip single + position-in-window) attends over its window's row/column of the grid
  pair_bias       no vector readout; the compressed pair, expanded to residue pairs, biases the post-expansion attention
  basis_pool      diag / row / column / template means of the window's grid row -> MLP -> FiLM by position-in-window
  spectral        eigenvectors of symmetrised projections of Z_KK, made sign-invariant with SignNet
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES, PAIR_FEATURE_MODES, PAIR_FEATURE_NAMES
from proteinfoundation.datasets.sse_topology import MASK_TOKEN as TOPOLOGY_MASK_TOKEN
from proteinfoundation.nn.contact_map_tri import N_BLOCK_TYPES, TimestepEmbedding, TriBlock
from proteinfoundation.nn.protein_transformer import MultiheadAttnAndTransition, PairReprBuilder, Transition
from proteinfoundation.nn.feature_factory import FeatureFactory

DECOMPRESSORS = ("pair_row_xattn", "pair_bias", "basis_pool", "spectral")
WIN, STRIDE = 9, 5  # user 2026-10-09: windows of 9 residues, each overlapping the next by 4
POS_SIN_DIM = 32  # sinusoid width for the offset-in-window features (Non-Attentive Tacotron uses 32)


# ── window helpers ────────────────────────────────────────────────────────────────────────────────
def n_windows(n_res):
    """Windows needed to cover n_res residues; the last one is right-padded."""
    return torch.div((n_res - WIN).clamp(min=0) + STRIDE - 1, STRIDE, rounding_mode="floor") + 1


def window_slots(valid, K_per):
    """The (at most two) windows holding each residue: hi = min(i // STRIDE, K_b - 1), lo = hi - 1 if it reaches i.
    Returns w [2, B, L] (lo, hi), ok [2, B, L] bool, offset-in-window p [2, B, L]."""
    B, L = valid.shape
    i = torch.arange(L, device=valid.device)[None].expand(B, -1)
    hi = torch.minimum(i // STRIDE, (K_per - 1)[:, None])
    lo = hi - 1
    lo_ok = valid & (lo >= 0) & (lo * STRIDE + WIN > i)
    w = torch.stack([lo.clamp(min=0), hi])
    return w, torch.stack([lo_ok, valid]), i[None] - w * STRIDE


def sinusoid(x, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / max(half - 1, 1))
    ang = x.float()[..., None] * freq
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class WindowAttentionPool(nn.Module):
    """Compression: one token per window by multi-head attention over the window's residues, with the window MEAN as
    the query and as the residual (Funnel Transformer's pooled-query attention, Dai et al. 2020)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.ch = heads, d // heads
        self.ln = nn.LayerNorm(d)
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, s, win_idx, win_mask):
        B, K, W = win_mask.shape
        sw = s[torch.arange(B, device=s.device)[:, None, None], win_idx]   # [B, K, W, d]
        m = win_mask.to(s.dtype)
        mean = (sw * m[..., None]).sum(2) / m.sum(2).clamp(min=1)[..., None]
        x = self.ln(sw)
        q = self.q(self.ln(mean)).view(B, K, self.h, self.ch)
        k, v = self.k(x).view(B, K, W, self.h, self.ch), self.v(x).view(B, K, W, self.h, self.ch)
        logit = torch.einsum("bkhc,bkwhc->bkhw", q, k) / math.sqrt(self.ch)
        logit = logit.masked_fill(~win_mask[:, :, None, :], torch.finfo(logit.dtype).min)  # empty windows: uniform, zeroed below
        out = torch.einsum("bkhw,bkwhc->bkhc", torch.softmax(logit, -1), v).reshape(B, K, -1)
        return (mean + self.o(out)) * (m.sum(2) > 0)[..., None].to(s.dtype)


# ── decompressors ─────────────────────────────────────────────────────────────────────────────────
class PairRowCrossAttention(nn.Module):
    """Design B: o_i = W_o concat_h[ g_i^h * sum_m softmax_m(q_i.k_{r,m} + w_b.Z_{r,m}) v_{r,m} ], r = a window of i."""

    def __init__(self, d, c, heads, pos_dim):
        super().__init__()
        self.h, self.ch = heads, c // heads
        self.ln_s, self.ln_z = nn.LayerNorm(d), nn.LayerNorm(c)
        self.q = nn.Linear(d, heads * self.ch, bias=False)
        self.p = nn.Linear(pos_dim, d, bias=False)
        self.k = nn.Linear(c, heads * self.ch, bias=False)
        self.v = nn.Linear(2 * c, heads * self.ch, bias=False)
        self.b = nn.Linear(c, heads, bias=False)
        self.g = nn.Linear(d, heads * self.ch)
        self.o = nn.Linear(heads * self.ch, d, bias=False)
        nn.init.zeros_(self.o.weight)

    def forward(self, s_pre, zn, sid, pos_feat, grid_valid, n_q):
        B, L, _ = s_pre.shape
        bidx = torch.arange(B, device=s_pre.device)[:, None]
        zg = self.ln_z(zn)
        row, col = zg[bidx, sid], zg.transpose(1, 2)[bidx, sid]           # [B, L, N, c]: Z_{r,m}, Z_{m,r}
        q = self.q(self.ln_s(s_pre) + self.p(pos_feat)).view(B, L, self.h, self.ch)
        k = self.k(row).view(B, L, -1, self.h, self.ch)
        v = self.v(torch.cat([row, col], -1)).view(B, L, -1, self.h, self.ch)
        logit = torch.einsum("blhc,blmhc->blhm", q, k) / math.sqrt(self.ch) + self.b(row).permute(0, 1, 3, 2)
        logit = logit.masked_fill(~grid_valid[:, None, None, :], float("-inf"))
        a = torch.softmax(logit, -1)
        out = torch.einsum("blhm,blmhc->blhc", a, v) * torch.sigmoid(self.g(self.ln_s(s_pre))).view(B, L, self.h, self.ch)
        return self.o(out.reshape(B, L, -1))


class ExpandedPairBias(nn.Module):
    """Design C: no vector readout; Linear(LN(Z_KK)) expanded to residue pairs (R E R^T, R = row-normalised window
    membership) and ADDED to the post pair bias."""

    def __init__(self, c, pair_dim):
        super().__init__()
        self.ln, self.proj = nn.LayerNorm(c), nn.Linear(c, pair_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, zn, R, n_q):
        e = self.proj(self.ln(zn[:, :n_q, :n_q]))                         # [B, K, K, pair_dim]
        return torch.einsum("bik,bklc,bjl->bijc", R, e, R)


class BasisPoolFiLM(nn.Module):
    """Design A: x_k = MLP([Z_kk, mean_j Z_kj, mean_j Z_jk, mean_t Z_kt, mean_t Z_tk, 1[T>0]]); FiLM by position."""

    def __init__(self, d, c, pos_dim):
        super().__init__()
        self.ln_z = nn.LayerNorm(c)
        self.mlp = nn.Sequential(nn.Linear(5 * c + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.ln_x = nn.LayerNorm(d)
        self.film = nn.Linear(pos_dim, 2 * d)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o.weight)

    def forward(self, zn, sid, pos_feat, q_valid, t_valid, n_q):
        zg = self.ln_z(zn)
        zq, zqt, ztq = zg[:, :n_q, :n_q], zg[:, :n_q, n_q:], zg[:, n_q:, :n_q]
        qv, tv = q_valid.to(zg.dtype), t_valid.to(zg.dtype)
        nq = qv.sum(1).clamp(min=1)[:, None, None]
        nt = tv.sum(1).clamp(min=1)[:, None, None]
        diag = torch.diagonal(zq, dim1=1, dim2=2).transpose(1, 2)
        row = (zq * qv[:, None, :, None]).sum(2) / nq
        col = (zq * qv[:, :, None, None]).sum(1) / nq
        trow = (zqt * tv[:, None, :, None]).sum(2) / nt
        tcol = (ztq * tv[:, :, None, None]).sum(1) / nt
        has_t = (tv.sum(1) > 0).to(zg.dtype)[:, None, None].expand(-1, n_q, 1)
        x = self.ln_x(self.mlp(torch.cat([diag, row, col, trow, tcol, has_t], -1)))  # [B, K, d]
        xi = torch.gather(x, 1, sid[..., None].expand(-1, -1, x.shape[-1]))
        gamma, beta = self.film(pos_feat).chunk(2, -1)
        return self.o(xi * (1 + gamma) + beta)


class SpectralSignNet(nn.Module):
    """Design D: per head, eigh of sym(Linear(LN(Z_KK))) on the VALID K x K block of each sample (padded rows would add
    repeated zero eigenvalues and NaN gradients), top-r by |lambda|; SignNet phi([v, l]) + phi([-v, l]) -> rho."""

    def __init__(self, d, c, heads, r, hidden):
        super().__init__()
        self.h, self.r = heads, r
        self.ln = nn.LayerNorm(c)
        self.proj = nn.Linear(c, heads, bias=False)
        self.phi = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(heads * hidden, d), nn.SiLU(), nn.Linear(d, d))
        self.o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o.weight)

    def forward(self, zn, sid, K_per, n_q):
        B = zn.shape[0]
        with torch.autocast(zn.device.type, enabled=False):  # fp32 matrix: eigh has no bf16 kernel, near-ties need precision
            m = self.proj(self.ln(zn[:, :n_q, :n_q].float())).permute(0, 3, 1, 2)    # [B, H, K, K]
        m = 0.5 * (m + m.transpose(-1, -2))
        feats = []
        for b in range(B):
            k = int(K_per[b])
            lam, vec = torch.linalg.eigh(m[b, :, :k, :k])                  # [H, k], [H, k, k]
            rr = min(self.r, k)
            idx = lam.abs().topk(rr, dim=-1).indices                        # [H, rr]
            lam_s = torch.gather(lam, 1, idx)                               # [H, rr]
            vec_s = torch.gather(vec, 2, idx[:, None, :].expand(-1, k, -1))  # [H, k, rr]
            lam_b = lam_s[:, None, :].expand(-1, k, -1)
            f = self.phi(torch.stack([vec_s, lam_b], -1)) + self.phi(torch.stack([-vec_s, lam_b], -1))
            f = f.sum(2).permute(1, 0, 2).reshape(k, -1)                    # [k, H*hidden]
            feats.append(F.pad(f, (0, 0, 0, n_q - k)))
        x = self.rho(torch.stack(feats))                                    # [B, K, d]
        return self.o(torch.gather(x, 1, sid[..., None].expand(-1, -1, x.shape[-1])))


# ── model ─────────────────────────────────────────────────────────────────────────────────────────
class CATemplateCompress1D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decompress = kwargs["decompress"]
        assert self.decompress in DECOMPRESSORS, self.decompress
        self.contact_map_mode = False
        self.predict_coords = True
        self.predict_dssp = True
        self.dssp_diffusion_mode = False
        self.use_torch_compile_sc = False
        d, pdim, dc = int(kwargs["token_dim"]), int(kwargs["pair_repr_dim"]), int(kwargs["dim_cond"])
        c = int(kwargs["mid_dim"])
        self.max_rel_pos = int(kwargs.get("max_rel_pos", 64))
        self.pair_feat_idx = [list(PAIR_FEATURE_NAMES).index(n) for n in PAIR_FEATURE_MODES[kwargs.get("pair_ref_features", "both")]]

        _fk = {k: v for k, v in kwargs.items() if k not in ("feature_embedding_mode", "individual_feat_ln")}
        self.linear_3d_embed = nn.Linear(3, d, bias=False)
        self.init_repr_factory = FeatureFactory(feats=kwargs["feats_init_seq"], dim_feats_out=d, use_ln_out=False, mode="seq",
                                                use_residue_type_emb=kwargs.get("residue_type_emb_init_seq", False),
                                                feature_embedding_mode=kwargs.get("feature_embedding_mode", "concat"),
                                                individual_feat_ln=kwargs.get("individual_feat_ln", True), **_fk)
        self.cond_factory = FeatureFactory(feats=kwargs["feats_cond_seq"], dim_feats_out=dc, use_ln_out=False, mode="seq",
                                           feature_embedding_mode=kwargs.get("feature_embedding_mode", "concat"),
                                           individual_feat_ln=kwargs.get("individual_feat_ln", True), **_fk)
        self.transition_c_1 = Transition(dc, expansion_factor=2)
        self.transition_c_2 = Transition(dc, expansion_factor=2)
        self.pair_repr_builder = PairReprBuilder(feats_repr=kwargs["feats_pair_repr"], feats_cond=kwargs.get("feats_pair_cond", []),
                                                 dim_feats_out=pdim, dim_cond_pair=dc, **kwargs)

        def layers(n):
            return nn.ModuleList(MultiheadAttnAndTransition(
                dim_token=d, dim_pair=pdim, nheads=kwargs["nheads"], dim_cond=dc, residual_mha=True,
                residual_transition=True, parallel_mha_transition=False, use_attn_pair_bias=True,
                use_qkln=kwargs.get("use_qkln", True), attn_impl=kwargs.get("attn_impl", "vanilla")) for _ in range(n))
        self.pre_layers, self.post_layers = layers(int(kwargs["n_pre"])), layers(int(kwargs["n_post"]))
        self.dssp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))

        # compression -> (K+T)^2 grid (template side identical to ContactMapTriSiT)
        self.win_pool = WindowAttentionPool(d, int(kwargs["nheads"]))
        self.seg_a, self.seg_b = nn.Linear(d, c, bias=False), nn.Linear(d, c, bias=False)
        self.pair_pool = nn.Linear(pdim, c, bias=False)
        self.topo_emb = nn.Embedding(int(kwargs.get("topology_vocab_size", 44)), c, padding_idx=0)
        self.block_type_emb = nn.Embedding(N_BLOCK_TYPES, c)
        self.rel_pos_emb = nn.Embedding(2 * self.max_rel_pos + 2, c)
        self.cell_in = nn.Linear(len(self.pair_feat_idx), c)
        dcm = int(kwargs.get("mid_dim_cond", 128))
        self.time_emb, self.cond_mlp = TimestepEmbedding(dcm), nn.Sequential(nn.Linear(dcm, dcm), nn.SiLU(), nn.Linear(dcm, dcm))
        self.mid_blocks = nn.ModuleList(TriBlock(c, int(kwargs.get("mid_tri_hidden", c)), int(kwargs.get("mid_transition_n", 4)), dcm)
                                        for _ in range(int(kwargs["n_mid"])))
        self.mid_norm = nn.LayerNorm(c)
        self.align_head, self.align_none = nn.Linear(c, 1), nn.Linear(c, 1)
        self.mlm_head = nn.Linear(c, int(kwargs.get("topology_vocab_size", 44)))

        pos_dim = 2 * POS_SIN_DIM + 1
        if self.decompress == "pair_row_xattn":
            self.decomp = PairRowCrossAttention(d, c, int(kwargs.get("xattn_heads", kwargs["nheads"])), pos_dim)
        elif self.decompress == "pair_bias":
            self.decomp = ExpandedPairBias(c, pdim)
        elif self.decompress == "basis_pool":
            self.decomp = BasisPoolFiLM(d, c, pos_dim)
        else:
            self.decomp = SpectralSignNet(d, c, int(kwargs["spectral_heads"]), int(kwargs["spectral_r"]), int(kwargs["spectral_hidden"]))

        self.coors_3d_decoder = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3, bias=False))
        nb, co = int(kwargs["num_buckets_predict_pair"]), int(kwargs.get("opm_dim", 32))
        self.opm_ln, self.opm_a, self.opm_b = nn.LayerNorm(d), nn.Linear(d, co), nn.Linear(d, co)
        self.opm_out = nn.Linear(co * co, pdim)
        self.pair_head = nn.Sequential(nn.LayerNorm(pdim), nn.Linear(pdim, nb))

    def forward(self, batch: Dict, force_compile: bool = False) -> Dict:
        valid = batch["mask"].bool()  # proteina's attention layers take the BOOLEAN mask
        mask = valid.float()
        B, L = mask.shape
        dev = mask.device
        c_seq = self.cond_factory(batch)
        c_seq = self.transition_c_2(self.transition_c_1(c_seq, valid), valid)
        s = (self.linear_3d_embed(batch["x_t"] * mask[..., None]) + self.init_repr_factory(batch)) * mask[..., None]
        pair = self.pair_repr_builder(batch)
        for lyr in self.pre_layers:
            s = lyr(s, pair, c_seq, valid)
        s_pre = s
        dssp_logits = self.dssp_head(s_pre)

        # windows and compression
        n_res = valid.sum(1)
        assert bool((valid == (torch.arange(L, device=dev)[None] < n_res[:, None])).all()), "windows assume right padding"
        K_per = n_windows(n_res)
        K = int(K_per.max())
        w, ok, p = window_slots(valid, K_per)                                # [2, B, L] each
        A = (F.one_hot(w, K).to(s.dtype) * ok[..., None].to(s.dtype)).sum(0)   # [B, L, K] membership
        R = A / A.sum(-1, keepdim=True).clamp(min=1)                         # row-normalised: residue -> its windows
        r = ok.to(s.dtype) / ok.to(s.dtype).sum(0, keepdim=True).clamp(min=1)  # [2, B, L] slot weights (rows of R)
        n_k = A.sum(1)
        q_valid = n_k > 0
        nn_ = n_k.clamp(min=1)
        idx = torch.arange(L, device=dev, dtype=A.dtype)
        mid = torch.einsum("blk,l->bk", A, idx) / nn_
        win_idx = torch.arange(K, device=dev)[:, None] * STRIDE + torch.arange(WIN, device=dev)[None]   # [K, WIN]
        win_mask = (win_idx < L)[None] & valid[:, win_idx.clamp(max=L - 1)] & (torch.arange(K, device=dev)[None, :, None] < K_per[:, None, None])
        sk = self.win_pool(s_pre, win_idx.clamp(max=L - 1)[None].expand(B, -1, -1), win_mask)
        pz = self.pair_pool(pair)
        zqq = torch.einsum("bik,bijc,bjl->bklc", A, pz, A) / (nn_[:, :, None, None] * nn_[:, None, :, None])
        zqq = zqq + self.seg_a(sk)[:, :, None] + self.seg_b(sk)[:, None]

        he_tok = batch.get("topology_he_tokens")
        if he_tok is None:
            he_tok = torch.full((B, 1), TOPOLOGY_MASK_TOKEN, dtype=torch.long, device=dev)
            he_pos, he_feat = torch.zeros(B, 1, device=dev), torch.zeros(B, 1, 1, N_PAIR_FEATURES, device=dev)
        else:
            he_pos, he_feat = batch["topology_he_pos_raw"].float(), batch["topology_he_feat"].float()
        T = he_tok.shape[1]
        t_valid = he_tok > 0
        g_valid = torch.cat([q_valid, t_valid], 1)
        g_pair = (g_valid[:, :, None] & g_valid[:, None, :]).to(s.dtype)
        pos = torch.cat([mid, he_pos], 1)
        rel = (pos[:, :, None] - pos[:, None, :]).round().long().clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        z = self.rel_pos_emb(rel)
        is_t = torch.zeros(K + T, dtype=torch.long, device=dev)
        is_t[K:] = 1
        z = z + self.block_type_emb(is_t[:, None] * 2 + is_t[None, :])[None]
        te = self.topo_emb(he_tok.clamp(min=0)) * t_valid[..., None]
        z[:, :K, :K] = z[:, :K, :K] + zqq
        z[:, K:, K:] = z[:, K:, K:] + te[:, :, None] + te[:, None] + self.cell_in(he_feat[..., self.pair_feat_idx].to(z.dtype))
        z = z * g_pair[..., None]
        cond = self.cond_mlp(self.time_emb(batch["t"]))
        for blk in self.mid_blocks:
            z = blk(z, g_pair, cond)
        zn = self.mid_norm(z)

        out = {"dssp_logits": dssp_logits}
        row_t = torch.einsum("blk,bkec->blec", R, zn[:, :K, K:])              # [B, L, T, c]: mean of i's windows' template rows
        qt_valid = valid[:, :, None] & t_valid[:, None, :]
        out["align_logits"] = self.align_head(row_t)[..., 0] * qt_valid.to(zn.dtype)
        diag = torch.diagonal(zn[:, :K, :K], dim1=1, dim2=2).transpose(1, 2)  # [B, K, c]
        out["align_none_logits"] = self.align_none(torch.einsum("blk,bkc->blc", R, diag))[..., 0] * valid.to(zn.dtype)
        out["mlm_logits"] = self.mlm_head(torch.diagonal(zn[:, K:, K:], dim1=1, dim2=2).transpose(1, 2))

        # 2D -> 1D: each window decompresses to its residues; a residue in two windows averages them (weights r)
        pos_feat = [torch.cat([sinusoid(p[j], POS_SIN_DIM), sinusoid(WIN - 1 - p[j], POS_SIN_DIM), (p[j] / (WIN - 1))[..., None]], -1).to(s.dtype)
                    for j in range(2)]
        pair_post = pair
        if self.decompress == "pair_row_xattn":
            o = sum(r[j][..., None] * self.decomp(s_pre, zn, w[j], pos_feat[j], g_valid, K) for j in range(2))
        elif self.decompress == "pair_bias":
            o = 0
            pair_post = pair + self.decomp(zn, R, K)
        elif self.decompress == "basis_pool":
            o = sum(r[j][..., None] * self.decomp(zn, w[j], pos_feat[j], q_valid, t_valid, K) for j in range(2))
        else:
            o = sum(r[j][..., None] * self.decomp(zn, w[j], K_per, K) for j in range(2))
        h = (s_pre + o) * mask[..., None]
        for lyr in self.post_layers:
            h = lyr(h, pair_post, c_seq, valid)

        out["coords_pred"] = self.coors_3d_decoder(h) * mask[..., None]
        hn = self.opm_ln(h)
        a, b = self.opm_a(hn), self.opm_b(hn)
        op = torch.einsum("bic,bjd->bijcd", a, b).reshape(B, L, L, -1)
        logits = self.pair_head(self.opm_out(op) + pair_post)
        pm = (valid[:, :, None] & valid[:, None, :]).to(logits.dtype)
        out["pair_logits"] = 0.5 * (logits + logits.transpose(1, 2)) * pm[..., None]
        return out
