"""AF3 atom-level attention for all-atom output (SI Algorithms 5 and 6).

⭐ Attention here is SEQUENCE-LOCAL and is computed in AF3's BLOCKED layout: the atoms are cut into
blocks of n_queries=32, and every query in a block attends to the same centred window of
n_keys=128. Protenix, OpenFold3 and IntelliFold all use exactly those two numbers.

⛔ The blocked layout is not an optional optimisation. The previous version of this file built the
[A, A] band as a dense bool mask and then collapsed it with `.any(dim=1)` before handing it to the
attention -- which reduces to "was this key seen by ANY query", i.e. plain atom padding. The
locality was silently discarded and the model ran FULL global attention over every atom, while
also materialising a dense [B, A, A, c_atompair] pair tensor. At L=384 that is A=5376 atoms:
28.9 M attention entries per head instead of 688 K (42x), and a 925 MB pair tensor instead of
22 MB. Measured cost of that bug: two A100s pinned at 100% for 0.65 samples/s.

⛔ SCOPE: standard amino acids only, which is what our contact model handles. AF3's atom machinery
exists mainly to represent ligands and nucleic acids; for 20 canonical residues the per-atom
reference features come straight from the repo's own residue constants, so no new data is needed.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinfoundation.nn.af3_diffusion import AdaLN, ConditionedTransitionBlock

N_QUERIES = 32   # SI Alg. 5/6; identical in Protenix, OpenFold3, IntelliFold
N_KEYS = 128


def blocked_indices(n_atom: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Query/key index tables for the blocked local layout.

    Returns (qidx [NB, N_QUERIES], kidx [NB, N_KEYS], kvalid [NB, N_KEYS], n_padded). qidx is
    exactly arange(n_padded) reshaped, so scattering block outputs back to atoms is a reshape.
    """
    nb = (n_atom + N_QUERIES - 1) // N_QUERIES
    ap = nb * N_QUERIES
    qidx = torch.arange(ap, device=device).view(nb, N_QUERIES)
    centre = torch.arange(nb, device=device) * N_QUERIES + N_QUERIES // 2
    off = torch.arange(N_KEYS, device=device) - N_KEYS // 2
    kidx = centre[:, None] + off[None, :]
    kvalid = (kidx >= 0) & (kidx < ap)
    return qidx, kidx.clamp(0, ap - 1), kvalid, ap


def gather_blocked_pair(z_pair, tok_idx, qidx, kidx):
    """[B,L,L,c] token pair -> [B,NB,Q,K,c] atom pair, via each atom's token index."""
    B = z_pair.shape[0]
    tq = tok_idx[:, qidx]                                    # [B, NB, Q]
    tk = tok_idx[:, kidx]                                    # [B, NB, K]
    bi = torch.arange(B, device=z_pair.device)[:, None, None, None]
    return z_pair[bi, tq[..., None], tk[:, :, None, :]]


def _pad_atoms(t, n_padded):
    """Right-pad the atom axis (dim 1) with zeros to n_padded."""
    pad = n_padded - t.shape[1]
    if pad <= 0:
        return t
    return F.pad(t, (0, 0) * (t.dim() - 2) + (0, pad))


class LocalAtomAttention(nn.Module):
    """Sequence-local attention with pair bias, blocked layout. SI Alg. 24 restricted to a window.

    Structurally identical to AttentionPairBias -- same projections, same adaLN conditioning, same
    -2.0 output gate -- but q/k/v are gathered into [NB, Q] / [NB, K] blocks so the softmax runs
    over N_KEYS keys instead of every atom in the chain.
    """

    def __init__(self, c_a: int, c_s: int, c_z: int, n_heads: int, bias_init: float = -2.0):
        super().__init__()
        assert c_a % n_heads == 0, (c_a, n_heads)
        self.n_heads = n_heads
        self.c_head = c_a // n_heads
        self.adaln = AdaLN(c_a, c_s)
        self.to_q = nn.Linear(c_a, c_a)
        self.to_k = nn.Linear(c_a, c_a, bias=False)
        self.to_v = nn.Linear(c_a, c_a, bias=False)
        self.norm_z = nn.LayerNorm(c_z)
        self.to_bias = nn.Linear(c_z, n_heads, bias=False)
        self.to_gate = nn.Linear(c_a, c_a, bias=False)
        self.to_out = nn.Linear(c_a, c_a, bias=False)
        self.out_scale = nn.Linear(c_s, c_a)
        nn.init.zeros_(self.out_scale.weight)
        nn.init.constant_(self.out_scale.bias, bias_init)
        # ⛔ to_out is NOT zero-initialised -- see AttentionPairBias for why (it severs z's gradient).

    def forward(self, a, s, pair, key_mask, qidx, kidx):
        """a,s [B,Ap,c]; pair [B,NB,Q,K,c_z]; key_mask [B,NB,K] bool."""
        B, Ap, _ = a.shape
        NB, Q = qidx.shape
        K = kidx.shape[1]
        H, D = self.n_heads, self.c_head

        a_n = self.adaln(a, s)
        q, k, v = self.to_q(a_n), self.to_k(a_n), self.to_v(a_n)
        qb = q[:, qidx].view(B, NB, Q, H, D).permute(0, 1, 3, 2, 4).reshape(B * NB, H, Q, D)
        kb = k[:, kidx].view(B, NB, K, H, D).permute(0, 1, 3, 2, 4).reshape(B * NB, H, K, D)
        vb = v[:, kidx].view(B, NB, K, H, D).permute(0, 1, 3, 2, 4).reshape(B * NB, H, K, D)

        bias = self.to_bias(self.norm_z(pair)).permute(0, 1, 4, 2, 3)      # [B, NB, H, Q, K]
        # A block whose every key is padding would give an all -inf softmax row and NaN out.
        # Unmask it wholesale; its queries are padding too, and the atom mask discards them below.
        km = key_mask | (~key_mask.any(-1, keepdim=True))                  # [B, NB, K]
        bias = bias.masked_fill(~km[:, :, None, None, :], torch.finfo(bias.dtype).min)

        out = F.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=bias.reshape(B * NB, H, Q, K)
        )                                                                  # [B*NB, H, Q, D]
        out = out.reshape(B, NB, H, Q, D).permute(0, 1, 3, 2, 4).reshape(B, Ap, H * D)
        out = out * torch.sigmoid(self.to_gate(a_n))
        return torch.sigmoid(self.out_scale(s)) * self.to_out(out)


class AtomTransformerBlock(nn.Module):
    """SI Alg. 7: local attention-with-pair-bias, then the same conditioned transition as Alg. 23."""

    def __init__(self, c_a: int, c_s: int, c_z: int, n_heads: int):
        super().__init__()
        self.attn = LocalAtomAttention(c_a, c_s, c_z, n_heads)
        self.transition = ConditionedTransitionBlock(c_a, c_s)

    def forward(self, a, s, pair, key_mask, qidx, kidx):
        a = a + self.attn(a, s, pair, key_mask, qidx, kidx)
        return a + self.transition(a, s)


class AtomAttentionEncoder(nn.Module):
    """Atom features (+ optional noisy coords) -> atom repr, aggregated to per-token repr. SI Alg. 5."""

    def __init__(self, c_atom: int = 128, c_atompair: int = 16, c_token: int = 768,
                 c_s: int = 384, c_z: int = 128, n_blocks: int = 3, n_heads: int = 4,
                 n_ref_feats: int = 8, has_coords: bool = True):
        super().__init__()
        self.c_atom, self.has_coords = c_atom, has_coords
        self.ref_proj = nn.Linear(n_ref_feats, c_atom)
        self.pair_proj = nn.Linear(3, c_atompair)          # from reference offset vectors
        self.s_to_atom = nn.Linear(c_s, c_atom, bias=False)
        self.z_to_atompair = nn.Linear(c_z, c_atompair, bias=False)
        if has_coords:
            self.pos_proj = nn.Linear(3, c_atom, bias=False)
        self.blocks = nn.ModuleList(
            AtomTransformerBlock(c_atom, c_atom, c_atompair, n_heads) for _ in range(n_blocks)
        )
        self.to_token = nn.Linear(c_atom, c_token, bias=False)

    def forward(self, ref_feats, ref_pos, atom_to_token, s, z, atom_mask, noisy_pos=None):
        """ref_feats [B,A,F], ref_pos [B,A,3], atom_to_token [B,A] long, s [B,L,c_s],
        z [B,L,L,c_z], atom_mask [B,A]. Returns (a_token [B,L,c_token], q_atom [B,A,c_atom])."""
        B, A, _ = ref_pos.shape
        idx = atom_to_token.clamp(min=0)
        q = self.ref_proj(ref_feats)
        # Broadcast the token-level conditioning down to the atoms that belong to each token.
        q = q + torch.gather(self.s_to_atom(s), 1, idx[..., None].expand(-1, -1, self.c_atom))
        if self.has_coords and noisy_pos is not None:
            q = q + self.pos_proj(noisy_pos)

        qidx, kidx, kvalid, ap = blocked_indices(A, ref_pos.device)
        qp = _pad_atoms(q, ap)
        rp = _pad_atoms(ref_pos, ap)
        mp = _pad_atoms(atom_mask[..., None], ap)[..., 0]
        ip = _pad_atoms(idx[..., None], ap)[..., 0]

        p = self.pair_proj(rp[:, qidx][:, :, :, None, :] - rp[:, kidx][:, :, None, :, :])
        p = p + gather_blocked_pair(self.z_to_atompair(z), ip, qidx, kidx)
        key_mask = mp.bool()[:, kidx] & kvalid[None]
        for blk in self.blocks:
            qp = blk(qp, qp, p, key_mask, qidx, kidx)
        q = qp[:, :A]
        a_atom = self.to_token(q) * atom_mask[..., None]

        # Mean-pool atoms into their token. scatter_add then divide by the count -- a plain scatter
        # would keep only the last atom of each residue.
        L = s.shape[1]
        a_token = torch.zeros(B, L, a_atom.shape[-1], device=q.device, dtype=a_atom.dtype)
        a_token.scatter_add_(1, idx[..., None].expand(-1, -1, a_atom.shape[-1]), a_atom)
        cnt = torch.zeros(B, L, 1, device=q.device, dtype=a_atom.dtype)
        cnt.scatter_add_(1, idx[..., None], atom_mask[..., None].to(a_atom.dtype))
        return a_token / cnt.clamp_min(1.0), q


class AtomAttentionDecoder(nn.Module):
    """Per-token repr -> per-atom coordinate update. SI Alg. 6."""

    def __init__(self, c_atom: int = 128, c_atompair: int = 16, c_token: int = 768,
                 n_blocks: int = 3, n_heads: int = 4):
        super().__init__()
        self.c_atom = c_atom
        self.from_token = nn.Linear(c_token, c_atom, bias=False)
        self.blocks = nn.ModuleList(
            AtomTransformerBlock(c_atom, c_atom, c_atompair, n_heads) for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(c_atom)
        # ⛔ NOT zero-initialised, and that is deliberate. A zero output projection makes the
        # backward through it `grad @ W.T = 0`, so EVERY upstream module -- the token transformer,
        # the atom encoder, the tri blocks, the contact embedding -- receives exactly zero gradient
        # until this layer itself moves off zero. Measured: 15/19 gate checks, with four sub-modules
        # at 0.000e+00 grad while only the decoder learned.
        # Protenix does not zero-init here either: its AtomAttentionDecoder.linear_no_bias_out is a
        # plain LinearNoBias (transformer.py:988), and its `zero_init` flag is reserved for
        # has_s=False attention blocks (transformer.py:94), which this is not. Standard init, and
        # the EDM output scaling c_out = sigma/sqrt(1+r^2) keeps the early update small anyway.
        self.to_pos = nn.Linear(c_atom, 3, bias=False)

    def forward(self, a_token, q_atom, atom_to_token, atom_mask, z_atompair):
        """z_atompair [B,L,L,c_atompair]: the token pair already projected to the atompair width.
        Blocked here rather than densified to [B,A,A,c] by the caller."""
        B, A = atom_mask.shape
        idx = atom_to_token.clamp(min=0)
        q = q_atom + torch.gather(
            self.from_token(a_token), 1, idx[..., None].expand(-1, -1, self.c_atom)
        )
        qidx, kidx, kvalid, ap = blocked_indices(A, atom_mask.device)
        qp = _pad_atoms(q, ap)
        mp = _pad_atoms(atom_mask[..., None], ap)[..., 0]
        ip = _pad_atoms(idx[..., None], ap)[..., 0]

        pair = gather_blocked_pair(z_atompair, ip, qidx, kidx)
        key_mask = mp.bool()[:, kidx] & kvalid[None]
        for blk in self.blocks:
            qp = blk(qp, qp, pair, key_mask, qidx, kidx)
        return self.to_pos(self.norm(qp[:, :A])) * atom_mask[..., None]
