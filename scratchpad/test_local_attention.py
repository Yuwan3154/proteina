"""Gate for the blocked sequence-local atom attention.

The bug this replaces ran GLOBAL attention while claiming to be local, so the decisive test is not
"does it run" but "is a query provably unaffected by atoms outside its key window". Three gates:

  1. EQUIVALENCE -- the blocked kernel matches an explicit per-block reference (plain softmax, no
     reshapes) to float tolerance. This checks the permute/reshape gymnastics and the bias/mask
     placement independently of the fast path.
  2. LOCALITY -- perturbing an atom outside a query's window changes that query's output by
     exactly 0. This is the gate the old implementation would have failed.
  3. MASKING -- an all-padding block produces finite output, not NaN from an all -inf softmax row.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.atom_attention import (
    N_KEYS,
    N_QUERIES,
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    LocalAtomAttention,
    blocked_indices,
)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def reference_local_attn(mod, a, s, pair, key_mask, qidx, kidx):
    """Explicit per-block attention: no reshapes, plain softmax, float64."""
    B, Ap, _ = a.shape
    NB, Q = qidx.shape
    K = kidx.shape[1]
    H, D = mod.n_heads, mod.c_head
    a_n = mod.adaln(a, s)
    q, k, v = mod.to_q(a_n), mod.to_k(a_n), mod.to_v(a_n)
    bias = mod.to_bias(mod.norm_z(pair))                      # [B, NB, Q, K, H]
    out = torch.zeros(B, Ap, H * D, dtype=torch.float64)
    for b in range(B):
        for nb in range(NB):
            km = key_mask[b, nb]
            if not km.any():
                km = torch.ones_like(km)
            for h in range(H):
                qh = q[b, qidx[nb], h * D:(h + 1) * D].double()          # [Q, D]
                kh = k[b, kidx[nb], h * D:(h + 1) * D].double()          # [K, D]
                vh = v[b, kidx[nb], h * D:(h + 1) * D].double()
                sc = qh @ kh.T / (D ** 0.5) + bias[b, nb, :, :, h].double()
                sc = sc.masked_fill(~km[None, :], float("-inf"))
                out[b, qidx[nb], h * D:(h + 1) * D] = torch.softmax(sc, -1) @ vh
    out = out.to(a.dtype) * torch.sigmoid(mod.to_gate(a_n))
    return torch.sigmoid(mod.out_scale(s)) * mod.to_out(out)


def main():
    torch.manual_seed(0)
    dev = "cpu"
    B, A, c_a, c_z, H = 2, 200, 32, 8, 4
    qidx, kidx, kvalid, ap = blocked_indices(A, dev)
    NB = qidx.shape[0]
    print(f"A={A} -> padded {ap}, {NB} blocks of {N_QUERIES} queries x {N_KEYS} keys")

    mod = LocalAtomAttention(c_a, c_a, c_z, H).double()
    a = torch.randn(B, ap, c_a, dtype=torch.float64)
    pair = torch.randn(B, NB, N_QUERIES, N_KEYS, c_z, dtype=torch.float64)
    amask = torch.zeros(B, ap, dtype=torch.bool)
    amask[0, :A] = True
    amask[1, :A - 37] = True
    key_mask = amask[:, kidx] & kvalid[None]

    with torch.no_grad():
        fast = mod(a, a, pair, key_mask, qidx, kidx)
        ref = reference_local_attn(mod, a, a, pair, key_mask, qidx, kidx)
    d = (fast - ref).abs().max().item()
    check("blocked kernel == explicit per-block reference", d < 1e-9, f"max|diff|={d:.2e}")

    # ---- locality: an atom far outside block 0's window must not move block 0's output ----
    far = ap - 1                       # last atom; block 0's window is keys 0..79
    in_window = set(kidx[0][kvalid[0]].tolist())
    check("far atom really is outside block 0's window", far not in in_window,
          f"window={min(in_window)}..{max(in_window)}")
    a2 = a.clone()
    a2[:, far] += 10.0
    with torch.no_grad():
        moved = mod(a2, a2, pair, key_mask, qidx, kidx)
    delta_blk0 = (moved[:, :N_QUERIES] - fast[:, :N_QUERIES]).abs().max().item()
    delta_far = (moved[:, far] - fast[:, far]).abs().max().item()
    check("perturbing a far atom leaves block 0 EXACTLY unchanged", delta_blk0 == 0.0,
          f"delta={delta_blk0:.2e}")
    check("...while the perturbed atom's own output does change", delta_far > 1e-6,
          f"delta={delta_far:.2e}")

    # ---- all-padding block must not NaN ----
    empty = torch.zeros(1, ap, dtype=torch.bool)
    empty[0, :N_QUERIES] = True        # only block 0 has real atoms
    km2 = empty[:, kidx] & kvalid[None]
    check("some block is entirely padding", (~km2.any(-1)).any().item())
    with torch.no_grad():
        o = mod(a[:1], a[:1], pair[:1], km2, qidx, kidx)
    check("all-padding block gives finite output (no -inf softmax NaN)",
          bool(torch.isfinite(o).all()))

    # ---- encoder / decoder end to end ----
    L, c_s, c_tok, c_ap = 20, 16, 24, 8
    A2 = L * 14
    enc = AtomAttentionEncoder(c_atom=c_a, c_atompair=c_ap, c_token=c_tok, c_s=c_s, c_z=c_z,
                               n_blocks=2, n_heads=H, n_ref_feats=8)
    dec = AtomAttentionDecoder(c_atom=c_a, c_atompair=c_ap, c_token=c_tok, n_blocks=2, n_heads=H)
    a2t = torch.arange(L).repeat_interleave(14)[None].expand(B, -1).contiguous()
    am = torch.ones(B, A2)
    am[1, -20:] = 0.0
    s = torch.randn(B, L, c_s)
    z = torch.randn(B, L, L, c_z)
    a_tok, q_atom = enc(torch.randn(B, A2, 8), torch.randn(B, A2, 3), a2t, s, z, am,
                        noisy_pos=torch.randn(B, A2, 3))
    check("encoder token shape", tuple(a_tok.shape) == (B, L, c_tok), str(tuple(a_tok.shape)))
    check("encoder atom shape", tuple(q_atom.shape) == (B, A2, c_a), str(tuple(q_atom.shape)))
    upd = dec(a_tok, q_atom, a2t, am, torch.randn(B, L, L, c_ap))
    check("decoder coord shape", tuple(upd.shape) == (B, A2, 3), str(tuple(upd.shape)))
    check("decoder output finite", bool(torch.isfinite(upd).all()))
    check("decoder zeroes masked atoms", float(upd[1, -20:].abs().max()) == 0.0)

    # ---- gradients reach every sub-module ----
    upd.sum().backward()
    dead = [n for n, p in list(enc.named_parameters()) + list(dec.named_parameters())
            if p.grad is None or p.grad.abs().max() == 0]
    check("every encoder+decoder parameter receives gradient", not dead, str(dead[:4]))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
