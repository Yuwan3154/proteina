"""Does QK-norm actually bound the attention logits, and is the rest of the block untouched?

The instability QK-norm targets is attention-logit growth: during training ||W_q||, ||W_k|| drift
upward, the logits q·k/sqrt(d) grow, softmax saturates toward one-hot, and gradient through the
attention collapses. The block then stops learning and degrades -- which matches our symptom
(diffusion module degrades while the trunk stays healthy).

⛔ Scaling the INPUT would not test this. AdaLN starts with `norm_a`, a LayerNorm with
elementwise_affine=False, so any input scale is removed before the projections. The growth that
matters is in the WEIGHTS, so that is what this scales.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import AttentionPairBias
from proteinfoundation.nn.atom_attention import LocalAtomAttention, blocked_indices

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def token_logits(mod, a, s, scale):
    """Max |logit| for AttentionPairBias with its q/k projection weights scaled by `scale`."""
    B, L, _ = a.shape
    H, D = mod.n_heads, mod.c_head
    with torch.no_grad():
        m = type(mod)(a.shape[-1], s.shape[-1], 8, H,
                      qk_norm=not isinstance(mod.q_norm, torch.nn.Identity))
        m.load_state_dict(mod.state_dict())
        m.to_q.weight.mul_(scale)
        m.to_k.weight.mul_(scale)
        a_n = m.adaln(a, s)
        q = m.q_norm(m.to_q(a_n).view(B, L, H, D)).transpose(1, 2)
        k = m.k_norm(m.to_k(a_n).view(B, L, H, D)).transpose(1, 2)
        return float((q @ k.transpose(-1, -2) / D ** 0.5).abs().max())


def main():
    torch.manual_seed(0)
    B, L, c_a, c_s, c_z, H = 2, 24, 32, 16, 8, 4
    a = torch.randn(B, L, c_a)
    s = torch.randn(B, L, c_s)

    on = AttentionPairBias(c_a, c_s, c_z, H, qk_norm=True)
    off = AttentionPairBias(c_a, c_s, c_z, H, qk_norm=False)
    off.load_state_dict({k: v for k, v in on.state_dict().items()
                         if not k.startswith(("q_norm", "k_norm"))}, strict=False)

    print("  max|logit| as the q/k projection weights grow:")
    print(f"  {'weight scale':>13} {'qk_norm OFF':>13} {'qk_norm ON':>12}")
    rows = []
    for sc in [1.0, 4.0, 16.0, 64.0]:
        lo, ln = token_logits(off, a, s, sc), token_logits(on, a, s, sc)
        rows.append((sc, lo, ln))
        print(f"  {sc:>13.0f} {lo:>13.2f} {ln:>12.2f}")

    growth_off = rows[-1][1] / max(rows[0][1], 1e-9)
    growth_on = rows[-1][2] / max(rows[0][2], 1e-9)
    check("without QK-norm the logits grow with the weights",
          growth_off > 100, f"x{growth_off:.0f} over a 64x weight scale")
    check("with QK-norm the logits are essentially INVARIANT to weight scale",
          growth_on < 1.05, f"x{growth_on:.3f}")
    check("QK-norm keeps logits in a sane range even at 64x",
          rows[-1][2] < 50.0, f"max|logit|={rows[-1][2]:.2f}")

    # The block must still work and still train.
    z = torch.randn(B, L, L, c_z)
    mask = torch.ones(B, L)
    out = on(a, s, z, mask)
    check("output shape unchanged", tuple(out.shape) == (B, L, c_a), str(tuple(out.shape)))
    check("output finite", bool(torch.isfinite(out).all()))
    out.sum().backward()
    dead = [n for n, p in on.named_parameters()
            if p.grad is None or p.grad.abs().max() == 0]
    expected = [n for n in dead if n.endswith(("adaln.norm_s.weight", "adaln.norm_s.bias"))]
    check("only the known adaLN-zero params are zero-grad", sorted(dead) == sorted(expected),
          str([n for n in dead if n not in expected][:4]))
    check("q_norm/k_norm receive gradient",
          all(getattr(on, m).weight.grad.abs().max() > 0 for m in ("q_norm", "k_norm")))

    # Atom track.
    qidx, kidx, kvalid, ap = blocked_indices(64, "cpu")
    NB = qidx.shape[0]
    la = LocalAtomAttention(c_a, c_a, c_z, H, qk_norm=True)
    aa = torch.randn(B, ap, c_a)
    pair = torch.randn(B, NB, qidx.shape[1], kidx.shape[1], c_z)
    km = torch.ones(B, NB, kidx.shape[1], dtype=torch.bool) & kvalid[None]
    o2 = la(aa, aa, pair, km, qidx, kidx)
    check("atom-track block runs with QK-norm", tuple(o2.shape) == (B, ap, c_a),
          str(tuple(o2.shape)))
    check("atom-track output finite", bool(torch.isfinite(o2).all()))
    o2.sum().backward()
    check("atom-track q_norm/k_norm receive gradient",
          all(getattr(la, m).weight.grad.abs().max() > 0 for m in ("q_norm", "k_norm")))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
