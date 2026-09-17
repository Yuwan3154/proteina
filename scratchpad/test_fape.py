"""Backbone FAPE: does it actually do the job it is being added for?

⭐⭐ The one property that matters here is C: FAPE must SEPARATE a structure from its MIRROR, which
no distance-only term can. Everything else (A, B, D, E) exists so that a FAPE which is subtly wrong
-- transposed rotation, wrong einsum, mask ignored -- cannot pass while looking plausible.

⛔ Exercises the REAL methods off ContactToCoordTrainer, bound onto a stub so the 200M-parameter
model is never built. Testing a retyped copy would prove nothing about the shipped code.

Run: python scratchpad/test_fape.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)          # exact-zero checks need more than fp32

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


class Stub:
    """Real code, no model. Binds the unbound functions straight off the class."""
    fape_chunk = 0
    _frames_from_backbone = staticmethod(ContactToCoordTrainer._frames_from_backbone)
    _fape_pairs = ContactToCoordTrainer._fape_pairs
    _fape_loss = ContactToCoordTrainer._fape_loss


def helix(L, rise=1.5, radius=2.3, turn=100.0):
    """Idealised CA helix with N and C placed off the local frame, so frames are well conditioned."""
    t = torch.arange(L, dtype=torch.get_default_dtype())
    a = t * math.radians(turn)
    ca = torch.stack([radius * torch.cos(a), radius * torch.sin(a), rise * t], dim=-1)
    tang = torch.zeros_like(ca)
    tang[:-1] = ca[1:] - ca[:-1]
    tang[-1] = tang[-2]
    tang = tang / tang.norm(dim=-1, keepdim=True)
    radial = ca - torch.stack([torch.zeros_like(t), torch.zeros_like(t), rise * t], dim=-1)
    radial = radial / radial.norm(dim=-1, keepdim=True)
    n = ca - 1.458 * tang                      # N behind along the chain
    cdir = 0.6 * tang + 0.8 * radial           # C off-axis so N/CA/C are not collinear
    c = ca + 1.525 * cdir / cdir.norm(dim=-1, keepdim=True)
    x = torch.zeros(L, 14, 3)
    x[:, 0], x[:, 1], x[:, 2] = n, ca, c
    return x


L = 24
base = helix(L)
xt = base[None].clone()                         # [S=1, L, 14, 3]
m = torch.zeros(1, L, 14)
m[:, :, 0:3] = 1.0


def fape(pred, true, mask, chunk=0):
    s = Stub()
    s.fape_chunk = chunk
    return ContactToCoordTrainer._fape_loss(s, pred, true, mask, L)


def rand_rot():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]                      # force a PROPER rotation
    return q


# ═══ A. frames are proper rotations ═══════════════════════════════════════════════════════════
print("== A. frame construction ==")
R, t = ContactToCoordTrainer._frames_from_backbone(xt[:, :, 0], xt[:, :, 1], xt[:, :, 2])
det = torch.det(R)
check("R is orthonormal", torch.allclose(R.transpose(-1, -2) @ R,
                                         torch.eye(3).expand_as(R), atol=1e-8))
check("det(R) = +1 everywhere (PROPER rotation)", torch.allclose(det, torch.ones_like(det), atol=1e-8),
      f"min {det.min():.6f} max {det.max():.6f}")
check("translation is CA", torch.allclose(t, xt[:, :, 1]))

# ═══ B. identity and invariance ═══════════════════════════════════════════════════════════════
print("\n== B. identity, and SE(3) invariance (FAPE's defining property) ==")
f_id = fape(xt.clone(), xt, m)
check("FAPE(x, x) == 0", float(f_id) < 1e-12, f"{float(f_id):.3e}")

Rg, tg = rand_rot(), torch.randn(3) * 7.0
xr = xt @ Rg.T + tg
f_rot = fape(xr, xt, m)
check("FAPE invariant to global rotation + translation", float(f_rot) < 1e-12, f"{float(f_rot):.3e}")

# ═══ C. ⭐ THE POINT: mirrors must NOT be invariant ════════════════════════════════════════════
print("\n== C. the mirror, which distance-only terms cannot see ==")
refl = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
xm = xt @ refl.T
f_mir = fape(xm, xt, m)
check("FAPE(mirror, x) is LARGE", float(f_mir) > 0.1, f"{float(f_mir):.4f}")
check("mirror >> identity by orders of magnitude", float(f_mir) > 1e6 * max(float(f_id), 1e-18))

# the achiral control: pairwise distances cannot tell them apart AT ALL
d_t = torch.cdist(xt[0, :, 1], xt[0, :, 1])
d_m = torch.cdist(xm[0, :, 1], xm[0, :, 1])
check("CONTROL: pairwise distances are IDENTICAL under reflection",
      torch.allclose(d_t, d_m, atol=1e-10), f"max|dD| = {float((d_t - d_m).abs().max()):.2e}")
# and a mirror composed with a rotation is still caught -- the usual way a wrong impl passes
xmr = (xt @ refl.T) @ rand_rot().T + torch.randn(3) * 3
check("mirror + rotation + translation is STILL caught",
      float(fape(xmr, xt, m)) > 0.1, f"{float(fape(xmr, xt, m)):.4f}")

# ═══ D. chunking must not change the answer (default-OFF branch) ══════════════════════════════
print("\n== D. chunking over the sample axis ==")
S = 5
xt_s = xt.repeat(S, 1, 1, 1)
m_s = m.repeat(S, 1, 1)
xp_s = xt_s + torch.randn_like(xt_s) * 0.4
f_one = fape(xp_s, xt_s, m_s, chunk=0)
for ck in (1, 2, 5, 99):
    f_ch = fape(xp_s, xt_s, m_s, chunk=ck)
    check(f"chunk={ck} matches one-shot", torch.allclose(f_one, f_ch, atol=1e-12),
          f"{float(f_one):.10f} vs {float(f_ch):.10f}")

# ═══ E. masking ═══════════════════════════════════════════════════════════════════════════════
print("\n== E. masking (padding must not contribute) ==")
m_half = m.clone()
m_half[:, L // 2:, :] = 0.0
xt_pad = xt.clone()
xp_pad = xt.clone()
xp_pad[:, L // 2:] += 500.0                      # garbage in the padded region
f_masked = fape(xp_pad, xt_pad, m_half, 0)
check("garbage in masked residues does not leak into the loss", float(f_masked) < 1e-12,
      f"{float(f_masked):.3e}")
f_unmasked = fape(xp_pad, xt_pad, m, 0)
check("...and WOULD have, without the mask", float(f_unmasked) > 0.1, f"{float(f_unmasked):.4f}")

# ═══ F. degenerate geometry ═══════════════════════════════════════════════════════════════════
print("\n== F. degenerate input is finite ==")
xd = torch.zeros(1, L, 14, 3)                    # every atom coincident
check("coincident backbone -> finite", torch.isfinite(fape(xd, xt, m)).all())
check("coincident on BOTH sides -> finite", torch.isfinite(fape(xd, xd, m)).all())

# ═══ G. monotone in error, and the clamp ══════════════════════════════════════════════════════
print("\n== G. behaviour ==")
prev = -1.0
for s in (0.0, 0.1, 0.3, 1.0, 3.0):
    v = float(fape(xt + torch.randn_like(xt) * s, xt, m))
    print(f"    noise {s:>4} A -> FAPE {v:.4f}")
    check(f"non-decreasing at noise {s}", v >= prev - 1e-9)
    prev = v
check("clamped: FAPE cannot exceed clamp/z = 1.0",
      float(fape(xt + torch.randn_like(xt) * 500.0, xt, m)) <= 1.0 + 1e-9)

print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
