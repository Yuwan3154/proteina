"""Unit test for the CA sin-dihedral chirality loss.

The whole premise is that `sin(dihedral)` is ODD under reflection while distances are invariant.
If that fails, the term supervises nothing and the experiment is meaningless -- so it is tested
rather than assumed.

⛔ Tests the INVARIANT (reflection flips the sign), never an absolute sign convention: the
handedness sign depends on atom ordering, and ca_handedness_filter.py's own header records that a
previous comment about that sign was wrong while the code was right.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer as T


def helix(n=40, radius=2.3, rise=1.5, turn_deg=100.0):
    i = torch.arange(n, dtype=torch.float64)
    a = torch.deg2rad(torch.tensor(turn_deg, dtype=torch.float64)) * i
    return torch.stack([radius * torch.cos(a), radius * torch.sin(a), rise * i], dim=-1)


def pack14(ca):
    """Put CA into atom14 slot 1, the layout x_denoised actually uses."""
    b, n, _ = ca.shape
    x = torch.zeros(b, n, 14, 3, dtype=ca.dtype)
    x[:, :, 1, :] = ca
    return x.reshape(b, n * 14, 3)


ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


ca = helix()[None]                      # [1, n, 3] right-handed by construction
ca_mirror = ca * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)   # reflect in z

s_r = T._ca_sin_dihedral(ca)
s_m = T._ca_sin_dihedral(ca_mirror)

print("== A. sin(dihedral) is ODD under reflection ==")
check("reflected == -original", torch.allclose(s_m, -s_r, atol=1e-10),
      f"max|s_m+s_r| = {(s_m + s_r).abs().max():.2e}")
check("signal is non-trivial (|sin| well away from 0)", s_r.abs().mean() > 0.5,
      f"mean|sin| = {s_r.abs().mean():.4f}")
check("a helix has ONE consistent handedness", (s_r.sign() == s_r.sign()[0, 0]).all(),
      f"sign = {int(s_r.sign()[0, 0])}")

print("\n== B. distances are INVARIANT under the same reflection (so contacts cannot see it) ==")
d_r = torch.cdist(ca[0], ca[0])
d_m = torch.cdist(ca_mirror[0], ca_mirror[0])
check("cdist identical", torch.allclose(d_r, d_m, atol=1e-10),
      f"max|dd| = {(d_r - d_m).abs().max():.2e}")

print("\n== C. the loss itself ==")
n = ca.shape[1]
m = torch.ones(1, n * 14)
mod = object.__new__(T)          # no nn.Module init needed for a staticmethod + pure-tensor method
l_same, _ = T._chirality_loss(mod, pack14(ca), pack14(ca), m, n)
l_mirror, (sp, st, wm) = T._chirality_loss(mod, pack14(ca_mirror), pack14(ca), m, n)
check("loss(x, x) == 0", float(l_same.max()) < 1e-12, f"{float(l_same.max()):.2e}")
check("loss(mirror, x) is large", float(l_mirror.min()) > 1.0, f"{float(l_mirror.mean()):.4f}")
check("sign-wrong readout == 1.0 for a full mirror",
      abs(float((((sp * st) < 0) & wm).sum() / wm.sum()) - 1.0) < 1e-9)

print("\n== D. masking: padded residues must not contribute ==")
ca_pad = torch.cat([ca, torch.zeros(1, 10, 3, dtype=ca.dtype)], dim=1)
n_pad = ca_pad.shape[1]
m_pad = torch.zeros(1, n_pad * 14)
m_pad.reshape(1, n_pad, 14)[:, :n, 1] = 1.0
l_pad, (_, _, wm_pad) = T._chirality_loss(mod, pack14(ca_pad), pack14(ca_pad), m_pad, n_pad)
check("padded loss still 0", float(l_pad.max()) < 1e-12)
check("window count excludes padding", int(wm_pad.sum()) == n - 3,
      f"{int(wm_pad.sum())} windows, expected {n - 3}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
