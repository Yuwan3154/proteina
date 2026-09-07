"""Gate for AF3 SI Alg. 19 CentreRandomAugmentation.

⛔⛔ THE CRITICAL TEST IS CHIRALITY. If this augmentation ever emits an improper rotation it would
mirror that fraction of the training targets and MANUFACTURE the exact 50/50 failure we are trying
to fix -- while looking like a harmless data-pipeline change. A QR of a Gaussian gives O(3), det=-1
half the time; the Gram-Schmidt construction used here is det=+1 by design, and this measures it
over 20000 draws rather than trusting the construction.

Also pins the two bugs it fixes: no centring (measured mean |centroid| 61 A, max 222 A) and
replicas sharing one orientation.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.contact2coord import centre_random_augmentation

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


torch.manual_seed(0)
B, A, N = 3, 200, 8
# A chiral test object far from the origin, mimicking deposited PDB coordinates.
x = torch.randn(B, A, 3) * 12.0 + torch.tensor([60.0, -40.0, 25.0])
mask = torch.ones(B, A)
mask[1, -60:] = 0.0                       # padding, which the COM must ignore

out = centre_random_augmentation(x, mask, n_sample=N)

print("\n=== shape and masking ===")
check("output is [B, n, A, 3]", tuple(out.shape) == (B, N, A, 3), str(tuple(out.shape)))
check("padded atoms stay exactly zero",
      out[1, :, -60:].abs().max().item() == 0.0,
      f"max {out[1, :, -60:].abs().max().item():.2e}")

print("\n=== centring (the 61 A offset bug) ===")
m = mask[:, None, :, None]
com = (out * m).sum(2) / m.sum(2).clamp_min(1e-8)          # [B, n, 3]
# After centring the only offset left is the s_trans=1.0 A translation.
check("masked COM is within a few A of the origin (was 61 A, max 222 A)",
      com.norm(dim=-1).max().item() < 6.0,
      f"max |COM| {com.norm(dim=-1).max().item():.2f} A")
check("COM spread is consistent with s_trans=1.0, not with the raw offset",
      0.2 < com.norm(dim=-1).mean().item() < 4.0,
      f"mean |COM| {com.norm(dim=-1).mean().item():.2f} A")

print("\n=== rigid motion: distances must be preserved exactly ===")
keep = mask[0].bool()
d0 = torch.cdist(x[0][keep], x[0][keep])
worst = max((torch.cdist(out[0, j][keep], out[0, j][keep]) - d0).abs().max().item()
            for j in range(N))
check("all pairwise distances preserved", worst < 1e-3, f"max |dd| {worst:.2e}")

print("\n=== ⛔ CHIRALITY: the augmentation must NEVER reflect ===")
def signed_vol(p):
    return torch.einsum("ni,ni->n", torch.cross(p[1::4] - p[0::4], p[2::4] - p[0::4], dim=-1),
                        p[3::4] - p[0::4])
v0 = signed_vol(x[0][keep])
flips = 0
for j in range(N):
    vj = signed_vol(out[0, j][keep])
    flips += int((torch.sign(vj) != torch.sign(v0)).sum())
check("no signed volume changes sign across any replica", flips == 0, f"{flips} sign flips")

# Measure the determinant directly over many draws -- the decisive check.
big = centre_random_augmentation(torch.randn(1, 4, 3), torch.ones(1, 4), n_sample=20000)
p = big[0]                                                   # [20000, 4, 3]
e = p[:, 1:] - p[:, :1]
det = torch.linalg.det(e)
v_ref = torch.linalg.det((torch.randn(1, 4, 3)[0, 1:] - torch.randn(1, 4, 3)[0, :1]))
n_improper = int((torch.sign(det) != torch.sign(det[0])).sum())
check("all 20000 replicas share one handedness (no improper rotations)",
      n_improper == 0, f"{n_improper}/20000 differ")

print("\n=== replicas must be INDEPENDENT (the shared-orientation bug) ===")
pair = (out[0, 0] - out[0, 1]).abs().max().item()
check("replica 0 and replica 1 differ", pair > 1.0, f"max |diff| {pair:.2f} A")
coms = com[0]
check("each replica gets its own translation",
      (coms[0] - coms[1]).norm().item() > 1e-3)

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
