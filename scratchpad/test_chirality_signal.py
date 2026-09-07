"""Does anything in the c2c loss actually SEE a mirrored fold?

Generation puts 44% of samples out as near-perfect reflections (mirror_rmsd 1.2-2.3 A) whose
per-residue stereocentres are nonetheless correct (chirality_agree 0.998). So the failure is a
GLOBAL handedness error, and the question is which loss terms can penalise it.

⛔ The distogram, smooth-lDDT and distance-MAE terms are all functions of pairwise DISTANCES, and a
reflection preserves every pairwise distance exactly. They are mathematically incapable of seeing
this. The det=+1 Kabsch MSE is the only candidate -- so if its reflection guard silently stopped
firing, the mirror would be invisible to the entire objective and the 44% would be fully explained.

These tests pin that down numerically instead of by reading the code.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import smooth_lddt, weighted_rigid_align

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def rand_chain(b=2, n=64):
    """A self-avoiding-ish random walk: a straight line would be its own mirror image."""
    step = torch.randn(b, n, 3)
    step = step / step.norm(dim=-1, keepdim=True) * 3.81
    return torch.cumsum(step, dim=1)


def mirror(x):
    m = x.clone()
    m[..., 2] = -m[..., 2]
    return m


def rand_rotation(b):
    q, _ = torch.linalg.qr(torch.randn(b, 3, 3))
    det = torch.linalg.det(q)
    q[:, :, 2] = q[:, :, 2] * det[:, None]      # force det=+1, a PROPER rotation
    return q


print("\n=== 1. the reflection guard in weighted_rigid_align ===")
x_gt = rand_chain()
b, n, _ = x_gt.shape
w = torch.ones(b, n)
mask = torch.ones(b, n)

rot = rand_rotation(b)
x_rot = torch.einsum("bni,bij->bnj", x_gt, rot) + torch.randn(b, 1, 3) * 5.0
aligned = weighted_rigid_align(x_rot, x_gt, w, mask)
rmsd_rot = (aligned - x_rot).pow(2).sum(-1).mean().sqrt()
check("proper rotation is undone (rmsd ~ 0)", rmsd_rot < 1e-3, f"rmsd={rmsd_rot:.2e}")

x_mir = mirror(x_gt)
aligned_m = weighted_rigid_align(x_mir, x_gt, w, mask)
rmsd_mir = (aligned_m - x_mir).pow(2).sum(-1).mean().sqrt()
check("MIRROR is NOT undone (rmsd stays large)", rmsd_mir > 1.0, f"rmsd={rmsd_mir:.3f}")

r = torch.einsum("bni,bij->bnj", x_gt - x_gt.mean(1, keepdim=True), rand_rotation(b))
det_ok = True
for _ in range(20):
    a = weighted_rigid_align(mirror(rand_chain()), rand_chain(), w, mask)
    det_ok &= torch.isfinite(a).all().item()
check("guard is numerically stable over 20 random mirrors", det_ok)

print("\n=== 2. which loss terms can see the mirror at all ===")
d_gt = torch.cdist(x_gt, x_gt)
d_mir = torch.cdist(x_mir, x_mir)
max_dd = (d_gt - d_mir).abs().max()
check("pairwise distances are IDENTICAL under reflection", max_dd < 1e-4, f"max|dd|={max_dd:.2e}")

l_self = smooth_lddt(x_gt, x_gt, mask)
l_mir = smooth_lddt(x_mir, x_gt, mask)
check("smooth_lddt is BLIND to the mirror", (l_self - l_mir).abs().max() < 1e-4,
      f"lddt_loss self={l_self.mean():.6f} mirror={l_mir.mean():.6f}")

mse_mir = (weighted_rigid_align(x_mir, x_gt, w, mask) - x_mir).pow(2).sum(-1).mean()
mse_rot = (weighted_rigid_align(x_rot, x_gt, w, mask) - x_rot).pow(2).sum(-1).mean()
check("aligned MSE SEES the mirror (mirror >> rotation)", mse_mir > 100 * mse_rot.clamp_min(1e-12),
      f"mse mirror={mse_mir:.4f} rotation={mse_rot:.2e}")

print("\n=== 3. the signal is there, but how strong per noise draw? ===")
# A mirrored structure and the true one are equally consistent with the contact map, so the ONLY
# thing separating them is this MSE, averaged over n_diff noise draws. Report the separation.
for nd in (1, 8, 16, 48):
    errs = []
    for _ in range(nd):
        noise = torch.randn_like(x_gt) * 2.0
        good = (weighted_rigid_align(x_gt + noise, x_gt, w, mask) - (x_gt + noise)).pow(2).sum(-1).mean()
        bad = (weighted_rigid_align(x_mir + noise, x_gt, w, mask) - (x_mir + noise)).pow(2).sum(-1).mean()
        errs.append((bad - good).item())
    m = sum(errs) / len(errs)
    sd = (sum((e - m) ** 2 for e in errs) / len(errs)) ** 0.5
    print(f"    n_diff={nd:>2d}  mean margin(bad-good)={m:8.3f}  sd={sd:7.3f}  "
          f"sd/sqrt(n)={sd / (nd ** 0.5):7.3f}")
print("    ^ the margin is what pushes away from the mirror; sd/sqrt(n) is its noise floor.")
print("      Raising n_diff shrinks the floor as 1/sqrt(n) -- that is the whole mechanism of fix B.")

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
