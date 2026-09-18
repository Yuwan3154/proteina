"""Where does FAPE's mirror signal actually live, and does AF2's 10 A clamp kill it?"""
import math

import torch

torch.set_default_dtype(torch.float64)
exec(open('/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/scratchpad/'
          'fape_mirror_test.py').read().split('L = 30')[0])

# compact multi-segment fold: helix - loop - helix - loop - strand - loop - helix
seg = [('H', 14), ('L', 4), ('H', 14), ('L', 4), ('E', 10), ('L', 4), ('H', 14)]
TOR = {'H': (-57., -47.), 'E': (-135., 135.), 'L': (-70., 140.)}
phis, psis = [], []
for kind, n in seg:
    p, q = TOR[kind]
    phis += [p] * n
    psis += [q] * n
L = len(phis)
NAT = build(phis, psis, [180.] * (L - 1))
M = torch.diag(torch.tensor([1., 1., -1.]))
MIRb = NAT @ M
# physically-realisable c2c-style mirror: negate every torsion, keep L bond geometry
MIRt = build([-p for p in phis], [-q for q in psis], [-180.] * (L - 1))


def frames(bb):
    return from_3_points(bb[:, 0], bb[:, 1], bb[:, 2])


Rn, tn = frames(NAT)
Rb, tb = frames(MIRb)
Rt_, tt_ = frames(MIRt)
xn, xb, xt = NAT.reshape(-1, 3), MIRb.reshape(-1, 3), MIRt.reshape(-1, 3)

g = kabsch_align_gt_to_pred(xt, xb)
print(f"L={L} residues. Compact fold: helix/loop/helix/loop/strand/loop/helix")
print(f"torsion-negated chain vs reflected chain, RMSD = "
      f"{torch.sqrt((((xt - xt.mean(0)) - g) ** 2).sum(-1).mean()):.6f} A "
      "(0 => sign-flipping torsions IS a reflection of the backbone)")
print(f"Rg = {torch.sqrt(((xn - xn.mean(0)) ** 2).sum(-1).mean()):.2f} A")

print()
print("=" * 84)
print("FAPE(mirror vs native), length_scale=10 A (openfold config.py:747,752)")
for cl, lbl in [(None, 'unclamped'), (10.0, 'clamped @10 A (AF2 default)')]:
    f, lp, lt = fape(Rb, tb, Rn, tn, xb, xn, clamp=cl)
    fs, _, _ = fape(Rn, tn, Rn, tn, xn, xn, clamp=cl)
    print(f"  {lbl:30s}  mirror={f:8.5f}   self={fs:.6f}")

print()
print("Decomposition by sequence separation |i-j| of (frame i, atom of residue j).")
print("The 10 A clamp kills long-range pairs; does ANY signal survive at short range?")
_, lp, lt = fape(Rb, tb, Rn, tn, xb, xn, clamp=None)
err = torch.sqrt(((lp - lt) ** 2).sum(-1) + 1e-8)        # [F, 3L]
res_of_atom = torch.arange(L).repeat_interleave(3)
sep = (torch.arange(L)[:, None] - res_of_atom[None, :]).abs()
print(f"  {'|i-j|':>10s} {'n_pairs':>9s} {'mean err A':>11s} {'mean clamped':>13s} {'%clamped':>9s}")
for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 9)]:
    m = (sep >= lo) & (sep <= hi)
    if m.sum() == 0:
        continue
    e = err[m]
    print(f"  {f'{lo}-{hi if hi < 10**9 else L}':>10s} {int(m.sum()):9d} {e.mean():11.4f} "
          f"{e.clamp(0, 10).mean():13.4f} {100 * (e > 10).double().mean():8.1f}%")

print()
print("=" * 84)
print("Same decomposition, but for the ONLY structure the diffusion loss ever sees:")
print("a NOISED native (sigma sweep). If FAPE-vs-native is already ~0 there, a FAPE")
print("term adds no branch-selecting gradient anywhere in training.")
gen = torch.Generator().manual_seed(0)
SIGMA_DATA = 16.0
xc = xn - xn.mean(0)
for sigma in [0.5, 2.0, 8.0, 16.0, 40.0, 160.0]:
    noisy = xc + sigma * torch.randn(xc.shape, generator=gen)
    bb = noisy.reshape(L, 3, 3)
    Rq, tq = from_3_points(bb[:, 0], bb[:, 1], bb[:, 2])
    f, _, _ = fape(Rq, tq, Rn, tn, noisy, xn, clamp=10.0)
    # is the global hand still readable from the noised sample?
    ca = noisy.reshape(L, 3, 3)[:, 1]
    v1, v2, v3 = ca[1:-2] - ca[:-3], ca[2:-1] - ca[1:-2], ca[3:] - ca[2:-1]
    dih_sign = torch.sign((torch.cross(v1, v2, dim=-1) * v3).sum(-1))
    ca0 = xc.reshape(L, 3, 3)[:, 1]
    w1, w2, w3 = ca0[1:-2] - ca0[:-3], ca0[2:-1] - ca0[1:-2], ca0[3:] - ca0[2:-1]
    ref_sign = torch.sign((torch.cross(w1, w2, dim=-1) * w3).sum(-1))
    agree = (dih_sign == ref_sign).double().mean()
    edm_w = (sigma ** 2 + SIGMA_DATA ** 2) / (sigma * SIGMA_DATA) ** 2
    print(f"  sigma={sigma:6.1f}  FAPE(noisy,native)={f:7.4f}  "
          f"CA-dihedral-sign agreement with native = {agree:5.1%}  EDM w={edm_w:8.4f}")
