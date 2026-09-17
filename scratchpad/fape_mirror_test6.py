"""Why does FAPE lose the branch signal so fast, and can a windowed FAPE recover it?

Hypothesis: frames inferred from three PREDICTED atoms 1.46/1.53 A apart amplify coordinate
noise into large frame rotations; at a long lever arm that error swamps the 2|z| mirror signal.
"""
import math

import torch

torch.set_default_dtype(torch.float64)
exec(open('/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/scratchpad/'
          'fape_mirror_test.py').read().split('L = 30')[0])

seg = [('H', 14), ('L', 4), ('H', 14), ('L', 4), ('E', 10), ('L', 4), ('H', 14)]
TOR = {'H': (-57., -47.), 'E': (-135., 135.), 'L': (-70., 140.)}
phis, psis = [], []
for k, n in seg:
    p, q = TOR[k]
    phis += [p] * n
    psis += [q] * n
L = len(phis)
NAT = build(phis, psis, [180.] * (L - 1))
MIR = NAT @ torch.diag(torch.tensor([1., 1., -1.]))
xn = NAT.reshape(-1, 3) - NAT.reshape(-1, 3).mean(0)
xm = MIR.reshape(-1, 3) - MIR.reshape(-1, 3).mean(0)
bb = xn.reshape(L, 3, 3)
Rn, tn = from_3_points(bb[:, 0], bb[:, 1], bb[:, 2])
g = torch.Generator().manual_seed(7)

print("A. Frame-rotation noise amplification: a delta-A coordinate error on N/CA/C rotates the")
print("   residue frame by how much? (N-CA = 1.458 A, CA-C = 1.525 A lever arms)")
print(f"  {'delta A':>8s} {'median frame rotation':>24s} {'error at 20 A lever arm':>26s}")
for delta in [0.1, 0.25, 0.5, 1.0, 2.0]:
    e = delta * torch.randn((8,) + xn.shape, generator=g)
    ang = []
    for k in range(8):
        b = (xn + e[k]).reshape(L, 3, 3)
        Rp, _ = from_3_points(b[:, 0], b[:, 1], b[:, 2])
        dR = torch.einsum('fji,fjk->fik', Rn, Rp)
        tr = dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]
        ang.append(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
    a = torch.cat(ang).median()
    print(f"  {delta:8.2f} {math.degrees(a):21.1f} deg {20 * a:23.1f} A")

print()
print("B. WINDOWED FAPE: restrict (frame i, atom j) pairs to |i-j| <= W. Short lever arm =>")
print("   less frame-noise amplification. Does the mirror GAP survive a realistic prediction error?")
res_of_atom = torch.arange(L).repeat_interleave(3)
sep = (torch.arange(L)[:, None] - res_of_atom[None, :]).abs()


def wfape(xp, W, clamp=10.0):
    b = xp.reshape(L, 3, 3)
    Rp, tp = from_3_points(b[:, 0], b[:, 1], b[:, 2])
    lp = torch.einsum('fji,fpj->fpi', Rp, xp[None] - tp[:, None])
    lt = torch.einsum('fji,fpj->fpi', Rn, xn[None] - tn[:, None])
    e = torch.sqrt(((lp - lt) ** 2).sum(-1) + 1e-8)
    if clamp is not None:
        e = e.clamp(0, clamp)
    m = sep <= W
    return (e[m] / 10.0).mean()


hdr = "  " + f"{'delta A':>8s}" + "".join(f"{'W=' + str(W):>22s}" for W in [1, 2, 4, 8, 10 ** 6])
print(hdr)
print("  " + " " * 8 + "".join(f"{'right':>7s}{'mirr':>7s}{'GAP':>8s}" for _ in range(5)))
for delta in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
    s = f"  {delta:8.2f}"
    for W in [1, 2, 4, 8, 10 ** 6]:
        a = b = 0.0
        for _ in range(12):
            e = delta * torch.randn(xn.shape, generator=g)
            a += wfape(xn + e, W) / 12
            b += wfape(xm + e, W) / 12
        s += f"{a:7.3f}{b:7.3f}{b - a:8.3f}"
    print(s)

print()
print("C. Same GAP for the achiral / existing terms, as a baseline (delta = 1.0 A):")
e = torch.randn(xn.shape, generator=g)
for nm, f in [("distogram |D-D_gt| (achiral)",
               lambda x: (torch.cdist(x, x) - torch.cdist(xn, xn)).abs().mean()),
              ("det=+1 Kabsch RMSD",
               lambda x: torch.sqrt(((x - x.mean(0) - kabsch_align_gt_to_pred(x, xn)) ** 2)
                                    .sum(-1).mean()))]:
    a, b = f(xn + e), f(xm + e)
    print(f"    {nm:32s} right={a:8.4f}  mirror={b:8.4f}  GAP={b - a:8.4f}")
