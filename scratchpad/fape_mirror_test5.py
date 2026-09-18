"""Does a FAPE term still SEPARATE the two hands when the prediction is bad?

At high sigma the denoiser output is far from any native, which is exactly the regime where
the mirror branch is still undecided. If clamped FAPE saturates there, a sigma-flat FAPE term
delivers NO branch-selecting gradient in the only regime where one is needed.
"""
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
Rn, tn = from_3_points(xn.reshape(L, 3, 3)[:, 0], xn.reshape(L, 3, 3)[:, 1],
                       xn.reshape(L, 3, 3)[:, 2])
g = torch.Generator().manual_seed(3)


def metrics(xp, clamp):
    bb = xp.reshape(L, 3, 3)
    Rp, tp = from_3_points(bb[:, 0], bb[:, 1], bb[:, 2])
    f, _, _ = fape(Rp, tp, Rn, tn, xp, xn, clamp=clamp)
    al = kabsch_align_gt_to_pred(xp, xn)
    return f, torch.sqrt(((xp - xp.mean(0) - al) ** 2).sum(-1).mean())


print("Prediction = (right-handed or mirrored native) + Gaussian perturbation of scale delta.")
print("GAP = value(mirrored branch) - value(correct branch). GAP -> 0 means no branch signal.")
print(f"  {'delta A':>8s} | {'FAPE_clamp10':>22s} | {'FAPE_unclamped':>22s} | {'RMSD (det=+1)':>22s}")
print(f"  {'':>8s} | {'right':>7s}{'mirror':>7s}{'GAP':>8s} | "
      f"{'right':>7s}{'mirror':>7s}{'GAP':>8s} | {'right':>7s}{'mirror':>7s}{'GAP':>8s}")
for delta in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    row = []
    for clamp in [10.0, None]:
        a = b = 0.0
        for _ in range(8):
            e = delta * torch.randn(xn.shape, generator=g)
            a += metrics(xn + e, clamp)[0] / 8
            b += metrics(xm + e, clamp)[0] / 8
        row.append((a, b, b - a))
    ra = rb = 0.0
    for _ in range(8):
        e = delta * torch.randn(xn.shape, generator=g)
        ra += metrics(xn + e, 10.0)[1] / 8
        rb += metrics(xm + e, 10.0)[1] / 8
    row.append((ra, rb, rb - ra))
    s = f"  {delta:8.1f} |"
    for a, b, d in row:
        s += f" {a:7.3f}{b:7.3f}{d:8.3f} |"
    print(s)
