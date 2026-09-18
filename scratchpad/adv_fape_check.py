"""ADVERSARIAL CHECK of the 'FAPE is a worse mirror detector' claim.

test5 degrades the prediction with IID per-atom Gaussian noise of scale delta. Bonds are
1.458/1.525 A, so delta=0.5 puts a ~0.71 A RMS error ON THE BOND VECTORS. That is not
'prediction error', it is destruction of local geometry. Here we (a) quantify that, and (b)
re-run the identical GAP measurement under a degradation that keeps local geometry EXACT and
gets the global fold wrong -- torsion-space noise -- which is the measured c2c failure mode.
"""
import math
import torch

torch.set_default_dtype(torch.float64)
SRC = ('/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/scratchpad/'
       'fape_mirror_test.py')
exec(open(SRC).read().split('L = 30')[0])

seg = [('H', 14), ('L', 4), ('H', 14), ('L', 4), ('E', 10), ('L', 4), ('H', 14)]
TOR = {'H': (-57., -47.), 'E': (-135., 135.), 'L': (-70., 140.)}
phis, psis = [], []
for k, n in seg:
    p, q = TOR[k]
    phis += [p] * n
    psis += [q] * n
L = len(phis)
PHI = torch.tensor(phis)
PSI = torch.tensor(psis)
OM = [180.] * (L - 1)

NAT = build(phis, psis, OM)
xn = NAT.reshape(-1, 3)
xn = xn - xn.mean(0)
Rn, tn = from_3_points(xn.reshape(L, 3, 3)[:, 0], xn.reshape(L, 3, 3)[:, 1],
                       xn.reshape(L, 3, 3)[:, 2])

res_of_atom = torch.arange(L).repeat_interleave(3)
SEP = (torch.arange(L)[:, None] - res_of_atom[None, :]).abs()


def fape_full(xp, clamp, window=None):
    bb = xp.reshape(L, 3, 3)
    Rp, tp = from_3_points(bb[:, 0], bb[:, 1], bb[:, 2])
    lp = torch.einsum('fji,fpj->fpi', Rp, xp[None] - tp[:, None])
    lt = torch.einsum('fji,fpj->fpi', Rn, xn[None] - tn[:, None])
    e = torch.sqrt(((lp - lt) ** 2).sum(-1) + 1e-8)
    if clamp is not None:
        e = e.clamp(0, clamp)
    e = e / 10.0
    if window is None:
        return e.mean()
    m = (SEP <= window).to(e.dtype)
    return (e * m).sum() / m.sum()


def rmsd_kabsch(xp):
    al = kabsch_align_gt_to_pred(xp, xn)
    return torch.sqrt(((xp - xp.mean(0) - al) ** 2).sum(-1).mean())


def bond_stats(xp):
    bb = xp.reshape(L, 3, 3)
    b1 = (bb[:, 1] - bb[:, 0]).norm(dim=-1)
    b2 = (bb[:, 2] - bb[:, 1]).norm(dim=-1)
    return ((b1 - B_N_CA).abs().mean() + (b2 - B_CA_C).abs().mean()) / 2


print("=" * 104)
print("STEP 1. What does test5's delta axis actually do to LOCAL geometry?")
print("  (a trained structure model has bond-length error << 0.05 A)")
g = torch.Generator().manual_seed(11)
print(f"  {'delta A':>8s} {'mean|bond err| A':>17s} {'as % of 1.458 A':>17s} "
      f"{'global RMSD':>12s} {'FAPE_c10 of CORRECT branch':>28s}")
for delta in [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]:
    be = rm = fa = 0.0
    for _ in range(8):
        xp = xn + delta * torch.randn(xn.shape, generator=g)
        be += bond_stats(xp) / 8
        rm += rmsd_kabsch(xp) / 8
        fa += fape_full(xp, 10.0) / 8
    print(f"  {delta:8.2f} {be:17.4f} {be / B_N_CA:16.1%} {rm:12.3f} {fa:28.3f}")

print()
print("=" * 104)
print("STEP 2. SAME GAP measurement, degradation = TORSION noise (local geometry stays IDEAL).")
print("  right branch  = build(phi+e, psi+e);  mirror branch = build(-phi+e, -psi+e)")
print("  GAP = value(mirror branch) - value(correct branch).  W3 = FAPE clamped, |i-j|<=3 only.")
print(f"  {'tors sd':>7s} {'RMSDcorr':>8s} | {'FAPEc10 r':>9s}{'m':>7s}{'GAP':>7s} | "
      f"{'FAPEunc r':>9s}{'m':>8s}{'GAP':>8s} | {'W3 r':>7s}{'m':>7s}{'GAP':>7s} | "
      f"{'Kab r':>7s}{'m':>7s}{'GAP':>7s}")
for sd in [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]:
    acc = torch.zeros(12)
    NR = 8
    for _ in range(NR):
        ep = sd * torch.randn(L, generator=g)
        eq = sd * torch.randn(L, generator=g)
        xr = build((PHI + ep).tolist(), (PSI + eq).tolist(), OM).reshape(-1, 3)
        xm = build((-PHI + ep).tolist(), (-PSI + eq).tolist(), OM).reshape(-1, 3)
        xr = xr - xr.mean(0)
        xm = xm - xm.mean(0)
        v = torch.tensor([rmsd_kabsch(xr),
                          fape_full(xr, 10.0), fape_full(xm, 10.0),
                          fape_full(xr, None), fape_full(xm, None),
                          fape_full(xr, 10.0, 3), fape_full(xm, 10.0, 3),
                          rmsd_kabsch(xr), rmsd_kabsch(xm), 0, 0, 0])
        acc += v / NR
    r = acc
    print(f"  {sd:7.1f} {r[0]:8.3f} | {r[1]:9.3f}{r[2]:7.3f}{r[2]-r[1]:7.3f} | "
          f"{r[3]:9.3f}{r[4]:8.3f}{r[4]-r[3]:8.3f} | {r[5]:7.3f}{r[6]:7.3f}{r[6]-r[5]:7.3f} | "
          f"{r[7]:7.3f}{r[8]:7.3f}{r[8]-r[7]:7.3f}")

print()
print("=" * 104)
print("STEP 3. MATCHED-RMSD head-to-head: IID vs TORSION degradation at the SAME global RMSD.")


def iid_at(delta, n=8):
    out = torch.zeros(5)
    M = torch.diag(torch.tensor([1., 1., -1.]))
    xm0 = xn @ M
    xm0 = xm0 - xm0.mean(0)
    for _ in range(n):
        e = delta * torch.randn(xn.shape, generator=g)
        out += torch.tensor([rmsd_kabsch(xn + e),
                             fape_full(xm0 + e, 10.0) - fape_full(xn + e, 10.0),
                             fape_full(xm0 + e, None) - fape_full(xn + e, None),
                             fape_full(xm0 + e, 10.0, 3) - fape_full(xn + e, 10.0, 3),
                             rmsd_kabsch(xm0 + e) - rmsd_kabsch(xn + e)]) / n
    return out


def tors_at(sd, n=8):
    out = torch.zeros(5)
    for _ in range(n):
        ep = sd * torch.randn(L, generator=g)
        eq = sd * torch.randn(L, generator=g)
        xr = build((PHI + ep).tolist(), (PSI + eq).tolist(), OM).reshape(-1, 3)
        xm = build((-PHI + ep).tolist(), (-PSI + eq).tolist(), OM).reshape(-1, 3)
        xr = xr - xr.mean(0)
        xm = xm - xm.mean(0)
        out += torch.tensor([rmsd_kabsch(xr),
                             fape_full(xm, 10.0) - fape_full(xr, 10.0),
                             fape_full(xm, None) - fape_full(xr, None),
                             fape_full(xm, 10.0, 3) - fape_full(xr, 10.0, 3),
                             rmsd_kabsch(xm) - rmsd_kabsch(xr)]) / n
    return out


print(f"  {'mode':>10s} {'knob':>8s} {'RMSDcorr':>9s} | {'GAP FAPEc10':>12s} {'GAP FAPEunc':>12s} "
      f"{'GAP W3':>8s} {'GAP Kabsch A':>13s}")
for d in [0.5, 1.0, 2.0]:
    r = iid_at(d)
    print(f"  {'IID':>10s} {d:8.2f} {r[0]:9.2f} | {r[1]:12.3f} {r[2]:12.3f} {r[3]:8.3f} "
          f"{r[4]:13.3f}")
for s in [5.0, 10.0, 20.0, 30.0]:
    r = tors_at(s)
    print(f"  {'TORSION':>10s} {s:8.1f} {r[0]:9.2f} | {r[1]:12.3f} {r[2]:12.3f} {r[3]:8.3f} "
          f"{r[4]:13.3f}")
