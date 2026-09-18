"""Where in the sigma spectrum does the mirror penalty actually live?

Existing c2c mirror penalty  = w(sigma) * mse_mirror        (af3_diffusion.py:344-345)
Hypothetical FAPE term       = alpha_fape * FAPE_mirror     (sigma-flat, like smooth_lddt:349)
"""
import torch

torch.set_default_dtype(torch.float64)

SIGMA_DATA, P_MEAN, P_STD = 16.0, -1.2, 1.5      # af3_diffusion.py:28-30
N = 2_000_000
g = torch.Generator().manual_seed(0)
sig = SIGMA_DATA * torch.exp(P_MEAN + P_STD * torch.randn(N, generator=g))
w = (sig ** 2 + SIGMA_DATA ** 2) / (sig * SIGMA_DATA) ** 2

print("Training noise distribution, af3_diffusion.py:43-49 (sigma_data=16, P_mean=-1.2, P_std=1.5)")
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
print("  quantiles of sigma :", "  ".join(f"q{int(q*100)}={torch.quantile(sig, q):.2f}" for q in qs))
print("  quantiles of EDM w :", "  ".join(f"q{int(q*100)}={torch.quantile(w, q):.4f}" for q in qs))
print()

# Threshold above which a noised native no longer reveals the global hand.
# Measured in fape_mirror_test2.py: CA-dihedral-sign agreement 93% @0.5, 82% @2, 56% @8, 51% @16.
for S in [2.0, 4.0, 8.0, 16.0]:
    hi = sig > S
    frac_steps = hi.double().mean()
    # fraction of the TOTAL sigma-integrated mirror penalty contributed by sigma > S
    frac_edm = (w * hi).sum() / w.sum()
    frac_flat = frac_steps                       # a sigma-flat term weights every step equally
    print(f"  sigma > {S:5.1f} A : {frac_steps:6.2%} of training steps | "
          f"carries {frac_edm:7.4%} of the EDM-weighted mirror penalty | "
          f"{frac_flat:6.2%} of a sigma-flat one   (ratio {float(frac_flat/frac_edm):6.1f}x)")

print()
exec(open('/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/scratchpad/'
          'fape_mirror_test.py').read().split('L = 30')[0])
seg = [('H', 14), ('L', 4), ('H', 14), ('L', 4), ('E', 10), ('L', 4), ('H', 14)]
TOR = {'H': (-57., -47.), 'E': (-135., 135.), 'L': (-70., 140.)}
phis, psis = [], []
for kind, n in seg:
    p, q = TOR[kind]
    phis += [p] * n
    psis += [q] * n
L = len(phis)
NAT = build(phis, psis, [180.] * (L - 1))
MIR = NAT @ torch.diag(torch.tensor([1., 1., -1.]))
xn, xm = NAT.reshape(-1, 3), MIR.reshape(-1, 3)
al = kabsch_align_gt_to_pred(xm, xn)
sq = ((xm - xm.mean(0) - al) ** 2).sum(-1).mean()
mse_mirror = sq / 3.0                               # af3_diffusion.py:343, the 1/3 prefactor
Rn, tn = from_3_points(NAT[:, 0], NAT[:, 1], NAT[:, 2])
Rm, tm = from_3_points(MIR[:, 0], MIR[:, 1], MIR[:, 2])
f10, _, _ = fape(Rm, tm, Rn, tn, xm, xn, clamp=10.0)

print(f"L={L} compact fold: mirror RMSD (det=+1 Kabsch) = {sq.sqrt():.2f} A -> "
      f"mse_mirror = {mse_mirror:.1f}; FAPE_clamp10(mirror) = {f10:.3f}")
print("Per-step mirror penalty actually delivered, by sigma bin "
      "(ALPHA_DIFFUSION=4.0, contact2coord_trainer.py:25):")
print(f"  {'sigma bin':>16s} {'%steps':>7s} {'4*w*mse_mirror':>15s} {'ratio to sigma-flat 1.0':>24s}")
edges = [0, 1, 2, 4, 8, 16, 32, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (sig > lo) & (sig <= hi)
    if m.sum() == 0:
        continue
    pen = 4.0 * w[m].mean() * mse_mirror
    print(f"  {f'{lo:g}-{hi:g}':>16s} {m.double().mean():6.2%} {pen:15.2f} {float(pen / f10):23.1f}")
