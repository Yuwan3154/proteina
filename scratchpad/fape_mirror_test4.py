"""Numerical-stability risk: frames built by Gram-Schmidt on a DIFFUSION model's raw output.

openfold from_3_points uses eps=1e-8 inside sqrt (rigid_utils.py:1176-1218). If predicted
N-CA-C are near collinear the e1 normalisation and the e2 cross product blow up.
"""
import torch

torch.set_default_dtype(torch.float64)
exec(open('/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/scratchpad/'
          'fape_mirror_test.py').read().split('L = 30')[0])

L = 64
NAT = build([-57.] * L, [-47.] * L, [180.] * (L - 1))
xn = NAT.reshape(-1, 3)
xc = xn - xn.mean(0)
g = torch.Generator().manual_seed(1)

print("sin(N-CA-C angle) after Gram-Schmidt orthogonalisation, i.e. |e1_raw - e0(e0.e1_raw)| / |e1_raw|")
print("Small values => degenerate frame. eps in from_3_points is 1e-8.")
print(f"  {'sigma':>7s} {'min sin':>10s} {'p1 sin':>10s} {'median':>10s} "
      f"{'frac<1e-2':>10s} {'max |dR/dx|':>12s}")
for sigma in [0.0, 0.5, 2.0, 8.0, 16.0, 40.0, 160.0]:
    noisy = (xc + sigma * torch.randn(xc.shape, generator=g)).reshape(L, 3, 3)
    p, o, q = noisy[:, 0], noisy[:, 1], noisy[:, 2]
    e0 = o - p
    e0 = e0 / e0.norm(dim=-1, keepdim=True)
    e1r = q - o
    perp = e1r - e0 * (e0 * e1r).sum(-1, keepdim=True)
    s = perp.norm(dim=-1) / e1r.norm(dim=-1)
    # gradient sensitivity of the frame wrt the atom coords
    nn_ = noisy.clone().requires_grad_(True)
    R, _ = from_3_points(nn_[:, 0], nn_[:, 1], nn_[:, 2])
    R.sum().backward()
    print(f"  {sigma:7.1f} {s.min():10.2e} {torch.quantile(s, 0.01):10.2e} "
          f"{s.median():10.4f} {(s < 1e-2).double().mean():10.4f} {nn_.grad.abs().max():12.2e}")
