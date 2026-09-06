import math
import sys

import torch

import importlib.util
_sp = importlib.util.spec_from_file_location(
    "af3d", "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/proteinfoundation/nn/af3_diffusion.py")
af3d = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(af3d)
SIGMA_DATA, P_MEAN, P_STD, S_MAX, S_MIN, RHO = af3d.SIGMA_DATA, af3d.P_MEAN, af3d.P_STD, af3d.S_MAX, af3d.S_MIN, af3d.RHO
noise_schedule, sample_noise_level = af3d.noise_schedule, af3d.sample_noise_level
smooth_lddt, weighted_rigid_align, diffusion_loss = af3d.smooth_lddt, af3d.weighted_rigid_align, af3d.diffusion_loss

torch.manual_seed(0)

print("== 1. EDM identity  w * c_out^2 == 1 ==")
sig = torch.tensor([1e-4, 1e-3, 1e-2, 0.1, 1.0, 4.8, 16.0, 100.0, 1000.0, 2560.0], dtype=torch.float64)
r = sig / SIGMA_DATA
c_out = sig / torch.sqrt(1 + r ** 2)
c_skip = 1 / (1 + r ** 2)
c_in = 1 / torch.sqrt(SIGMA_DATA ** 2 + sig ** 2)
w = (sig ** 2 + SIGMA_DATA ** 2) / (sig * SIGMA_DATA) ** 2
for i in range(len(sig)):
    print(f"  sigma={sig[i]:>10.4g}  c_in={c_in[i]:.4g} c_skip={c_skip[i]:.4g} c_out={c_out[i]:.4g} "
          f"w={w[i]:.4g}  w*c_out^2={(w[i]*c_out[i]**2).item():.12f}")
print("  max |w*c_out^2 - 1| =", (w * c_out ** 2 - 1).abs().max().item())

print()
print("== 2. noise_schedule range vs training sigma distribution ==")
t = torch.linspace(0, 1, 21)
s = noise_schedule(t)
print("  sigma[0] =", s[0].item(), " sigma[-1] =", s[-1].item())
print("  first 5:", [round(v, 4) for v in s[:5].tolist()])
print("  last  5:", [round(v, 6) for v in s[-5:].tolist()])
print("  monotone decreasing:", bool((s[1:] < s[:-1]).all()))
mu, sd = math.log(SIGMA_DATA) + P_MEAN, P_STD
for q in [0.0001, 0.001, 0.5, 0.999, 0.9999]:
    from statistics import NormalDist
    z = NormalDist().inv_cdf(q)
    print(f"  train sigma q={q:<8} = {math.exp(mu + sd*z):.6g}")
print(f"  z-score of inference sigma_max {s[0].item():.1f} in train dist:",
      (math.log(s[0].item()) - mu) / sd)
print(f"  z-score of inference sigma_min {s[-1].item():.6f}:",
      (math.log(s[-1].item()) - mu) / sd)

print()
print("== 3. training sigma tail statistics (1e7 draws) ==")
x = sample_noise_level((10_000_000,), "cpu")
w_all = (x ** 2 + SIGMA_DATA ** 2) / (x * SIGMA_DATA) ** 2
print("  sigma  min/med/max:", x.min().item(), x.median().item(), x.max().item())
print("  weight min/med/max:", w_all.min().item(), w_all.median().item(), w_all.max().item())
print("  P(sigma < 0.01) =", (x < 0.01).float().mean().item())
print("  P(sigma > 160)  =", (x > 160).float().mean().item())
print("  P(sigma > 2560) =", (x > 2560).float().mean().item())

print()
print("== 4. cdist backward with MANY coincident padded atoms at the origin ==")
B, A = 2, 200
xm = torch.zeros(B, A)
xm[:, :120] = 1.0
xx = torch.randn(B, A, 3, requires_grad=True) * 10
xx = (torch.randn(B, A, 3) * 10 * xm[..., None]).requires_grad_(True)
gt = torch.randn(B, A, 3) * 10 * xm[..., None]
l = smooth_lddt(xx, gt, xm)
l.sum().backward()
print("  loss:", l.tolist())
print("  grad has nan:", bool(torch.isnan(xx.grad).any()), " has inf:", bool(torch.isinf(xx.grad).any()))
print("  grad on PADDED slots (should be 0):", xx.grad[:, 120:].abs().max().item())

print()
print("== 5. diffusion_loss boundedness sweep, network output F ~ N(0,1) ==")
torch.manual_seed(1)
B, A = 8, 128
maskA = torch.ones(B, A)
gt = torch.randn(B, A, 3) * 10.0
gt = gt - gt.mean(1, keepdim=True)
for sg in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 4.8, 16.0, 100.0, 1000.0, 2560.0, 1e5]:
    sigma = torch.full((B,), sg)
    b = sigma[:, None, None]
    xn = gt + torch.randn_like(gt) * b
    F_ = torch.randn(B, A, 3)                       # unit-scale network output
    D = xn / (1 + (b / SIGMA_DATA) ** 2) + F_ * b / torch.sqrt(1 + (b / SIGMA_DATA) ** 2)
    loss, aux = diffusion_loss(D, gt, sigma, maskA)
    print(f"  sigma={sg:<9g} loss={loss.mean().item():>12.4f}  mse={aux['mse'].mean().item():>12.4g}"
          f"  w={aux['edm_weight'].mean().item():>10.4g}")

print()
print("== 6. same sweep but F == 0 (well-gated net at init) ==")
for sg in [1e-4, 1e-2, 1.0, 16.0, 1000.0, 2560.0]:
    sigma = torch.full((B,), sg)
    b = sigma[:, None, None]
    xn = gt + torch.randn_like(gt) * b
    D = xn / (1 + (b / SIGMA_DATA) ** 2)
    loss, aux = diffusion_loss(D, gt, sigma, maskA)
    print(f"  sigma={sg:<9g} loss={loss.mean().item():>12.4f}  mse={aux['mse'].mean().item():>12.4g}")

print()
print("== 7. weighted_rigid_align on a DEGENERATE prediction (all atoms ~ origin) ==")
xd = torch.zeros(4, 64, 3)
g = torch.randn(4, 64, 3) * 10
m = torch.ones(4, 64)
al = weighted_rigid_align(xd, g, m, m)
print("  aligned finite:", bool(torch.isfinite(al).all()), " norm:", al.norm().item())
xd2 = torch.randn(4, 64, 3) * 1e-7
al2 = weighted_rigid_align(xd2, g, m, m)
print("  near-degenerate finite:", bool(torch.isfinite(al2).all()))

print()
print("== 8. padded rows leak into the align/loss? ==")
B, A = 3, 64
m = torch.zeros(B, A); m[:, :40] = 1
gt = torch.randn(B, A, 3) * 10 * m[..., None]
pred = gt.clone()
pred[:, 40:] = 1e4                       # garbage in padded slots
sigma = torch.full((B,), 1.0)
l1, a1 = diffusion_loss(pred, gt, sigma, m, use_smooth_lddt=True)
pred2 = gt.clone()
l2, a2 = diffusion_loss(pred2, gt, sigma, m, use_smooth_lddt=True)
print("  loss with garbage padding:", l1.tolist())
print("  loss with clean   padding:", l2.tolist())
