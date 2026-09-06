import importlib.util

import torch

_sp = importlib.util.spec_from_file_location(
    "af3d", "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/proteinfoundation/nn/af3_diffusion.py")
af3d = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(af3d)

torch.manual_seed(11)
head = af3d.AF3DiffusionHead(c_s=16, c_z=8, c_token=16, n_blocks=1, n_heads=2, c_noise_embedding=16)
head.eval()

Lr = 58          # real residues
Lp = 100         # padded length (42% padding, the real run's ratio)
s_full = torch.randn(1, Lp, 16)
z_full = torch.randn(1, Lp, Lp, 8)
m_pad = torch.zeros(1, Lp); m_pad[:, :Lr] = 1
m_tight = torch.ones(1, Lr)

# Same RNG stream for both; the padded run draws MORE numbers per step (shape [1,Lp,3] vs [1,Lr,3]),
# so use a generator seeded identically and compare only the STATISTICS, plus a direct
# per-step centroid check which needs no RNG matching.
g1 = torch.Generator().manual_seed(7)
g2 = torch.Generator().manual_seed(7)
x_pad = head.rollout(s_full, z_full, m_pad, n_steps=8, generator=g1)
x_tight = head.rollout(s_full[:, :Lr], z_full[:, :Lr, :Lr], m_tight, n_steps=8, generator=g2)

print("== AF3DiffusionHead.rollout: padded vs tight, same seed ==")
real = x_pad[0, :Lr]
print("  padded-run real-region  |centroid| = %.3f   Rg = %.3f" %
      (real.mean(0).norm().item(),
       (real - real.mean(0)).pow(2).sum(-1).mean().sqrt().item()))
print("  tight-run               |centroid| = %.3f   Rg = %.3f" %
      (x_tight[0].mean(0).norm().item(),
       (x_tight[0] - x_tight[0].mean(0)).pow(2).sum(-1).mean().sqrt().item()))

# Decisive, RNG-free check: instrument the loop and report the centring error the code makes.
print()
print("== how far off is the UNMASKED centring, per step? ==")
torch.manual_seed(5)
B, L = 1, Lp
m = m_pad
sig = af3d.noise_schedule(torch.linspace(0, 1, 9))
x = sig[0] * torch.randn(B, L, 3)
for i in range(4):
    unmasked_mean = x.mean(dim=1)
    masked_mean = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    print(f"  step {i}: sigma={sig[i]:9.2f}  |unmasked mean|={unmasked_mean.norm():9.3f} "
          f" |true (masked) mean|={masked_mean.norm():9.3f} "
          f" centring error={(unmasked_mean-masked_mean).norm():9.3f} A")
    x = x - unmasked_mean[:, None]          # what the code does
    x = x + 1.0 * torch.randn(B, 1, 3)
    t_hat = sig[i]
    x = x + torch.randn_like(x) * 0.0
    x = x + (sig[i + 1] - t_hat) * torch.randn_like(x) * 0.1
