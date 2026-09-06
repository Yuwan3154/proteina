import importlib.util
import math

import torch
import torch.nn as nn

_sp = importlib.util.spec_from_file_location(
    "af3d", "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/proteinfoundation/nn/af3_diffusion.py")
af3d = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(af3d)

torch.manual_seed(0)
SD = af3d.SIGMA_DATA

print("== A. denoise() == EDM D = c_skip*x + c_out*F, exactly? ==")
head = af3d.AF3DiffusionHead(c_s=32, c_z=16, c_token=32, n_blocks=2, n_heads=4, c_noise_embedding=32)
head.eval()
B, L = 3, 12
s = torch.randn(B, L, 32); z = torch.randn(B, L, L, 16); m = torch.ones(B, L)
for sg in [1e-3, 1.0, 16.0, 2560.0]:
    sigma = torch.full((B,), sg)
    x = torch.randn(B, L, 3) * 10
    with torch.no_grad():
        D = head.denoise(x, sigma, s, z, m)
        Fo = head._f_forward(x / math.sqrt(SD ** 2 + sg ** 2), sigma, s, z, m)
    c_skip = SD ** 2 / (sg ** 2 + SD ** 2)
    c_out = sg * SD / math.sqrt(sg ** 2 + SD ** 2)
    ref = c_skip * x + c_out * Fo
    print(f"  sigma={sg:<9g} max|D-ref| = {(D-ref).abs().max().item():.3e}  (|D|={D.abs().max():.3g})")

print()
print("== B. pair-bias orientation: does z[b,i,j] bias query i -> key j ? ==")
torch.manual_seed(3)
at = af3d.AttentionPairBias(c_a=8, c_s=8, c_z=4, n_heads=1)
at.eval()
B, L = 1, 5
a = torch.randn(B, L, 8); ss = torch.randn(B, L, 8); m = torch.ones(B, L)
zz = torch.zeros(B, L, L, 4)
with torch.no_grad():
    at.to_bias.weight.fill_(0.0); at.to_bias.weight[0, 0] = 50.0
    at.norm_z.weight.fill_(1.0); at.norm_z.bias.fill_(0.0)
    # make a "z" that is huge only at (i=1, j=3) in channel 0
    zz[0, 1, 3, 0] = 1.0
    # recover the actual attention matrix by probing v
    at.to_v.weight.copy_(torch.eye(8))
    a_n = at.adaln(a, ss)
    q = at.q_norm(at.to_q(a_n).view(B, L, 1, 8)).transpose(1, 2)
    k = at.k_norm(at.to_k(a_n).view(B, L, 1, 8)).transpose(1, 2)
    bias = at.to_bias(at.norm_z(zz)).permute(0, 3, 1, 2)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(8) + bias
    A = logits.softmax(-1)[0, 0]
print("  argmax key per query row:", A.argmax(-1).tolist(), " (expect row 1 -> key 3)")
print("  A[1,3] =", A[1, 3].item(), "  A[3,1] =", A[3, 1].item())

print()
print("== C. does the key mask really kill padded keys? ==")
torch.manual_seed(4)
at2 = af3d.AttentionPairBias(c_a=8, c_s=8, c_z=4, n_heads=2)
at2.eval()
B, L = 2, 6
m = torch.zeros(B, L); m[:, :4] = 1
a = torch.randn(B, L, 8); ss = torch.randn(B, L, 8); zz = torch.randn(B, L, L, 4)
with torch.no_grad():
    o1 = at2(a, ss, zz, m)
    a2 = a.clone(); a2[:, 4:] = 1e3          # garbage in padded token slots
    zz2 = zz.clone(); zz2[:, :, 4:] = 1e3    # garbage in padded pair columns
    o2 = at2(a2, ss, zz2, m)
print("  max |out_real_rows difference| =", (o1[:, :4] - o2[:, :4]).abs().max().item())

print()
print("== D. bf16 autocast: finfo.min mask through SDPA, and loss autocast-escape ==")
if torch.backends.mps.is_available() or True:
    dev = "cpu"
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    o = at2(a, ss, zz, m)
    print("  attn out dtype:", o.dtype, " finite:", bool(torch.isfinite(o).all()))
    B2, A2 = 2, 32
    mm = torch.ones(B2, A2)
    gt = torch.randn(B2, A2, 3) * 10
    pred = gt + torch.randn(B2, A2, 3)
    sg = torch.full((B2,), 4.0)
    l, aux = af3d.diffusion_loss(pred, gt, sg, mm)
    print("  loss dtype inside bf16 autocast:", l.dtype, " val:", l.tolist())

print()
print("== E. AdaLN / gate init: is s dead at init? ==")
blk = af3d.DiffusionTransformerBlock(c_a=16, c_s=8, c_z=4, n_heads=2)
blk.eval()
B, L = 2, 5
a = torch.randn(B, L, 16); z = torch.randn(B, L, L, 4); m = torch.ones(B, L)
s1 = torch.randn(B, L, 8); s2 = torch.randn(B, L, 8) * 7 + 3
with torch.no_grad():
    y1 = blk(a, s1, z, m); y2 = blk(a, s2, z, m)
print("  max|block(a,s1) - block(a,s2)| at init =", (y1 - y2).abs().max().item())
print("  -> 0 means the conditioning s (and hence sigma) has NO effect on any block at init")
print("  adaln.to_gamma.weight all-zero:", bool((blk.attn.adaln.to_gamma.weight == 0).all()))
print("  adaln.to_beta.weight  all-zero:", bool((blk.attn.adaln.to_beta.weight == 0).all()))
print("  attn.out_scale.weight all-zero:", bool((blk.attn.out_scale.weight == 0).all()))
print("  trans.out_scale.weight all-zero:", bool((blk.transition.out_scale.weight == 0).all()))
# gradient reach from block output back to s
s3 = torch.randn(B, L, 8, requires_grad=True)
blk.zero_grad()
blk(a, s3, z, m).sum().backward()
print("  |dOut/ds| at init =", s3.grad.abs().max().item())

print()
print("== F. AF3DiffusionHead.rollout: is padding masked inside the loop? ==")
import inspect
src = inspect.getsource(af3d.AF3DiffusionHead.rollout)
print("  'x.mean(dim=1' present (UNMASKED centring):", "x.mean(dim=1" in src)
print("  any '* mask' inside the loop:", src.count("mask[..., None]"))
# empirical: does padding move the centroid?
head2 = af3d.AF3DiffusionHead(c_s=16, c_z=8, c_token=16, n_blocks=1, n_heads=2, c_noise_embedding=16)
B, L = 1, 100
mk = torch.zeros(B, L); mk[:, :58] = 1        # 42% padding, as in the real run
sH = torch.randn(B, L, 16); zH = torch.randn(B, L, L, 8)
torch.manual_seed(9)
out = head2.rollout(sH, zH, mk, n_steps=6)
real = out[0, :58]
print("  masked-region centroid after rollout:", real.mean(0).tolist())
print("  |centroid| =", real.mean(0).norm().item(), " Rg =", (real - real.mean(0)).norm(dim=-1).pow(2).mean().sqrt().item())
