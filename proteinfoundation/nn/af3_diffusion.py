"""AF3-style token-level diffusion head, ported to a residue-level contact-map trunk.

Faithful to AlphaFold3 (Abramson et al., Nature 2024) supplementary information wherever the SI
specifies a value; every constant below carries its citation. Cross-checked against the two readable
reimplementations, Protenix (Apache-2.0) and OpenFold3.

⭐ SCOPE. AF3's DiffusionModule is all-atom: AtomAttentionEncoder -> DiffusionTransformer ->
AtomAttentionDecoder, where the atom machinery exists to represent ligands, nucleic acids and side
chains. Our trunk is residue-level with an L x L pair track and CA-only ground truth, so this is a
port of the TOKEN-LEVEL DiffusionTransformer (SI Algorithm 23) conditioned on (s, z), predicting CA
coordinates. AF3's EDM preconditioning, noise schedule, sampler and losses are kept exactly.

⛔ THE MINI-ROLLOUT IS NOT THE TRAINING PATH. SI 4.1: "at training time we do a short rollout of the
Diffusion Module from pure noise with 20 steps ... No gradients are applied to this mini-rollout."
It produces coordinates for the CONFIDENCE heads (pLDDT 4.3.1, PAE 4.3.2, PDE 4.3.3) and fixes the
ground-truth permutation (4.2). The module itself is trained by SINGLE-STEP denoising of noised
ground truth (SI 3.7.1). Both are provided here: `denoise` for training, `rollout` for confidence.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── AF3 constants, all cited ──────────────────────────────────────────────────────────────────
SIGMA_DATA = 16.0          # SI Alg. 20 header; DeepMind diffusion_head.py:39
P_MEAN = -1.2              # SI 3.7.1 training noise distribution
P_STD = 1.5                # SI 3.7.1
S_MAX = 160.0              # SI 3.7.1 Eq. 7 inference schedule
S_MIN = 4e-4               # SI 3.7.1 Eq. 7
RHO = 7.0                  # SI 3.7.1 Eq. 7 (the exponent p)
GAMMA_0 = 0.8              # SI Alg. 18
GAMMA_MIN = 1.0            # SI Alg. 18
NOISE_SCALE = 1.003        # SI Alg. 18 (lambda)
STEP_SCALE = 1.5           # SI Alg. 18 (eta)
MINI_ROLLOUT_STEPS = 20    # SI 4.1
FULL_INFERENCE_STEPS = 200  # SI 3.7.1; DeepMind diffusion_head.py:126
S_TRANS = 1.0              # SI Alg. 19 per-step random translation, Angstrom


def sample_noise_level(shape, device, dtype=torch.float32) -> torch.Tensor:
    """sigma = sigma_data * exp(P_mean + P_std * N(0,1))  -- SI 3.7.1.

    This is the STRUCTURE noise level, independent of the contact trunk's own diffusion time.
    """
    n = torch.randn(shape, device=device, dtype=dtype)
    return SIGMA_DATA * torch.exp(P_MEAN + P_STD * n)


def noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """SI 3.7.1 Eq. 7: sigma_data * (s_max^(1/p) + t*(s_min^(1/p) - s_max^(1/p)))^p, t in [0,1]."""
    a = S_MAX ** (1.0 / RHO)
    b = S_MIN ** (1.0 / RHO)
    return SIGMA_DATA * (a + t * (b - a)) ** RHO


class FourierEmbedding(nn.Module):
    """Random Fourier features for the noise level. SI Alg. 22; weights are fixed, not learned."""

    def __init__(self, c: int = 256):
        super().__init__()
        self.register_buffer("w", torch.randn(c), persistent=True)
        self.register_buffer("b", torch.rand(c), persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(2.0 * math.pi * (t[..., None] * self.w + self.b))


class AdaLN(nn.Module):
    """Adaptive LayerNorm: normalise a, then scale/shift from the conditioning s. SI Alg. 26."""

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        # elementwise_affine=False: the affine part comes from s, not from free parameters.
        self.norm_a = nn.LayerNorm(c_a, elementwise_affine=False)
        self.norm_s = nn.LayerNorm(c_s)
        self.to_gamma = nn.Linear(c_s, c_a)
        self.to_beta = nn.Linear(c_s, c_a, bias=False)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)      # sigmoid(1) ~ 0.73, AF3's non-zero default gate
        nn.init.zeros_(self.to_beta.weight)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s = self.norm_s(s)
        return torch.sigmoid(self.to_gamma(s)) * self.norm_a(a) + self.to_beta(s)


class AttentionPairBias(nn.Module):
    """Self-attention over tokens, biased by the pair representation. SI Alg. 24.

    The pair bias is what carries the trunk's structural reasoning into the diffusion transformer,
    so this is the load-bearing connection between the contact model and the coordinates.
    """

    def __init__(self, c_a: int, c_s: int, c_z: int, n_heads: int, bias_init: float = -2.0):
        super().__init__()
        assert c_a % n_heads == 0, (c_a, n_heads)
        self.n_heads = n_heads
        self.c_head = c_a // n_heads
        self.adaln = AdaLN(c_a, c_s)
        self.to_q = nn.Linear(c_a, c_a)
        self.to_k = nn.Linear(c_a, c_a, bias=False)
        self.to_v = nn.Linear(c_a, c_a, bias=False)
        self.norm_z = nn.LayerNorm(c_z)
        self.to_bias = nn.Linear(c_z, n_heads, bias=False)
        self.to_gate = nn.Linear(c_a, c_a, bias=False)
        self.to_out = nn.Linear(c_a, c_a, bias=False)
        # Output gate starts closed-ish so the residual branch begins near identity (SI Alg. 24
        # uses a -2.0 bias init on the conditioned output projection).
        self.out_scale = nn.Linear(c_s, c_a)
        nn.init.zeros_(self.out_scale.weight)
        nn.init.constant_(self.out_scale.bias, bias_init)
        # ⛔ to_out is NOT zero-initialised. The adaptive gate above already damps this branch to
        # sigmoid(-2.0) ~ 0.12 at init, which is the whole point of AF3's -2.0 bias. Zeroing the
        # weight on top of that makes the branch exactly zero, and since z reaches the loss ONLY
        # through this attention bias, it severs the gradient to z entirely -- the tri blocks and
        # the contact embedding then never learn. Protenix agrees: zero_init = not has_s
        # (transformer.py:94), so with conditioning present, as here, they do not zero-init.

    def forward(self, a, s, z, mask):
        B, L, _ = a.shape
        a_n = self.adaln(a, s)
        q = self.to_q(a_n).view(B, L, self.n_heads, self.c_head).transpose(1, 2)
        k = self.to_k(a_n).view(B, L, self.n_heads, self.c_head).transpose(1, 2)
        v = self.to_v(a_n).view(B, L, self.n_heads, self.c_head).transpose(1, 2)

        bias = self.to_bias(self.norm_z(z)).permute(0, 3, 1, 2)          # [B, H, L, L]
        # Mask keys, not queries: a padded query row is discarded downstream by the output mask,
        # but a padded KEY would leak into every real row's softmax.
        key_mask = mask[:, None, None, :].to(torch.bool)
        bias = bias.masked_fill(~key_mask, torch.finfo(bias.dtype).min)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = out * torch.sigmoid(self.to_gate(a_n))
        return torch.sigmoid(self.out_scale(s)) * self.to_out(out)


class ConditionedTransitionBlock(nn.Module):
    """SwiGLU transition with adaLN conditioning. SI Alg. 25, expansion n=2."""

    def __init__(self, c_a: int, c_s: int, n: int = 2, bias_init: float = -2.0):
        super().__init__()
        self.adaln = AdaLN(c_a, c_s)
        self.to_ab = nn.Linear(c_a, 2 * n * c_a, bias=False)
        self.to_out = nn.Linear(n * c_a, c_a, bias=False)
        self.out_scale = nn.Linear(c_s, c_a)
        nn.init.zeros_(self.out_scale.weight)
        nn.init.constant_(self.out_scale.bias, bias_init)
        # Not zero-initialised, for the same reason as AttentionPairBias above: the -2.0 gate is
        # already the damping mechanism, and a zero weight on top of it only severs gradients.

    def forward(self, a, s):
        h = self.to_ab(self.adaln(a, s))
        u, v = h.chunk(2, dim=-1)
        return torch.sigmoid(self.out_scale(s)) * self.to_out(F.silu(u) * v)


class DiffusionTransformerBlock(nn.Module):
    """SI Alg. 23 lines 2-3: attention-with-pair-bias, then a conditioned transition."""

    def __init__(self, c_a: int, c_s: int, c_z: int, n_heads: int):
        super().__init__()
        self.attn = AttentionPairBias(c_a, c_s, c_z, n_heads)
        self.transition = ConditionedTransitionBlock(c_a, c_s)

    def forward(self, a, s, z, mask):
        a = a + self.attn(a, s, z, mask)
        return a + self.transition(a, s)


class AF3DiffusionHead(nn.Module):
    """Token-level AF3 diffusion head: (s, z, noisy CA coords, sigma) -> denoised CA coords.

    EDM preconditioning exactly as AF3/Protenix:
        c_in   = 1 / sqrt(sigma_data^2 + sigma^2)
        c_noise= log(sigma / sigma_data) / 4
        c_skip = 1 / (1 + (sigma/sigma_data)^2)
        c_out  = sigma / sqrt(1 + (sigma/sigma_data)^2)
        D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise)
    """

    def __init__(self, c_s: int = 384, c_z: int = 128, c_token: int = 768,
                 n_blocks: int = 24, n_heads: int = 16, c_noise_embedding: int = 256):
        super().__init__()
        self.c_token = c_token
        self.fourier = FourierEmbedding(c_noise_embedding)
        self.norm_noise = nn.LayerNorm(c_noise_embedding)
        self.to_noise_s = nn.Linear(c_noise_embedding, c_s, bias=False)
        # Conditioning single: trunk single + noise embedding, as AF3's DiffusionConditioning does.
        self.norm_s = nn.LayerNorm(c_s)
        self.to_a_in = nn.Linear(3, c_token, bias=False)
        self.s_to_a = nn.Linear(c_s, c_token, bias=False)
        self.blocks = nn.ModuleList(
            DiffusionTransformerBlock(c_token, c_s, c_z, n_heads) for _ in range(n_blocks)
        )
        self.norm_out = nn.LayerNorm(c_token)
        # ⛔ NOT zero-initialised. A zero output projection is appealing (the head starts as the
        # pure EDM skip, D = c_skip * x) but it makes the backward through it `grad @ W.T = 0`, so
        # every upstream module gets exactly zero gradient until this layer moves off zero. The
        # same choice in the atom decoder was caught by a gradient-reach gate showing four
        # sub-modules at 0.000e+00. Protenix does not zero-init its equivalent output projection.
        # The EDM scaling c_out = sigma/sqrt(1+r^2) already keeps the early update small.
        self.to_coords = nn.Linear(c_token, 3, bias=False)

    def _f_forward(self, r_noisy, sigma, s, z, mask):
        """F_theta(c_in * x, c_noise(sigma)) -- the raw network, before EDM output scaling."""
        c_noise = torch.log(sigma / SIGMA_DATA) / 4.0
        n = self.to_noise_s(self.norm_noise(self.fourier(c_noise)))     # [B, c_s]
        s_cond = self.norm_s(s) + n[:, None, :]
        a = self.to_a_in(r_noisy) + self.s_to_a(s_cond)
        for blk in self.blocks:
            a = blk(a, s_cond, z, mask)
        return self.to_coords(self.norm_out(a))

    def denoise(self, x_noisy, sigma, s, z, mask):
        """One denoising step. This IS the training path (SI 3.7.1), not the rollout."""
        broadcast = sigma[:, None, None]
        r_noisy = x_noisy / torch.sqrt(SIGMA_DATA ** 2 + broadcast ** 2)
        r_update = self._f_forward(r_noisy, sigma, s, z, mask)
        s_ratio = broadcast / SIGMA_DATA
        return x_noisy / (1.0 + s_ratio ** 2) + r_update * broadcast / torch.sqrt(1.0 + s_ratio ** 2)

    @torch.no_grad()
    def rollout(self, s, z, mask, n_steps: int = MINI_ROLLOUT_STEPS, generator=None):
        """SI Alg. 18 sampler. Defaults to the 20-step MINI-ROLLOUT (SI 4.1), which is used to make
        coordinates for the confidence heads and carries NO gradients. Pass
        n_steps=FULL_INFERENCE_STEPS for real sampling.
        """
        B, L = mask.shape
        device = s.device
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        sigmas = noise_schedule(ts)
        x = sigmas[0] * torch.randn(B, L, 3, device=device, generator=generator)
        for i in range(n_steps):
            s_prev, s_cur = sigmas[i], sigmas[i + 1]
            # Per-step centre + random rigid augmentation (SI Alg. 19). Rotation is omitted here:
            # our loss is rigid-aligned anyway, and a CA-only token model has no chirality signal
            # that a random rotation would regularise. Translation kept.
            x = x - x.mean(dim=1, keepdim=True)
            x = x + S_TRANS * torch.randn(B, 1, 3, device=device, generator=generator)
            gamma = GAMMA_0 if s_cur > GAMMA_MIN else 0.0
            t_hat = s_prev * (gamma + 1.0)
            noise = NOISE_SCALE * torch.sqrt(t_hat ** 2 - s_prev ** 2) * torch.randn(
                x.shape, device=device, generator=generator
            )
            x_noisy = x + noise
            sig = t_hat.expand(B) if t_hat.dim() == 0 else t_hat
            x_denoised = self.denoise(x_noisy, sig, s, z, mask)
            d = (x_noisy - x_denoised) / t_hat
            x = x_noisy + STEP_SCALE * (s_cur - t_hat) * d
        return x * mask[..., None]


def weighted_rigid_align(x, x_gt, weights, mask):
    """Kabsch alignment of ground truth onto the prediction, weighted. SI Alg. 28.

    ⛔ AF3 aligns the GROUND TRUTH to the prediction, not the other way round, and detaches the
    result -- the alignment must not carry gradient into the model.
    """
    # ⛔ fp32 ONLY, and autocast must be disabled explicitly rather than relying on a .float() cast.
    # Under torch.autocast(bf16), autocast intercepts einsum/matmul and re-downcasts fp32 inputs, so
    # casting `cov` alone is not enough: the u @ vt einsum below comes back bf16 and
    # torch.linalg.det then dies with "lu_factor_cusolver not implemented for 'BFloat16'".
    # Kabsch is an SVD; it wants full precision regardless.
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf, gf = x.float(), x_gt.float()
        w = (weights * mask)[..., None].float()
        wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        centre = (xf * w).sum(dim=1, keepdim=True) / wsum
        xc = xf - centre
        gc = gf - (gf * w).sum(dim=1, keepdim=True) / wsum
        cov = torch.einsum("bni,bnj->bij", w * gc, xc)
        u, _, vt = torch.linalg.svd(cov)
        # Reflection guard: force det(R) = +1 so the alignment is a rotation, never a mirror.
        d = torch.sign(torch.linalg.det(torch.einsum("bij,bjk->bik", u, vt)))
        diag = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        rot = torch.einsum("bij,bjk,bkl->bil", u, diag, vt)
        out = torch.einsum("bni,bij->bnj", gc, rot) + centre
    return out.to(x.dtype).detach()


def smooth_lddt(x, x_gt, mask, cutoff: float = 15.0):
    """Smooth LDDT. SI Alg. 27. Returns a LOSS in [0,1] (1 - lddt), lower is better.

    Absent from this repo before now; AF2's bucketed lddt in openfold_stub is a different quantity.
    """
    d = torch.cdist(x, x)
    d_gt = torch.cdist(x_gt, x_gt)
    delta = (d - d_gt).abs()
    eps = 0.25 * (
        torch.sigmoid(0.5 - delta) + torch.sigmoid(1.0 - delta)
        + torch.sigmoid(2.0 - delta) + torch.sigmoid(4.0 - delta)
    )
    pair = mask[:, :, None] * mask[:, None, :]
    pair = pair * (d_gt < cutoff).to(pair.dtype)
    pair = pair * (1.0 - torch.eye(x.shape[1], device=x.device, dtype=pair.dtype))[None]
    return 1.0 - (eps * pair).sum(dim=(1, 2)) / pair.sum(dim=(1, 2)).clamp_min(1e-8)


def diffusion_loss(x_denoised, x_gt, sigma, mask, use_smooth_lddt: bool = True):  # noqa: C901
    """SI 3.7.1 Eqs. 2-6, CA-only.

    ⛔ EDM WEIGHT: the SI prints (t_hat + sigma_data)^2 as the denominator, but BOTH faithful
    reimplementations use EDM's (t_hat * sigma_data)^2 -- Protenix model/loss.py:1638-1640 and
    OpenFold3 core/loss/diffusion.py:579-581, the latter with an explicit "Changed from SI" comment.
    Karras et al. 2022 agrees. We follow the implementations, not the printed SI.

    The per-atom type weights w_l (alpha_dna=alpha_rna=5, alpha_ligand=10, SI Eq. 4) are all 1 here:
    this is a protein CA-only model, so every token is the base case.
    """
    # ⛔ THE WHOLE LOSS RUNS IN fp32, autocast explicitly disabled. Not a stylistic choice:
    #  - the EDM weight (sigma^2 + sd^2)/(sigma*sd)^2 spans ~5 orders of magnitude over the training
    #    noise distribution, and bf16 carries ~3 decimal digits;
    #  - smooth_lddt sums four sigmoids over an L^2 pair grid, where bf16 accumulation error grows
    #    with L;
    #  - weighted_rigid_align is an SVD, and torch.linalg.det has no bf16 CUDA kernel at all.
    # ⭐ Protenix reaches the same conclusion independently: configs_base.py:137-144 sets
    #   skip_amp = {"sample_diffusion": True, "sample_diffusion_training": True, "loss": True}
    # i.e. bf16 for the network, fp32 for the loss and for diffusion sampling. AF3's own
    # diffusion_head.py:250-267 likewise casts activations and both trunk conditioning tensors back
    # to float32 inside its bfloat16 context.
    with torch.autocast(device_type=x_denoised.device.type, enabled=False):
        x_denoised = x_denoised.float()
        x_gt = x_gt.float()
        sigma = sigma.float()
        mask = mask.float()
        weights = mask
        x_gt_aligned = weighted_rigid_align(x_denoised, x_gt, weights, mask)
        err = ((x_denoised - x_gt_aligned) ** 2).sum(-1)
        mse = (err * mask).sum(1) / mask.sum(1).clamp_min(1e-8) / 3.0   # 1/3 prefactor, SI Eq. 3
        w = (sigma ** 2 + SIGMA_DATA ** 2) / (sigma * SIGMA_DATA) ** 2
        loss = w * mse
        if use_smooth_lddt:
            # Added OUTSIDE the noise-level factor and unweighted (SI Eq. 6). On in initial
            # training, off from fine-tuning 1 onward (SI 5.2).
            loss = loss + smooth_lddt(x_denoised, x_gt_aligned, mask)
    return loss, {"mse": mse.detach(), "edm_weight": w.detach()}
