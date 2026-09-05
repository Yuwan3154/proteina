"""Contact-to-coordinate all-atom diffusion model.

A STANDALONE model: contact map + sequence -> all-atom coordinates. It does not consume the tri
trunk's hidden state, only its OUTPUT, which is the point -- the interface is the contact map, so
this model does not need retraining for each new version of the contact model.

    contact map C [B,L,L] + aatype [B,L]
        |
        +-- pair init   : contact + relative position -> z [B,L,L,128]
        +-- single init : residue embedding           -> s [B,L,384]
        |
        +-- 2 x TriBlock on z          (triangle multiplication; the contact map's own geometry)
        +-- distogram head on z        (auxiliary, AF3 alpha_distogram = 3e-2)
        |
        +-- AF3 DiffusionModule
              AtomAttentionEncoder (3 blk, local 32/128)
           -> DiffusionTransformer (24 blk, c_token 768, 16 heads)
           -> AtomAttentionDecoder (3 blk)
           -> per-atom coordinates

All widths are AF3's, per the user directive: c_s 384, c_z 128, c_token 768, c_atom 128,
c_atompair 16. Depth is AF3's 24 blocks. Weights are randomly initialised -- no warm start.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from proteinfoundation.nn.af3_diffusion import (
    SIGMA_DATA,
    DiffusionTransformerBlock,
    FourierEmbedding,
    MINI_ROLLOUT_STEPS,
    noise_schedule,
    sample_noise_level,
)
from proteinfoundation.nn.atom_attention import AtomAttentionDecoder, AtomAttentionEncoder
from proteinfoundation.nn.contact_map_tri import TriBlock

MAX_REL_POS = 32   # AF3 SI DiffusionConditioning max_relative_idx


class ContactToCoord(nn.Module):
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_token: int = 768,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 24,
        n_heads: int = 16,
        n_tri_blocks: int = 2,
        tri_hidden: int = 128,
        transition_n: int = 2,
        atom_blocks: int = 3,
        atom_heads: int = 4,
        n_residue_types: int = 22,
        num_dist_buckets: int = 39,
        n_ref_feats: int = 8,
        c_noise_embedding: int = 256,
        n_diffusion_samples: int = 48,
    ):
        super().__init__()
        self.c_s, self.c_z, self.c_token, self.c_atom = c_s, c_z, c_token, c_atom
        # AF3's diffusion mini-batch (SI Alg. 20); Protenix ships 48 (configs_base.py:122).
        self.n_diffusion_samples = n_diffusion_samples

        # ── inputs ────────────────────────────────────────────────────────────────────────────
        # The contact map enters as a 2-way embedding rather than a scalar: a contact and a
        # non-contact are categories, not two ends of a continuum, and a scalar would impose a
        # spurious ordering on {0,1} while giving the model no way to represent "absent".
        self.contact_emb = nn.Embedding(2, c_z)
        self.rel_pos_emb = nn.Embedding(2 * MAX_REL_POS + 2, c_z)
        self.seq_emb = nn.Embedding(n_residue_types, c_s)
        self.s_to_z = nn.Linear(c_s, 2 * c_z, bias=False)   # outer sum, both directions

        self.tri_blocks = nn.ModuleList(
            TriBlock(c_z, tri_hidden, transition_n, c_s) for _ in range(n_tri_blocks)
        )
        self.dist_head = nn.Sequential(nn.LayerNorm(c_z), nn.Linear(c_z, num_dist_buckets))

        # ── AF3 diffusion ─────────────────────────────────────────────────────────────────────
        self.fourier = FourierEmbedding(c_noise_embedding)
        self.norm_noise = nn.LayerNorm(c_noise_embedding)
        self.to_noise_s = nn.Linear(c_noise_embedding, c_s, bias=False)
        self.norm_s = nn.LayerNorm(c_s)

        self.atom_enc = AtomAttentionEncoder(
            c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, c_s=c_s, c_z=c_z,
            n_blocks=atom_blocks, n_heads=atom_heads, n_ref_feats=n_ref_feats, has_coords=True,
        )
        self.blocks = nn.ModuleList(
            DiffusionTransformerBlock(c_token, c_s, c_z, n_heads) for _ in range(n_blocks)
        )
        self.atom_dec = AtomAttentionDecoder(
            c_atom=c_atom, c_atompair=c_atompair, c_token=c_token,
            n_blocks=atom_blocks, n_heads=atom_heads,
        )
        self.pair_to_atompair = nn.Linear(c_z, c_atompair, bias=False)

    # ── trunk: contact map -> (s, z) ──────────────────────────────────────────────────────────
    def encode(self, contacts, aatype, mask):
        B, L = aatype.shape
        pair_mask = (mask[:, :, None] * mask[:, None, :])
        idx = torch.arange(L, device=aatype.device)
        rel = (idx[None, :] - idx[:, None]).clamp(-MAX_REL_POS, MAX_REL_POS) + MAX_REL_POS
        z = self.rel_pos_emb(rel)[None].expand(B, -1, -1, -1).clone()
        z = z + self.contact_emb((contacts > 0.5).long())

        s = self.seq_emb(aatype.long().clamp(min=0, max=self.seq_emb.num_embeddings - 1))
        a, b = self.s_to_z(s).chunk(2, dim=-1)
        z = z + a[:, :, None, :] + b[:, None, :, :]
        z = z * pair_mask[..., None]

        # TriBlock's conditioning input is a per-sample vector; the contact map carries no
        # diffusion time of its own, so condition on the mean sequence embedding. This keeps the
        # block signature unchanged rather than forking it.
        cond = (s * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        for blk in self.tri_blocks:
            z = blk(z, pair_mask, cond)

        pair_logits = self.dist_head(z)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2)) * pair_mask[..., None]
        return s, z, pair_logits

    # ── EDM-preconditioned denoiser over ATOMS ────────────────────────────────────────────────
    def _f_forward(self, r_noisy, sigma, s, z, mask, ref_feats, ref_pos, atom_to_token, atom_mask):
        c_noise = torch.log(sigma / SIGMA_DATA) / 4.0
        n = self.to_noise_s(self.norm_noise(self.fourier(c_noise)))
        s_cond = self.norm_s(s) + n[:, None, :]

        a_token, q_atom = self.atom_enc(
            ref_feats, ref_pos, atom_to_token, s_cond, z, atom_mask, noisy_pos=r_noisy
        )
        for blk in self.blocks:
            a_token = blk(a_token, s_cond, z, mask)

        # The decoder blocks the token pair itself; densifying it to [B,A,A,c] here was 925 MB at
        # L=384 and is exactly what the blocked layout exists to avoid.
        return self.atom_dec(
            a_token, q_atom, atom_to_token, atom_mask, self.pair_to_atompair(z)
        )

    def denoise(self, x_noisy, sigma, s, z, mask, ref_feats, ref_pos, atom_to_token, atom_mask):
        b = sigma[:, None, None]
        r_noisy = x_noisy / torch.sqrt(SIGMA_DATA ** 2 + b ** 2)
        upd = self._f_forward(r_noisy, sigma, s, z, mask, ref_feats, ref_pos, atom_to_token, atom_mask)
        ratio = b / SIGMA_DATA
        return x_noisy / (1.0 + ratio ** 2) + upd * b / torch.sqrt(1.0 + ratio ** 2)

    def forward(self, batch: Dict[str, torch.Tensor], run_rollout: bool = False):
        contacts, aatype, mask = batch["contacts"], batch["aatype"], batch["mask"]
        ref_feats, ref_pos = batch["ref_feats"], batch["ref_pos"]
        atom_to_token, atom_mask = batch["atom_to_token"], batch["atom_mask"]
        s, z, pair_logits = self.encode(contacts, aatype, mask)
        out = {"pair_logits": pair_logits}

        x_gt = batch.get("atom_pos")
        if x_gt is not None:
            # Single-step denoising of noised ground truth -- SI 3.7.1. NOT the mini-rollout.
            # ⛔ Each structure is expanded into n_diffusion_samples INDEPENDENTLY noised copies
            # that share one trunk pass. This is AF3's diffusion mini-batch (SI Alg. 20) and it is
            # not optional decoration: Protenix ships diffusion_batch_size=48
            # (configs_base.py:122, consumed at protenix.py:802 as N_sample, producing
            # "[..., N_sample=48, N_atom, 3]"), and the published lr=1.8e-3 is calibrated for a
            # gradient averaged over that many noise draws. Training with one draw per structure
            # makes the diffusion gradient ~sqrt(48) noisier at the same lr -- which is exactly
            # how the first real run diverged (val/loss 3.82 -> 5.99 -> 7.10 at steps 500/1000/
            # 1500, turning the moment the 1000-step warmup handed over full lr).
            B, A, _ = x_gt.shape
            n = self.n_diffusion_samples
            sigma = sample_noise_level((B, n), x_gt.device, x_gt.dtype)      # [B, n]
            x_rep = x_gt[:, None].expand(B, n, A, 3).reshape(B * n, A, 3)
            sig_flat = sigma.reshape(B * n)
            x_noisy = x_rep + torch.randn_like(x_rep) * sig_flat[:, None, None]
            # The trunk runs ONCE; only the diffusion module sees the expanded batch.
            rep = lambda t: t.repeat_interleave(n, dim=0)
            out["x_denoised"] = self.denoise(
                x_noisy, sig_flat, rep(s), rep(z), rep(mask), rep(ref_feats), rep(ref_pos),
                rep(atom_to_token), rep(atom_mask)
            ) * rep(atom_mask)[..., None]
            out["atom_mask_rep"] = rep(atom_mask)
            out["x_gt_rep"] = x_rep
            out["sigma"] = sig_flat
        if run_rollout or x_gt is None:
            out["coords"] = self.rollout(s, z, mask, ref_feats, ref_pos, atom_to_token, atom_mask)
        return out

    @torch.no_grad()
    def rollout(self, s, z, mask, ref_feats, ref_pos, atom_to_token, atom_mask,
                n_steps: int = MINI_ROLLOUT_STEPS):
        """SI Alg. 18. Defaults to the 20-step mini-rollout; pass 200 for full inference."""
        B, A = atom_mask.shape
        dev = s.device
        # ⛔ EVERY step is masked, and the centring uses a MASKED mean. Padding is not a small
        # detail here: at L=224 in a 384-padded batch, 42% of the A=5376 atom slots are padding.
        # The previous version centred with `x.mean(dim=1)` over ALL slots and never masked the
        # denoiser output inside the loop, so noise on padded slots pulled the centre of mass every
        # step and fed back through x. Only the FINAL return was masked, which hides it from shape
        # checks while corrupting the trajectory. Symptom: sampled Rg wandering between 0.20x and
        # 8.76x of native across checkpoints while the distance-matrix correlation stayed at
        # 0.90-0.98 -- right shape, uncontrolled scale, which is exactly what a drifting centre does.
        m = atom_mask[..., None]
        nreal = atom_mask.sum(dim=1, keepdim=True).clamp_min(1.0)[..., None]
        sig = noise_schedule(torch.linspace(0.0, 1.0, n_steps + 1, device=dev))
        x = sig[0] * torch.randn(B, A, 3, device=dev) * m
        for i in range(n_steps):
            s_prev, s_cur = sig[i], sig[i + 1]
            x = (x - (x * m).sum(dim=1, keepdim=True) / nreal) * m
            gamma = 0.8 if s_cur > 1.0 else 0.0
            t_hat = s_prev * (gamma + 1.0)
            x_noisy = x + 1.003 * torch.sqrt((t_hat ** 2 - s_prev ** 2).clamp_min(0)) \
                * torch.randn_like(x) * m
            d = self.denoise(x_noisy, t_hat.expand(B), s, z, mask,
                             ref_feats, ref_pos, atom_to_token, atom_mask) * m
            x = (x_noisy + 1.5 * (s_cur - t_hat) * (x_noisy - d) / t_hat) * m
        return x * m
