"""Structure head bolted onto the contact-map trunk.

Two extra triangle-multiplication blocks consume the trunk's final pair representation, a distogram
head reads their output, and a structure module turns (single, pair) into coordinates. The trunk's
representation is DETACHED at entry, so no structure gradient ever reaches the contact model.

⛔ The trunk has no 1D track -- that is deliberate in ContactMapTriSiT ("no 1D track at all"). The
single representation the structure module needs is therefore manufactured here, per the 2026-09-04
decision: masked row-pooling of the query pair block PLUS a projection of the raw residue embedding,
summed. Pooling carries the trunk's twelve blocks of reasoning; the embedding makes residue identity
explicit, which pooling alone would leave implicit.
"""

from typing import Optional

import torch
import torch.nn as nn

from proteinfoundation.nn.af3_diffusion import (
    AF3DiffusionHead,
    MINI_ROLLOUT_STEPS,
    sample_noise_level,
)
from proteinfoundation.nn.contact_map_tri import TriBlock


class SingleFromPair(nn.Module):
    """Manufacture s [B, L, c_s] from the query pair block and the residue embedding."""

    def __init__(self, dim: int, c_s: int, n_residue_types: int = 22):
        super().__init__()
        self.norm_pool = nn.LayerNorm(dim)
        self.pool_proj = nn.Linear(dim, c_s, bias=False)
        self.res_emb = nn.Embedding(n_residue_types, c_s)
        self.out_norm = nn.LayerNorm(c_s)

    def forward(self, z_q: torch.Tensor, mask: torch.Tensor,
                residue_type: Optional[torch.Tensor]) -> torch.Tensor:
        # Masked mean over valid columns j. Padded columns must not dilute the row average, which a
        # plain .mean(dim=2) would silently do -- at L=384 with a 120-residue chain that is a factor
        # of three.
        m = mask[:, None, :, None].to(z_q.dtype)
        pooled = (z_q * m).sum(dim=2) / m.sum(dim=2).clamp_min(1e-8)
        s = self.pool_proj(self.norm_pool(pooled))
        if residue_type is not None:
            s = s + self.res_emb(residue_type.long().clamp(min=0, max=self.res_emb.num_embeddings - 1))
        return self.out_norm(s) * mask[..., None].to(s.dtype)


class StructureHead(nn.Module):
    """Two TriBlocks + distogram head + a structure module, on a detached trunk representation.

    mode="diffusion" -> AF3-style token-level diffusion predicting CA coordinates
    mode="ipa"       -> AF2 StructureModule (built lazily; it needs aatype and emits atom14)
    """

    def __init__(
        self,
        dim: int,
        tri_hidden: int,
        transition_n: int,
        dim_cond: int,
        mode: str = "diffusion",
        c_s: int = 384,
        c_z: int = 128,
        n_blocks_tri: int = 2,
        num_dist_buckets: int = 39,
        n_residue_types: int = 22,
        diffusion: Optional[dict] = None,
        structure_module_cfg: Optional[dict] = None,
    ):
        super().__init__()
        assert mode in ("diffusion", "ipa"), mode
        self.mode = mode
        self.dim = dim

        self.tri_blocks = nn.ModuleList(
            TriBlock(dim, tri_hidden, transition_n, dim_cond) for _ in range(n_blocks_tri)
        )
        # Distogram head at the END of the extra blocks, per the directive -- a second, independent
        # distogram from the trunk's own head, reading a representation the trunk never saw.
        self.dist_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_dist_buckets))

        self.single_from_pair = SingleFromPair(dim, c_s, n_residue_types)
        self.pair_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, c_z, bias=False))

        if mode == "diffusion":
            self.structure = AF3DiffusionHead(c_s=c_s, c_z=c_z, **(diffusion or {}))
        else:
            from proteinfoundation.openfold_stub.model.structure_module import StructureModule
            cfg = dict(structure_module_cfg or {})
            cfg.setdefault("c_ipa", 16)
            cfg.setdefault("c_resnet", 128)
            cfg.setdefault("no_heads_ipa", 12)
            cfg.setdefault("no_qk_points", 4)
            cfg.setdefault("no_v_points", 8)
            cfg.setdefault("dropout_rate", 0.1)
            cfg.setdefault("no_blocks", 8)
            cfg.setdefault("no_transition_layers", 1)
            cfg.setdefault("no_resnet_blocks", 2)
            cfg.setdefault("no_angles", 7)
            cfg.setdefault("trans_scale_factor", 10)
            cfg.setdefault("epsilon", 1e-12)
            cfg.setdefault("inf", 1e5)
            cfg["c_s"], cfg["c_z"] = c_s, c_z
            self.structure = StructureModule(**cfg)

    def features(self, z_full, L, mask, cond, residue_type):
        """Detach, take the query block, refine, and produce (s, z, distogram logits)."""
        # ⛔ THE detach. One line, and it is the entire gradient-isolation guarantee: everything
        # downstream is a separate computation graph rooted here.
        z = z_full.detach()[:, :L, :L]
        pair_mask = (mask[:, :, None] * mask[:, None, :]).to(z.dtype)
        for blk in self.tri_blocks:
            z = blk(z, pair_mask, cond)
        pair_logits = self.dist_head(z)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))
        pair_logits = pair_logits * pair_mask[..., None]
        s = self.single_from_pair(z, mask, residue_type)
        return s, self.pair_proj(z), pair_logits

    def forward(self, z_full, L, mask, cond, residue_type=None, x_gt=None,
                run_rollout: bool = False):
        s, z_s, pair_logits = self.features(z_full, L, mask, cond, residue_type)
        out = {"structure_pair_logits": pair_logits}

        if self.mode == "diffusion":
            if x_gt is not None:
                # Training: single-step denoising of noised ground truth (SI 3.7.1). NOT the rollout.
                sigma = sample_noise_level((x_gt.shape[0],), x_gt.device, x_gt.dtype)
                noise = torch.randn_like(x_gt) * sigma[:, None, None]
                x_denoised = self.structure.denoise(x_gt + noise, sigma, s, z_s, mask)
                out["x_denoised"] = x_denoised * mask[..., None]
                out["sigma"] = sigma
            if run_rollout or x_gt is None:
                # 20 steps, no_grad (SI 4.1) -- coordinates for the confidence heads, and the only
                # path available at inference time.
                out["coords_rollout"] = self.structure.rollout(
                    s, z_s, mask, n_steps=MINI_ROLLOUT_STEPS
                )
        else:
            assert residue_type is not None, "IPA StructureModule requires aatype"
            # ⛔ clamp BEFORE the call: StructureModule indexes residue-constant buffers of size 21
            # with aatype, so a 21 (mask token) is an out-of-bounds read. protein_transformer.py
            # clamps only afterwards at :1261, which is a latent bug there.
            aatype = residue_type.long().clamp(0, 20)
            sm = self.structure({"single": s, "pair": z_s}, aatype=aatype, mask=mask)
            out["positions_atom14"] = sm["positions"][-1]
            out["frames"] = sm["frames"][-1]
        return out
