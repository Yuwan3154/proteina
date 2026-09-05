"""LightningModule for the contact-to-coordinate all-atom diffusion model.

Standalone rather than folded into Proteina: this model shares none of the contact trunk's
conditioning machinery, and inheriting it would mean carrying flow-matching, topology references and
self-conditioning that have no meaning here.

⛔ Every hyperparameter is AF3's, cited inline. Nothing here is tuned by guess.
"""

import math
from typing import Any, Dict

import lightning as L
import torch
import torch.nn.functional as F

from proteinfoundation.datasets.atom_features import N_REF_FEATS, atom14_features
from proteinfoundation.datasets.contact_augment import augment_contacts
from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.nn.contact2coord import ContactToCoord

# AF3 SI §5.3 Eq. 15
ALPHA_DIFFUSION = 4.0
ALPHA_DISTOGRAM = 3e-2
# AF3 SI §5.4 / §5.6
BASE_LR = 1.8e-3
WARMUP_STEPS = 1000
DECAY_EVERY = 50_000
DECAY_FACTOR = 0.95
GRAD_CLIP = 10.0
# Distogram bins: the repo's existing convention, matching loss.num_dist_buckets elsewhere.
DIST_MIN, DIST_MAX, DIST_BINS = 0.325, 5.075, 39


class ContactToCoordTrainer(L.LightningModule):
    def __init__(self, model_cfg: Dict[str, Any], aug_rate: float = 0.1,
                 aug_mode: str = "balanced", lr: float = BASE_LR):
        super().__init__()
        self.save_hyperparameters()
        self.model = ContactToCoord(**model_cfg, n_ref_feats=N_REF_FEATS)
        self.aug_rate, self.aug_mode, self.lr = aug_rate, aug_mode, lr

    # ── batch adaptation ──────────────────────────────────────────────────────────────────────
    def _prepare(self, batch, train: bool):
        mask = batch["mask_dict"]["coords"][..., 0, 0].float()
        aatype = batch["residue_type"].long()
        contacts = batch["contact_map"].float()
        if train and self.aug_rate > 0:
            contacts = augment_contacts(contacts, mask, self.aug_rate, self.aug_mode)

        ref_feats, ref_pos, a2t, amask = atom14_features(aatype, mask)
        # coords arrive as atom37; gather the atom14 slots so the target matches the model's layout.
        coords = batch["coords"].float()
        atom_pos = self._atom37_to_atom14(coords, aatype) if coords.shape[-2] == 37 else coords
        B, L, _, _ = atom_pos.shape
        return {
            "contacts": contacts, "aatype": aatype, "mask": mask,
            "ref_feats": ref_feats, "ref_pos": ref_pos,
            "atom_to_token": a2t, "atom_mask": amask,
            "atom_pos": atom_pos.reshape(B, L * 14, 3) * amask[..., None],
        }

    @staticmethod
    def _atom37_to_atom14(coords37, aatype):
        from proteinfoundation.openfold_stub.np import residue_constants as rc
        idx = torch.as_tensor(rc.restype_atom14_to_atom37, device=coords37.device)[
            aatype.long().clamp(0, 20)
        ]                                                   # [B, L, 14]
        return torch.gather(coords37, 2, idx[..., None].expand(-1, -1, -1, 3))

    # ── losses ────────────────────────────────────────────────────────────────────────────────
    def _distogram_loss(self, pair_logits, atom_pos, aatype, mask):
        """CE against binned CA-CA distances. AF3 SI Eq. 15 weights this at 3e-2."""
        B, L = mask.shape
        ca = atom_pos.reshape(B, L, 14, 3)[:, :, 1, :]      # atom14 slot 1 is CA
        d = torch.cdist(ca, ca)
        edges = torch.linspace(DIST_MIN, DIST_MAX, DIST_BINS - 1, device=d.device)
        tgt = torch.bucketize(d, edges)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        ce = F.cross_entropy(
            pair_logits.reshape(-1, DIST_BINS), tgt.reshape(-1), reduction="none"
        ).reshape(B, L, L)
        return (ce * pair_mask).sum((1, 2)) / pair_mask.sum((1, 2)).clamp_min(1.0)

    def _step(self, batch, train: bool):
        b = self._prepare(batch, train)
        out = self.model(b)
        dl, aux = diffusion_loss(out["x_denoised"], b["atom_pos"], out["sigma"], b["atom_mask"])
        dg = self._distogram_loss(out["pair_logits"], b["atom_pos"], b["aatype"], b["mask"])
        loss = ALPHA_DIFFUSION * dl.mean() + ALPHA_DISTOGRAM * dg.mean()
        return loss, {"diffusion": dl.mean(), "distogram": dg.mean(),
                      "mse": aux["mse"].mean(), "sigma": out["sigma"].mean()}

    def training_step(self, batch, _):
        loss, logs = self._step(batch, True)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, prog_bar=False)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        # ⛔ No augmentation at validation: the metric must describe the model on real contact maps,
        # not on corrupted ones, or it cannot be compared against anything.
        loss, logs = self._step(batch, False)
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.95), eps=1e-8)

        def lr_lambda(step):
            warm = min(1.0, (step + 1) / WARMUP_STEPS)      # AF3: 1000-step linear warmup
            return warm * (DECAY_FACTOR ** (step / DECAY_EVERY))   # then x0.95 every 5e4

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
