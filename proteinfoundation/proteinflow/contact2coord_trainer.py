"""LightningModule for the contact-to-coordinate all-atom diffusion model.

Standalone rather than folded into Proteina: this model shares none of the contact trunk's
conditioning machinery, and inheriting it would mean carrying flow-matching, topology references and
self-conditioning that have no meaning here.

⛔ Every hyperparameter is AF3's, cited inline. Nothing here is tuned by guess.
"""

import math
import os
from typing import Any, Dict

import lightning as L
import torch
import torch.nn.functional as F

from proteinfoundation.datasets.atom_features import N_REF_FEATS, atom14_features
from proteinfoundation.datasets.contact_augment import augment_contacts
from proteinfoundation.nn.af3_diffusion import FULL_INFERENCE_STEPS, diffusion_loss
from proteinfoundation.nn.contact2coord import ContactToCoord
from proteinfoundation.utils.c2c_dump import dump_sample

# AF3 SI §5.3 Eq. 15
ALPHA_DIFFUSION = 4.0
ALPHA_DISTOGRAM = 3e-2
# AF3 SI §5.4 / §5.6
BASE_LR = 1.8e-3
WARMUP_STEPS = 1000
DECAY_EVERY = 50_000
DECAY_FACTOR = 0.95
GRAD_CLIP = 10.0
# Weight EMA decay: Protenix `train_demo.sh --ema_decay 0.999`.
EMA_DECAY = 0.999
# Distogram bins, in ANGSTROM -- the unit batch["coords"] actually carries (measured: median
# consecutive CA-CA = 3.81 in a real batch, and residue_constants' ref_pos agrees at N-CA = 1.46).
# ⛔ These were 0.325/5.075, copied from the repo's experiment configs whose comments read
# "3.25A in nm". Against Angstrom data every real CA-CA distance (3.8 to ~66) lands past 5.075, so
# bucketize returned the overflow bin for essentially EVERY pair: a constant target, a head that
# learned to always predict bin 38, and a cross-entropy of 0.018 that looked like fast convergence
# and was actually a dead loss. Same physical boundaries the repo intends, correct unit.
DIST_MIN, DIST_MAX, DIST_BINS = 3.25, 50.75, 39


class ContactToCoordTrainer(L.LightningModule):
    def __init__(self, model_cfg: Dict[str, Any], aug_rate: float = 0.1,
                 aug_mode: str = "balanced", lr: float = BASE_LR,
                 dump_dir: str = None, n_dump: int = 2, ema_decay: float = EMA_DECAY):
        super().__init__()
        self.save_hyperparameters()
        self.model = ContactToCoord(**model_cfg, n_ref_feats=N_REF_FEATS)
        self.aug_rate, self.aug_mode, self.lr = aug_rate, aug_mode, lr
        self.dump_dir, self.n_dump = dump_dir, n_dump
        # ⛔ Weight EMA. Every AF3 replica with training code keeps one and VALIDATES ON IT;
        # we had neither, so every val number so far was measured on raw weights. Diffusion
        # models bounce hard late in training, which is exactly the shape we saw: val/diffusion
        # doubling while val/distogram (the trunk, read through z) stayed flat.
        # decay 0.999 = Protenix train_demo.sh --ema_decay 0.999.
        # eval-on-EMA + restore = OpenFold3 core/runners/model_runner.py:137 and :125.
        self.ema_decay = ema_decay
        self._ema = None
        self._cached = None

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
        """Gather the 14 dense slots out of the 37 sparse ones.

        ⛔ The constant is UPPERCASE. residue_constants carries both conventions -- lowercase
        `restype_atom14_mask` and `restype_atom14_rigid_group_positions` exist, but the atom14<->37
        index tables are only published as RESTYPE_ATOM14_TO_ATOM37 / RESTYPE_ATOM37_TO_ATOM14
        (protein_transformer.py:894-909 registers exactly these). Guessing the lowercase form
        raised AttributeError on the first real batch.
        ⭐ Gather, not scatter: protein_transformer.py:905 notes that scattering lets dummy indices
        overwrite N/CA/C, which is why OpenFold converts this direction by gather.
        """
        from proteinfoundation.openfold_stub.np import residue_constants as rc
        idx = torch.as_tensor(
            rc.RESTYPE_ATOM14_TO_ATOM37, device=coords37.device, dtype=torch.long
        )[aatype.long().clamp(0, 20)]                       # [B, L, 14]
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
        # x_gt_rep/atom_mask_rep are the structure repeated once per diffusion noise sample.
        dl, aux = diffusion_loss(out["x_denoised"], out["x_gt_rep"], out["sigma"],
                                 out["atom_mask_rep"])
        dg = self._distogram_loss(out["pair_logits"], b["atom_pos"], b["aatype"], b["mask"])
        loss = ALPHA_DIFFUSION * dl.mean() + ALPHA_DISTOGRAM * dg.mean()
        # ⭐ rmsd is the interpretable one: diffusion_loss builds mse as
        # sum_atoms||dx||^2 / n_atoms / 3 (SI Eq. 3's 1/3 prefactor), so RMSD = sqrt(3*mse) in
        # ANGSTROM. It is a DENOISING rmsd at the sampled noise level, not a generation rmsd.
        return loss, {"diffusion": dl.mean(), "distogram": dg.mean(),
                      "mse": aux["mse"].mean(), "sigma": out["sigma"].mean(),
                      "rmsd": (3.0 * aux["mse"]).sqrt().mean()}

    def training_step(self, batch, _):
        loss, logs = self._step(batch, True)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, prog_bar=False)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # ⛔ No augmentation at validation: the metric must describe the model on real contact maps,
        # not on corrupted ones, or it cannot be compared against anything.
        loss, logs = self._step(batch, False)
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        if self.dump_dir and batch_idx < self.n_dump and self.global_rank == 0:
            self._dump_structures(batch, batch_idx)
        return loss

    @torch.no_grad()
    def _dump_structures(self, batch, batch_idx):
        """Full-rollout sample -> PDB + distance matrices, so quality has a visual readout."""
        b = self._prepare(batch, train=False)
        s, z, _ = self.model.encode(b["contacts"], b["aatype"], b["mask"])
        coords = self.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                    b["atom_to_token"], b["atom_mask"],
                                    n_steps=FULL_INFERENCE_STEPS)
        L = b["mask"].shape[1]
        gen14 = coords.reshape(-1, L, 14, 3)[0]
        gt14 = b["atom_pos"].reshape(-1, L, 14, 3)[0]
        out_dir = os.path.join(self.dump_dir, f"step{self.global_step:07d}")
        name = f"val{batch_idx:02d}"
        mad = dump_sample(out_dir, name, gen14, gt14, b["aatype"][0], b["mask"][0],
                          b["contacts"][0])
        # Mean |d_gen - d_gt| over CA pairs: alignment-free, so a bad superposition cannot
        # flatter it, and directly comparable in Angstrom to the denoising rmsd.
        self.log("val/dist_mae_sampled", mad, sync_dist=False, rank_zero_only=True)

    # ── weight EMA ────────────────────────────────────────────────────────────────────────────
    def _ema_init(self):
        if self._ema is None:
            self._ema = {k: v.detach().clone().float()
                         for k, v in self.model.state_dict().items()}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # ⛔ Only on real optimizer steps. Updating every micro-batch would advance the EMA
        # accumulate_grad_batches times faster than intended and silently change its horizon.
        # (OpenFold3 guards this identically, model_runner.py:121-125.)
        acc = self.trainer.accumulate_grad_batches
        if (batch_idx + 1) % acc != 0 and not self.trainer.is_last_batch:
            return
        self._ema_init()
        d = self.ema_decay
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if self._ema[k].is_floating_point():
                    self._ema[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    self._ema[k].copy_(v.detach())      # ints (buffers) are not averaged

    def on_validation_start(self):
        """Swap in the EMA weights for validation, exactly as OpenFold3 does."""
        if self._ema is None or self._cached is not None:
            return
        self._cached = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict({k: v.to(self._cached[k].dtype) for k, v in self._ema.items()})

    def on_validation_end(self):
        if self._cached is not None:
            self.model.load_state_dict(self._cached)
            self._cached = None

    def on_save_checkpoint(self, checkpoint):
        # ⛔ Stored under "ema"/"params" deliberately: that is the key path every downstream
        # reader in this project expects, and an offline eval or warm start that reads
        # state_dict instead would silently score the UNAVERAGED model.
        if self._ema is not None:
            checkpoint["ema"] = {"params": {k: v.cpu() for k, v in self._ema.items()},
                                 "decay": self.ema_decay}

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self._ema = {k: v.clone().float() for k, v in checkpoint["ema"]["params"].items()}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.95), eps=1e-8)

        def lr_lambda(step):
            warm = min(1.0, (step + 1) / WARMUP_STEPS)      # AF3: 1000-step linear warmup
            return warm * (DECAY_FACTOR ** (step / DECAY_EVERY))   # then x0.95 every 5e4

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
