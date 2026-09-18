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
from proteinfoundation.nn.af3_diffusion import C2C_INFERENCE_STEPS, diffusion_loss
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
                 dump_dir: str = None, n_dump: int = 2, ema_decay: float = EMA_DECAY,
                 warmup_steps: int = WARMUP_STEPS, use_smooth_lddt: bool = True,
                 overfit_batch_path: str = None, w_chiral: float = 0.0,
                 w_fape: float = 0.0, fape_chunk: int = 0, fape_sigma_max: float = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = ContactToCoord(**model_cfg, n_ref_feats=N_REF_FEATS)
        self.aug_rate, self.aug_mode, self.lr = aug_rate, aug_mode, lr
        self.dump_dir, self.n_dump = dump_dir, n_dump
        # smooth_lddt is built from cdist alone, so it is exactly reflection-INVARIANT, and it enters
        # unweighted while the chiral MSE is scaled by the EDM weight, which collapses to ~1/sd^2 at
        # high sigma -- the noise levels where the hand is committed. Turning it off makes the
        # structure loss strictly chiral. AF3 itself drops it from fine-tuning 1 onward (SI 5.2).
        self.use_smooth_lddt = use_smooth_lddt
        # ⛔ DEFAULT 0.0 => the term is absent from the graph and every existing run is
        # byte-identical. There is no published value for this weight; it is set per experiment and
        # the chosen value must be recorded with the run, never inferred here.
        self.w_chiral = w_chiral
        # ⛔ DEFAULT 0.0 => absent from the graph, every existing run byte-identical. FAPE's own
        # constants (10 A clamp, 10 A length scale) ARE published (AF2 SI Alg. 28), but the weight
        # relative to THIS model's diffusion loss is not -- it is set per experiment and recorded.
        self.w_fape = w_fape
        # FAPE is O(L^2) per diffusion sample. At L=384 with n_diffusion_samples=48 the [S,L,L,3]
        # intermediates are ~1 GB each in fp32 before autograd saves them. Chunk over the SAMPLE
        # axis to bound that; 0 = one shot (fine at small L or small n_diff).
        self.fape_chunk = fape_chunk
        # ⛔ Grounded, not invented: SIGMA_DATA is the EDM scale at which signal and noise are equal,
        # and it is already a constant of this model. Measured gap (job 22876261) is +0.11 at the
        # 4-16 bin and collapses above it, so this is also where the data says the term stops
        # carrying handedness information. None = no gate (the pre-change behaviour).
        self.fape_sigma_max = fape_sigma_max
        # Overfit-one-structure mode: pin the FIRST training batch and reuse it for every train and
        # val step. Not Lightning's overfit_batches: that only swaps a RandomSampler for a
        # SequentialSampler, and our ClusterSampler would hand it a different chain every epoch.
        # The batch is saved to disk so a resume, and the measurement script, see the same chain.
        self.overfit_batch_path = overfit_batch_path
        self._pinned = None
        # ⛔ Denominated in STEPS, so its meaning changes with the batch. At 2048
        # pairs/step the reference's 1000 steps is 2.05M pairs of warmup and takes
        # 38.6 h; our earlier runs warmed over 128k pairs. Set it to keep the DATA
        # budget comparable, not the step count.
        self.warmup_steps = warmup_steps
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

        ref_feats, ref_pos, a2t, amask, ruid = atom14_features(aatype, mask)
        # coords arrive as atom37; gather the atom14 slots so the target matches the model's layout.
        coords = batch["coords"].float()
        atom_pos = self._atom37_to_atom14(coords, aatype) if coords.shape[-2] == 37 else coords
        B, L, _, _ = atom_pos.shape
        return {
            "contacts": contacts, "aatype": aatype, "mask": mask,
            "ref_feats": ref_feats, "ref_pos": ref_pos, "ref_space_uid": ruid,
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

    @staticmethod
    def _ca_sin_dihedral(ca, eps: float = 1e-8):
        """sin of the CA pseudo-dihedral over (i-1, i, i+1, i+2). ca: [B, L, 3] -> [B, L-3].

        ⭐ WHY sin AND NOT THE ANGLE. Under a reflection every dihedral flips sign, so
        `cos d` is EVEN (carries no handedness at all) and `sin d` is ODD. Training on `sin d`
        therefore supervises EXACTLY the degree of freedom that mirroring corrupts, and nothing else
        -- a mirrored structure has identical distances, identical contacts and identical |d|.
        ⛔ Computed directly from cross products rather than via atan2: atan2's gradient is unstable
        near the +-pi wrap, and we never need the angle itself.
        """
        b1 = ca[:, 1:-2] - ca[:, :-3]
        b2 = ca[:, 2:-1] - ca[:, 1:-2]
        b3 = ca[:, 3:] - ca[:, 2:-1]
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(eps)
        num = (torch.cross(n1, n2, dim=-1) * b2n).sum(-1)
        den = n1.norm(dim=-1).clamp_min(eps) * n2.norm(dim=-1).clamp_min(eps)
        return num / den.clamp_min(eps)

    def _chirality_loss(self, x_denoised, x_gt, atom_mask_rep, L):
        """Local, scale-free handedness supervision on the model's OWN output coordinates.

        ⛔⛔ NOT a prediction head on the trunk. The conditioning input is a CONTACT MAP, which is
        chirality-blind: a structure and its mirror have identical pairwise distances, so the correct
        enantiomer is formally UNIDENTIFIABLE from the input and a head on `s`/`z` could only ever
        learn the marginal prior that is already failing. Measuring the handedness of the GENERATED
        coordinates is identifiable, and that is what this does.

        ⭐ Why it can beat the plain MSE at the same job: the MSE is dominated by large-scale
        positional error (these samples sit at ~20 A RMSD), so the sign of a local dihedral is a
        vanishing fraction of it. This term is per-window and NORMALISED, so every residue
        contributes equally regardless of global error -- dense, scale-free chirality gradient.

        ⚠️ Honest caveat: this trains the SAME quantity the native-free mirror detector thresholds
        (`helix_pos_frac`). An improvement in that detector after adding this term is therefore NOT
        independent confirmation -- it is partly circular. Judge it on RMSD-after-reflection and on
        the GT `val/is_mirrored` instead.
        """
        ca_p = x_denoised.reshape(-1, L, 14, 3)[:, :, 1, :]
        ca_t = x_gt.reshape(-1, L, 14, 3)[:, :, 1, :]
        # ⛔ MASK, not shape. atom14 slot 1 is CA; a window is valid only if all FOUR of its
        # residues are real, otherwise padding zeros would contribute a meaningless dihedral.
        res_m = atom_mask_rep.reshape(-1, L, 14)[:, :, 1] > 0.5
        win_m = res_m[:, :-3] & res_m[:, 1:-2] & res_m[:, 2:-1] & res_m[:, 3:]
        sp = self._ca_sin_dihedral(ca_p)
        st = self._ca_sin_dihedral(ca_t)
        se = ((sp - st) ** 2) * win_m
        n = win_m.sum(-1).clamp_min(1)
        return se.sum(-1) / n, (sp.detach(), st.detach(), win_m)

    @staticmethod
    def _frames_from_backbone(n, ca, c, eps: float = 1e-8):
        """Gram-Schmidt residue frames from N/CA/C (AF2 SI Alg. 21, rigidFrom3Points).

        Returns (R, t) with R[..., :, k] = e_k, i.e. R maps LOCAL -> GLOBAL, so the global -> local
        map is R^T. t is the CA position.

        ⭐⭐ THIS IS WHERE THE CHIRALITY LIVES. e3 = e1 x e2 is a CROSS PRODUCT, so R is always a
        PROPER rotation (det = +1). Reflect the structure and the reflected frame is NOT the
        reflection of the frame -- the handedness cannot be absorbed into R. That is precisely why
        FAPE separates a structure from its mirror while any distance-only term cannot.
        """
        v1 = c - ca
        v2 = n - ca
        e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(eps)
        u2 = v2 - e1 * (e1 * v2).sum(-1, keepdim=True)
        e2 = u2 / u2.norm(dim=-1, keepdim=True).clamp_min(eps)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-1), ca

    def _fape_pairs(self, xp, xt, frame_m, atom_m, clamp: float, z: float):
        """FAPE over one chunk. xp/xt [S, L, 14, 3]; returns (sum_of_clamped_d, n_valid_pairs)."""
        Rp, tp = self._frames_from_backbone(xp[:, :, 0], xp[:, :, 1], xp[:, :, 2])
        Rt, tt = self._frames_from_backbone(xt[:, :, 0], xt[:, :, 1], xt[:, :, 2])
        cap, cat = xp[:, :, 1, :], xt[:, :, 1, :]
        # d[s, i, j] = x_j - t_i ; then local = R_i^T d  (R[..., c, k] = e_k component c)
        dp = cap[:, None, :, :] - tp[:, :, None, :]
        dt = cat[:, None, :, :] - tt[:, :, None, :]
        lp = torch.einsum("sick,sijc->sijk", Rp, dp)
        lt = torch.einsum("sick,sijc->sijk", Rt, dt)
        d = (lp - lt).norm(dim=-1)
        pair_m = frame_m[:, :, None] & atom_m[:, None, :]
        return (d.clamp(max=clamp) * pair_m).sum(), pair_m.sum()

    def _fape_loss(self, x_denoised, x_gt, atom_mask_rep, L,
                   clamp: float = 10.0, z: float = 10.0, sigma=None, sigma_max: float = None):
        """Backbone FAPE: every residue's CA expressed in every residue's frame (AF2 SI Alg. 28).

        ⛔ BACKBONE, not all-atom, and that is a memory decision not a modelling preference. All-atom
        FAPE is O(L x 14L): at L=384 with 48 diffusion samples the pair tensor alone is ~1.2 GB in
        fp32 before autograd saves anything. Backbone FAPE is O(L x L), 14x smaller, and it is the
        term AF2 uses to drive GLOBAL structure -- which is the thing that is mirrored here.

        ⛔ clamp = 10 A and z = 10 A are AF2's published constants (SI 1.9.2), not invented. The
        clamp is what makes FAPE care about getting the fold roughly right rather than chasing
        already-hopeless pairs; z just puts the loss in units of "fraction of 10 A".

        ⚠️ Frames come from PREDICTED N/CA/C, so early in training, when the backbone is noise, the
        frames are noisy too. That is a real effect (a 1 A coordinate error rotates a frame ~83 deg)
        and it is why this is a FINE-TUNE term applied to an already-folding model, not a
        from-scratch one. AF2 does not pay this cost the same way: its IPA emits frames directly.
        """
        xp = x_denoised.reshape(-1, L, 14, 3)
        xt = x_gt.reshape(-1, L, 14, 3)
        m = atom_mask_rep.reshape(-1, L, 14)
        # ⛔ MASK, not shape. A frame needs all three of N/CA/C real; a target atom needs CA real.
        frame_m = (m[:, :, 0] > 0.5) & (m[:, :, 1] > 0.5) & (m[:, :, 2] > 0.5)
        atom_m = m[:, :, 1] > 0.5

        # ⛔⛔ SIGMA GATE. MEASURED (job 22876261, 288 samples at tbeta step 8076): FAPE's mirror gap
        # is +0.5389 for sigma<1 and decays monotonically to +0.0001 by sigma>256, because at high
        # noise BOTH the right-handed and the mirrored structure sit against the clamp ceiling. The
        # t_beta(1.3,2.0) schedule puts ~75% of samples above sigma 256, so an UNGATED FAPE would be
        # three-quarters saturation noise carrying no handedness information at all.
        # ⛔ Tightening the clamp does NOT rescue this -- it makes it worse: the peak gap falls
        # 0.539 -> 0.206 -> 0.040 -> 0.009 as clamp goes 10 -> 5 -> 2 -> 1 A. Keep AF2's 10 A.
        if sigma is not None and sigma_max is not None:
            keep = sigma.reshape(-1) <= sigma_max
            if not bool(keep.any()):
                # No low-noise sample this step. Return a real zero that still carries a gradient
                # path, never a bare constant that would detach the term from the graph.
                return (xp.sum() * 0.0)
            xp, xt = xp[keep], xt[keep]
            frame_m, atom_m = frame_m[keep], atom_m[keep]

        S = xp.shape[0]
        step = self.fape_chunk if self.fape_chunk and self.fape_chunk > 0 else S
        tot = xp.new_zeros(())
        cnt = xp.new_zeros(())
        for a in range(0, S, step):
            b_ = slice(a, min(a + step, S))
            s_, n_ = self._fape_pairs(xp[b_], xt[b_], frame_m[b_], atom_m[b_], clamp, z)
            tot = tot + s_
            cnt = cnt + n_
        return tot / cnt.clamp_min(1) / z

    def _step(self, batch, train: bool):
        b = self._prepare(batch, train)
        out = self.model(b)
        # x_gt_rep/atom_mask_rep are the structure repeated once per diffusion noise sample.
        dl, aux = diffusion_loss(out["x_denoised"], out["x_gt_rep"], out["sigma"],
                                 out["atom_mask_rep"], use_smooth_lddt=self.use_smooth_lddt)
        dg = self._distogram_loss(out["pair_logits"], b["atom_pos"], b["aatype"], b["mask"])
        loss = ALPHA_DIFFUSION * dl.mean() + ALPHA_DISTOGRAM * dg.mean()
        # ⭐ rmsd is the interpretable one: diffusion_loss builds mse as
        # sum_atoms||dx||^2 / n_atoms / 3 (SI Eq. 3's 1/3 prefactor), so RMSD = sqrt(3*mse) in
        # ANGSTROM. It is a DENOISING rmsd at the sampled noise level, not a generation rmsd.
        metrics = {"diffusion": dl.mean(), "distogram": dg.mean(),
                   "mse": aux["mse"].mean(), "sigma": out["sigma"].mean(),
                   "rmsd": (3.0 * aux["mse"]).sqrt().mean()}
        if self.w_chiral > 0.0:
            L = int(b["mask"].shape[1])
            ch, (sp, st, wm) = self._chirality_loss(
                out["x_denoised"], out["x_gt_rep"], out["atom_mask_rep"], L)
            loss = loss + self.w_chiral * ch.mean()
            metrics["chiral"] = ch.mean()
            # ⭐ The READOUT that matters, logged even though it is not the loss: the fraction of
            # windows whose predicted handedness has the WRONG SIGN. A mirrored structure drives
            # this toward 1, a correct one toward 0, and unlike the loss it is interpretable and
            # directly comparable to the detector's helix_pos_frac.
            wrong = ((sp * st) < 0) & wm
            metrics["chiral_sign_wrong"] = wrong.sum() / wm.sum().clamp_min(1)
        if self.w_fape > 0.0:
            L = int(b["mask"].shape[1])
            fp = self._fape_loss(out["x_denoised"], out["x_gt_rep"], out["atom_mask_rep"], L,
                                 sigma=out["sigma"], sigma_max=self.fape_sigma_max)
            loss = loss + self.w_fape * fp
            metrics["fape"] = fp
            # ⭐ Log how many samples actually cleared the sigma gate. If this collapses toward 0
            # the term is silently absent and the run would look healthy while testing nothing.
            if self.fape_sigma_max is not None:
                metrics["fape_frac"] = (out["sigma"].reshape(-1)
                                        <= self.fape_sigma_max).float().mean()
        return loss, metrics

    def _pin(self, batch):
        if self.overfit_batch_path is None:
            return batch
        if self._pinned is None:
            if os.path.exists(self.overfit_batch_path):
                # torch_geometric Batch: .to() mutates in place and returns self.
                self._pinned = torch.load(self.overfit_batch_path,
                                          weights_only=False).to(self.device)
            else:
                self._pinned = batch
                # clone() first: PyG's .cpu() is in place and would move the live batch off-GPU.
                torch.save(batch.clone().cpu(), self.overfit_batch_path)
            L_ = int(self._pinned["mask_dict"]["coords"][..., 0, 0].sum())
            print(f"[overfit] pinned ONE structure (L={L_}) via {self.overfit_batch_path}",
                  flush=True)
        return self._pinned

    def training_step(self, batch, _):
        batch = self._pin(batch)
        loss, logs = self._step(batch, True)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, prog_bar=False)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # ⛔ No augmentation at validation: the metric must describe the model on real contact maps,
        # not on corrupted ones, or it cannot be compared against anything.
        batch = self._pin(batch)
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
                                    b["atom_to_token"], b["atom_mask"], b["ref_space_uid"],
                                    n_steps=C2C_INFERENCE_STEPS)
        L = b["mask"].shape[1]
        gen_all = coords.reshape(-1, L, 14, 3)
        gt_all = b["atom_pos"].reshape(-1, L, 14, 3)
        out_dir = os.path.join(self.dump_dir, f"step{self.global_step:07d}")
        # ⛔ Previously this scored ONLY entry [0] of the batch and discarded the rest, even though
        # `rollout` had already generated every one of them. At batch_size=1 that was the whole
        # batch so nothing was lost -- but in the 16-structure overfit it would have reported the
        # mirror rate of a SINGLE structure repeated, and any batch_size>1 run silently threw away
        # most of its own samples. Score them all; the handedness metrics then average over the
        # batch instead of resting on one draw.
        nb = int(gen_all.shape[0])
        mads, chirs, hands = [], [], []
        for j in range(nb):
            # Keep the historical name when there is one structure, so existing sample directories
            # and the offline detector's `*_gen.pdb` glob keep working unchanged.
            name = f"val{batch_idx:02d}" if nb == 1 else f"val{batch_idx:02d}_{j:02d}"
            m_j, c_j, h_j = dump_sample(out_dir, name, gen_all[j], gt_all[j], b["aatype"][j],
                                        b["mask"][j], b["contacts"][j])
            mads.append(m_j)
            chirs.append(c_j)
            if h_j:
                hands.append(h_j)
        mad = float(sum(mads) / max(len(mads), 1))
        chir = float(sum(chirs) / max(len(chirs), 1))
        # Average each handedness key over whichever structures produced it (a very short chain
        # yields no helix_pos_frac, so the keys are not all present on every structure).
        hand = {}
        for k in {k for h in hands for k in h}:
            vals = [h[k] for h in hands if k in h]
            hand[k] = float(sum(vals) / len(vals))
        hand["n_scored"] = float(len(hands))
        # Mean |d_gen - d_gt| over CA pairs: alignment-free, so a bad superposition cannot
        # flatter it, and directly comparable in Angstrom to the denoising rmsd.
        self.log("val/dist_mae_sampled", mad, sync_dist=False, rank_zero_only=True)
        # ⛔⛔ chirality_agree does NOT detect mirrors, despite what its name suggests. It tests
        # per-residue CA stereocentres, and a mirrored generation keeps its residues correctly L:
        # measured 0.999 (min 0.980) across 122 mirrored chains. It is logged only to confirm the
        # residues stay L; a drop here would mean D-amino acids, a different failure entirely.
        if chir == chir:                      # NaN check without importing math
            self.log("val/chirality_agree", chir, sync_dist=False, rank_zero_only=True)
        # ⭐ THESE are the metrics that see a mirror. helix_pos_frac ~0.12 is native-handed and
        # ~0.89 is mirrored (calibrated on 254 real structures); is_mirrored is the operational
        # verdict from the proper-vs-improper superposition gap. Watch val/is_mirrored: it is the
        # single number that says whether a fix is working, within one validation pass.
        for k, v in hand.items():
            self.log(f"val/{k}", float(v), sync_dist=False, rank_zero_only=True)

    # ── weight EMA ────────────────────────────────────────────────────────────────────────────
    def _ema_init(self):
        if self._ema is None:
            self._ema = {k: v.detach().clone().float()
                         for k, v in self.model.state_dict().items()}
            return
        # ⛔ On RESUME the EMA comes back from the checkpoint on CPU (on_save_checkpoint stores it
        # with .cpu() so the file is loadable without a GPU), while the model is on cuda. The
        # update below is IN-PLACE mul_/add_, which does not promote across devices -- it raises
        # "Expected all tensors to be on the same device". This killed two chain handoffs at
        # ~2 min each while the fresh run went 6 h clean, because the fresh path builds the EMA
        # from an already-on-device state_dict and never crosses devices.
        dev = next(self.model.parameters()).device
        if next(iter(self._ema.values())).device != dev:
            self._ema = {k: v.to(dev) for k, v in self._ema.items()}

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
            warm = min(1.0, (step + 1) / self.warmup_steps)
            return warm * (DECAY_FACTOR ** (step / DECAY_EVERY))   # then x0.95 every 5e4

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
