# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import io
import math
import os
import random
import re
import tempfile

from abc import abstractmethod
from functools import partial
from typing import List, Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Bool, Float
from loguru import logger
from PIL import Image
from torch import Dict, Tensor

from proteinfoundation.utils.ff_utils.pdb_utils import (
    mask_cath_code_by_level,
    mask_seq,
    write_prot_to_pdb,
)


class ModelTrainerBase(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None):
        super(ModelTrainerBase, self).__init__()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only used for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        self.metric_factory = None

        # Scaling laws stuff
        self.nflops = 0
        self.nparams = None
        self.nsamples_processed = 0

        # Attributes re-written by classes that inherit from this one
        self.nn = None
        self.fm = None

        # For autoguidance, overridden in `self.configure_inference`
        self.nn_ag = None
        self.motif_conditioning = cfg_exp.training.get("motif_conditioning", False)

    def configure_optimizers(self):
        if self.cfg_exp.training.finetune_seq_cond_lora_only:
            opt_params = []
            opt_params_names = []
            for name, param in self.named_parameters():
                if "residue_type" in name or "lora" in name:
                    param.requires_grad = True
                    opt_params.append(param)
                    opt_params_names.append(name)
                else:
                    param.requires_grad = False
            optimizer = torch.optim.Adam(
                opt_params, lr=self.cfg_exp.opt.lr
            )
            print(f"Finetuning {opt_params_names}")
        else:
            for param in self.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.cfg_exp.opt.lr
            )
        
        # Check if learning rate warmup is enabled
        if self.cfg_exp.opt.get("use_lr_warmup", False):
            warmup_steps = self.cfg_exp.opt.get("warmup_steps", 1000)
            warmup_start_lr_ratio = self.cfg_exp.opt.get("warmup_start_lr_ratio", 0.0)
            scheduler_type = self.cfg_exp.opt.get("lr_scheduler_type", "constant")
            
            # Calculate total training steps dynamically from max_epochs and dataloader
            total_steps = None
            if scheduler_type in ["linear_decay", "cosine"]:
                # Try to get dataloader length and max_epochs
                if hasattr(self, 'trainer') and self.trainer is not None:
                    if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                        train_dataloader = self.trainer.datamodule.train_dataloader()
                        steps_per_epoch = len(train_dataloader)
                        max_epochs = self.trainer.max_epochs
                        accumulate_grad_batches = self.cfg_exp.opt.get("accumulate_grad_batches", 1)
                        # Total steps = (steps_per_epoch / accumulate_grad_batches) * max_epochs
                        total_steps = int((steps_per_epoch / accumulate_grad_batches) * max_epochs)
                        logger.info(f"Calculated total_training_steps = {total_steps} "
                                    f"(steps_per_epoch={steps_per_epoch}, max_epochs={max_epochs}, "
                                    f"accumulate_grad_batches={accumulate_grad_batches})")

                # Fallback to config value or default
                if total_steps is None:
                    total_steps = self.cfg_exp.opt.get("total_training_steps", 100000)
                    logger.warning(f"Using fallback total_training_steps = {total_steps}")
            
            # Create warmup function
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup from warmup_start_lr_ratio to 1.0
                    return warmup_start_lr_ratio + (1.0 - warmup_start_lr_ratio) * (current_step / warmup_steps)
                else:
                    # After warmup, apply the specified scheduler
                    if scheduler_type == "constant":
                        return 1.0
                    elif scheduler_type == "linear_decay":
                        # Linear decay from 1.0 to 0.0 over remaining steps
                        remaining_steps = total_steps - warmup_steps
                        decay_step = current_step - warmup_steps
                        return max(0.0, 1.0 - (decay_step / remaining_steps))
                    elif scheduler_type == "cosine":
                        # Cosine annealing after warmup
                        remaining_steps = total_steps - warmup_steps
                        decay_step = current_step - warmup_steps
                        return 0.5 * (1.0 + np.cos(np.pi * decay_step / remaining_steps))
                    else:
                        return 1.0
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Update learning rate every step
                    "frequency": 1,
                }
            }
        
        return optimizer

    def _nn_out_to_x_clean(self, nn_out, batch):
        """
        Transforms the output of the nn to a clean sample prediction. The transformation depends on the
        parameterization used. For now we admit x_1 or v.

        Args:
            nn_out: Dictionary, nerual network output
                - "coords_pred": Tensor of shape [b, n, 3], could be the clean sample or the velocity
                - "pair_pred" (Optional): Tensor of shape [b, n, n, num_buckets_predict_pair], could be the clean sample or the velocity
            batch: Dictionary, batch of data

        Returns:
            Clean sample prediction, tensor of shape [b, n, 3].
        """
        if "coors_pred" not in nn_out:
            return None

        nn_pred = nn_out["coors_pred"]
        t = batch["t"]  # [*]
        t_ext = t[..., None, None]  # [*, 1, 1]
        x_t = batch["x_t"]  # [*, n, 3]
        if self.cfg_exp.model.target_pred == "x_1":
            x_1_pred = nn_pred
        elif self.cfg_exp.model.target_pred == "v":
            x_1_pred = x_t + (1.0 - t_ext) * nn_pred
        else:
            raise IOError(
                f"Wrong parameterization chosen: {self.cfg_exp.model.target_pred}"
            )
        return x_1_pred

    def _nn_out_to_c_clean(self, nn_out: Dict, batch: Dict):
        """
        Converts nn output to clean contact map prediction.

        For contact maps, we always predict the clean contact map directly (x_1 parameterization).
        This mirrors _nn_out_to_x_clean for coordinates.

        Args:
            nn_out: Dictionary containing network outputs, expected to have:
                - "contact_map_pred": Tensor of shape [b, n, n], clean contact map prediction
            batch: Dictionary, batch of data (unused, for API consistency)

        Returns:
            Clean contact map prediction, tensor of shape [b, n, n], or None if not present.
        """
        if "contact_map_pred" not in nn_out:
            return None
        return nn_out["contact_map_pred"]

    def predict_clean(
        self,
        batch: Dict,
    ):
        """
        Predicts clean samples given noisy ones and time.

        Args:
            batch: a batch of data with some additions, including
                - "x_t": Type depends on the mode (see beluw, "returns" part)
                - "t": Time, shape [*]
                - "mask": Binary mask of shape [*, n]
                - "x_sc" (optional): Prediction for self-conditioning
                - Other features from the dataloader.

        Returns:
            Predicted clean sample, depends on the "modality" we're in.
                - For frameflow it returns a dictionary with keys "trans" and "rot", and values
                tensors of shape [*, n, 3] and [*, n, 3, 3] respectively,
                - For CAflow it returns a tensor of shape [*, n, 3].
            Other things predicted by nn (pair_pred for distogram loss)
        """
        nn_out = self.nn(batch)  # [*, n, 3]
        return self._nn_out_to_x_clean(nn_out, batch), nn_out  # [*, n, 3]

    def predict_clean_n_v_w_guidance(
        self,
        batch: Dict,
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
    ):
        """
        Logic for CFG and autoguidance goes here. This computes a clean sample prediction (can be single thing, tuple, etc)
        and the corresponding vector field used to initialize.

        Here if we want to do the different self conditioning for cond / ucond, ag / no ag, we can just return tuples of x_pred and
        modify the batches accordingly every time we call predict clean.

        w: guidance weight
        alpha: autoguidance ratio
        x_pred = w * x_pred + (1 - alpha) * (1 - w) * x_pred_uncond + alpha * (1 - w) * x_pred_auto_guidance

        WARNING: The ag checkpoint needs to rely on the same parameterization of the main model. This can be changed after training
        so no big deal but just in case leaving a note.

        Returns:
            Dictionary with keys:
                - "coords": Predicted clean coordinates, shape [*, n, 3]
                - "v": Velocity field for coordinates (only in coord diffusion mode), shape [*, n, 3]
                - "contact_map": Predicted clean contact map (if available), shape [*, n, n]
                - "contact_map_v": Velocity field for contact map (only in contact map mode), shape [*, n, n]
                - "distogram": Distogram logits (if available), shape [*, n, n, num_buckets]
        """
        if self.motif_conditioning and ("fixed_structure_mask" not in batch or "x_motif" not in batch):
            batch.update(self.motif_factory(batch, zeroes = True))  # for generation we have to pass conditioning info in. But for validation do the same as training

        nn_out = self.nn(batch)
        x_pred = self._nn_out_to_x_clean(nn_out, batch)
        c_pred = self._nn_out_to_c_clean(nn_out, batch)
        
        contact_map_mode = getattr(self, "contact_map_mode", False)

        # Apply CFG/autoguidance if needed
        if guidance_weight != 1.0:
            assert autoguidance_ratio >= 0.0 and autoguidance_ratio <= 1.0
            
            # Get autoguidance predictions
            nn_out_ag = None
            if autoguidance_ratio > 0.0:
                nn_out_ag = self.nn_ag(batch)
                x_pred_ag = self._nn_out_to_x_clean(nn_out_ag, batch) if x_pred is not None else None
                c_pred_ag = self._nn_out_to_c_clean(nn_out_ag, batch) if c_pred is not None else None
            else:
                x_pred_ag = torch.zeros_like(x_pred) if x_pred is not None else None
                c_pred_ag = torch.zeros_like(c_pred) if c_pred is not None else None

            # Get unconditional predictions (CFG)
            nn_out_uncond = None
            if autoguidance_ratio < 1.0:
                assert (
                    "cath_code" in batch
                ), "Only support CFG when cath_code is provided"
                uncond_batch = batch.copy()
                uncond_batch.pop("cath_code")
                nn_out_uncond = self.nn(uncond_batch)
                x_pred_uncond = self._nn_out_to_x_clean(nn_out_uncond, uncond_batch) if x_pred is not None else None
                c_pred_uncond = self._nn_out_to_c_clean(nn_out_uncond, uncond_batch) if c_pred is not None else None
            else:
                x_pred_uncond = torch.zeros_like(x_pred) if x_pred is not None else None
                c_pred_uncond = torch.zeros_like(c_pred) if c_pred is not None else None

            # Apply guidance to coordinates
            if x_pred is not None:
                x_pred = guidance_weight * x_pred + (1 - guidance_weight) * (
                    autoguidance_ratio * x_pred_ag
                    + (1 - autoguidance_ratio) * x_pred_uncond
                )
            
            # Apply guidance to contact map
            if c_pred is not None:
                c_pred = guidance_weight * c_pred + (1 - guidance_weight) * (
                    autoguidance_ratio * c_pred_ag
                    + (1 - autoguidance_ratio) * c_pred_uncond
                )

        result = {}

        # Coordinates: always include if available
        if x_pred is not None:
            result["coords"] = x_pred
            # Only compute velocity in coordinate diffusion mode (not contact map mode)
            if not contact_map_mode and "x_t" in batch:
                result["v"] = self.fm.xt_dot(
                    x_pred,
                    batch["x_t"],
                    batch["t"],
                    batch["mask"],
                    modality="coordinates",
                )

        # Contact map: always include if available
        if c_pred is not None:
            result["contact_map"] = c_pred
            # Only compute velocity in contact map diffusion mode
            if contact_map_mode and "contact_map_t" in batch:
                result["contact_map_v"] = self.fm.xt_dot(
                    c_pred,
                    batch["contact_map_t"],
                    batch["t"],
                    batch["mask"],
                    modality="contact_map",
                )
        
        # Include distogram if available
        if "pair_pred" in nn_out:
            result["distogram"] = nn_out["pair_pred"]

        return result

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    @abstractmethod
    def align_wrapper(self, x_0, x_1, mask):
        """Performs Kabsch on the CAs of x_0 and x_1."""

    @abstractmethod
    def extract_clean_sample(self, batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype)
        """

    def sample_t(self, shape):
        if self.cfg_exp.loss.t_distribution.name == "uniform":
            t_max = self.cfg_exp.loss.t_distribution.p2
            return torch.rand(shape, device=self.device) * t_max  # [*]
        elif self.cfg_exp.loss.t_distribution.name == "logit-normal":
            mean = self.cfg_exp.loss.t_distribution.p1
            std = self.cfg_exp.loss.t_distribution.p2
            noise = torch.randn(shape, device=self.device) * std + mean  # [*]
            return torch.nn.functional.sigmoid(noise)  # [*]
        elif self.cfg_exp.loss.t_distribution.name == "beta":
            p1 = self.cfg_exp.loss.t_distribution.p1
            p2 = self.cfg_exp.loss.t_distribution.p2
            dist = torch.distributions.beta.Beta(p1, p2)
            return dist.sample(shape).to(self.device)
        elif self.cfg_exp.loss.t_distribution.name == "mix_up02_beta":
            p1 = self.cfg_exp.loss.t_distribution.p1
            p2 = self.cfg_exp.loss.t_distribution.p2
            dist = torch.distributions.beta.Beta(p1, p2)
            samples_beta = dist.sample(shape).to(self.device)
            samples_uniform = torch.rand(shape, device=self.device)
            u = torch.rand(shape, device=self.device)
            return torch.where(u < 0.02, samples_uniform, samples_beta)
        else:
            raise NotImplementedError(
                f"Sampling mode for t {self.cfg_exp.loss.t_distribution.name} not implemented"
            )

    def training_step(self, batch, batch_idx):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batches.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"
        
        # Extract inputs from batch (our dataloader)
        # This may apply augmentations, if requested in the config file
        x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)

        # Center and mask input
        x_1 = self.fm._mask_and_zero_com(x_1, mask)

        # Sample time
        t = self.sample_t(batch_shape)
        contact_map_mode = getattr(self, "contact_map_mode", False)
        predict_coords = getattr(self.nn, "predict_coords", True)

        if contact_map_mode:
            # Contact map diffusion mode:
            # - Diffuse ONLY contact map
            # - Coordinates: direct prediction with FAPE loss (no noising)
            c_1 = self.extract_clean_contact_map(batch, mask)
            c_0 = self.fm.sample_reference(
                n=n,
                shape=batch_shape,
                device=self.device,
                dtype=dtype,
                mask=mask,
                modality="contact_map",
            )
            c_t = self.fm.interpolate(
                c_0, c_1, t, mask=mask, modality="contact_map"
            )
            batch["contact_map_t"] = c_t
            # x_t is a placeholder only (not used for diffusion in this mode)
            x_t = torch.zeros_like(x_1)
        else:
            x_0 = self.fm.sample_reference(
                n=n,
                shape=batch_shape,
                device=self.device,
                dtype=dtype,
                mask=mask,
                modality="coordinates",
            )
            x_t = self.fm.interpolate(x_0, x_1, t, modality="coordinates")
        
        if self.motif_conditioning:
            batch.update(self.motif_factory(batch))
            x_1 = batch["x_1"] # we need this since we change x_1 based n the motif center
        # Add a few things to batch, needed for nn
        batch["t"] = t
        batch["mask"] = mask
        batch["x_t"] = x_t

        # Fold conditional training
        if self.cfg_exp.training.fold_cond:
            bs = x_1.shape[0]
            cath_code_list = batch.cath_code
            for i in range(bs):
                if cath_code_list[i] is None:
                    cath_code_list[i] = ["x.x.x.x"]
                    continue
                # Progressively mask T, A, C levels
                cath_code_list[i] = mask_cath_code_by_level(
                    cath_code_list[i], level="H"
                )
                if random.random() < self.cfg_exp.training.mask_T_prob:
                    cath_code_list[i] = mask_cath_code_by_level(
                        cath_code_list[i], level="T"
                    )
                    if random.random() < self.cfg_exp.training.mask_A_prob:
                        cath_code_list[i] = mask_cath_code_by_level(
                            cath_code_list[i], level="A"
                        )
                        if random.random() < self.cfg_exp.training.mask_C_prob:
                            cath_code_list[i] = mask_cath_code_by_level(
                                cath_code_list[i], level="C"
                            )
            batch.cath_code = cath_code_list
        else:
            if "cath_code" in batch:
                batch.pop("cath_code")

        # Sequence conditional training
        if self.cfg_exp.training.seq_cond:
            # Get the sequence from the batch
            seq = batch["residue_type"]
            seq[seq == -1] = 20
            # Mask the sequence
            for i in range(len(seq)):
                seq[i] = mask_seq(seq[i], self.cfg_exp.training.mask_seq_proportion)
            batch["residue_type"] = seq
        else:
            # Remove residue_type from batch when seq_cond is False
            if "residue_type" in batch:
                batch.pop("residue_type")
                print("residue_type removed from batch")

        # CIRPIN conditional training
        if self.cfg_exp.training.get("cirpin_cond", False):
            # CIRPIN conditioning now expects cirpin_emb_fallback directly in the batch
            # The dataset should have already loaded the embeddings
            if "cirpin_emb_fallback" not in batch:
                # If cirpin_emb_fallback is not available, create zero embeddings
                logger.warning("cirpin_emb_fallback not found in batch for CIRPIN conditioning, using zero embeddings")
                bs = x_1.shape[0]
                batch["cirpin_emb_fallback"] = torch.zeros(bs, 1, 128, dtype=x_1.dtype, device=x_1.device)
            else:
                # Handle CIRPIN masking based on mask_cirpin_prob
                cirpin_emb_raw = batch["cirpin_emb_fallback"]  # May be [batch_size, seq_len, 128]
                
                mask_cirpin_prob = self.cfg_exp.training.get("mask_cirpin_prob", 0.0)
                if mask_cirpin_prob > 0.0:
                    bs = x_1.shape[0]
                    for i in range(bs):
                        if random.random() < mask_cirpin_prob:
                            # Mask CIRPIN by setting embedding to zero
                            if cirpin_emb_raw.dim() == 3:  # [batch_size, seq_len, 128]
                                cirpin_emb_raw[i, 0, :] = torch.zeros(128, dtype=cirpin_emb_raw.dtype, device=cirpin_emb_raw.device)
                            elif cirpin_emb_raw.dim() == 2:  # [batch_size, 128]
                                cirpin_emb_raw[i] = torch.zeros(128, dtype=cirpin_emb_raw.dtype, device=cirpin_emb_raw.device)
                    batch["cirpin_emb_fallback"] = cirpin_emb_raw
        else:
            # Remove cirpin_emb_fallback from batch when cirpin_cond is False (optional cleanup)
            if "cirpin_emb_fallback" in batch:
                batch.pop("cirpin_emb_fallback")

        # Prediction for self-conditioning
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            x_pred_sc, nn_out_sc = self.predict_clean(batch)
            if x_pred_sc is not None:
                batch["x_sc"] = self.detach_gradients(x_pred_sc)
            else:
                batch["x_sc"] = torch.zeros_like(x_1)
            if contact_map_mode and "contact_map_pred" in nn_out_sc:
                batch["contact_map_sc"] = self.detach_gradients(
                    nn_out_sc["contact_map_pred"]
                )
        else:
            batch["x_sc"] = torch.zeros_like(x_1)
            if contact_map_mode:
                batch["contact_map_sc"] = torch.zeros(
                    batch_shape + (n, n),
                    device=self.device,
                    dtype=x_1.dtype,
                )

        x_1_pred, nn_out = self.predict_clean(batch)

        if contact_map_mode:
            c_1_pred = nn_out.get("contact_map_pred")
            if c_1_pred is None:
                raise ValueError(
                    "contact_map_mode=True requires contact_map_pred in model output."
                )
            contact_map_loss = self.compute_contact_map_loss(
                c_1, c_1_pred, batch["contact_map_t"], t, mask, log_prefix=log_prefix
            )
            train_loss = torch.mean(contact_map_loss)

            if predict_coords and x_1_pred is not None:
                coord_loss = self.compute_fape_loss(
                    x_1, x_1_pred, mask, log_prefix=log_prefix
                )
                coord_weight = getattr(
                    self, "contact_map_coord_loss_weight", 0.0
                )
                train_loss = train_loss + (
                    coord_weight * torch.mean(coord_loss)
                )
        else:
            # Compute losses
            fm_loss = self.compute_fm_loss(
                x_1, x_1_pred, x_t, t, mask, log_prefix=log_prefix
            )  # [*]
            train_loss = torch.mean(fm_loss)
        
        if self.cfg_exp.loss.use_aux_loss:
            auxiliary_loss = self.compute_auxiliary_loss(
                x_1, x_1_pred, x_t, t, mask, nn_out=nn_out, log_prefix=log_prefix, batch=batch
            )  # [*] already includes loss weights
            train_loss = train_loss + torch.mean(auxiliary_loss)

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )

        # Don't log if validation step (indicated by batch_id)
        if not val_step:
            self.log(
                f"train_loss",
                train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )

            # For scaling laws
            b, n = mask.shape
            nflops_step = None
            if nflops_step is not None:
                self.nflops = (
                    self.nflops + nflops_step * self.trainer.world_size
                )  # Times number of processes so it logs sum across devices
                self.log(
                    "scaling/nflops",
                    self.nflops * 1.0,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=1,
                    sync_dist=True,
                )

            self.nsamples_processed = (
                self.nsamples_processed + b * self.trainer.world_size
            )
            self.log(
                "scaling/nsamples_processed",
                self.nsamples_processed * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

            self.log(
                "scaling/nparams",
                self.nparams * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )
            # Constant line but ok, easy to compare # params
            
            # Log structure visualization (if enabled)
            log_every_n = self.cfg_exp.training.get("log_structure_every_n_steps", 0)
            if log_every_n > 0 and self.global_step % log_every_n == 0:
                self._log_structure_visualization(
                    x_1_pred=x_1_pred,
                    contact_map_pred=nn_out.get("contact_map_pred"),
                    mask=mask,
                    log_prefix=log_prefix,
                )

        return train_loss
    
    def _log_structure_visualization(
        self,
        x_1_pred: torch.Tensor,
        contact_map_pred: torch.Tensor,
        mask: torch.Tensor,
        log_prefix: str,
    ):
        """
        Logs predicted structure (as temporary PDB) and contact map visualizations.

        Args:
            x_1_pred: Predicted coordinates, shape [b, n, 3]
            contact_map_pred: Predicted contact map, shape [b, n, n] or None
            mask: Boolean mask, shape [b, n]
            log_prefix: Prefix for log names ("train" or "validation_loss")
        """
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        if x_1_pred is not None:
            # Save first sample as temporary PDB for logging
            atom37 = self.samples_to_atom37(x_1_pred[:1]).detach().cpu().numpy()
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
                temp_pdb_path = tmp_pdb.name
            try:
                write_prot_to_pdb(atom37[0], temp_pdb_path, overwrite=True, no_indexing=True)
                self.logger.experiment.log(
                    {
                        f"{log_prefix}/structure": wandb.Molecule(temp_pdb_path),
                        "global_step": self.global_step,
                    }
                )
            finally:
                if os.path.exists(temp_pdb_path):
                    os.remove(temp_pdb_path)

        if contact_map_pred is None:
            return

        mask_np = mask[0].detach().cpu().numpy()
        contact_map_np = contact_map_pred[0].detach().cpu().numpy()
        pair_mask_np = mask_np[:, None] * mask_np[None, :]
        contact_map_np = contact_map_np * pair_mask_np

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(contact_map_np, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("Residue j")
        ax.set_ylabel("Residue i")
        ax.set_title(f"Contact Map - Step {self.global_step}")
        plt.colorbar(im, ax=ax, label="Contact probability")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_contact = Image.open(buf)
        plt.close(fig)

        self.logger.experiment.log(
            {
                f"{log_prefix}/contact_map": wandb.Image(img_contact),
                "global_step": self.global_step,
            }
        )

    @abstractmethod
    def compute_fm_loss(
        self, x_1, x_1_pred, x_t, t, mask
    ):
        """
        Computes and logs flow matching loss(es).

        Args:
            x_1: True clean sample.
            x_1_pred: Predicted clean sample.
            x_t: Sample at interpolation time t (used as input to predict clean sample).
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss per sample in the batch.
        """

    def compute_fape_loss(
        self,
        x_1,
        x_1_pred,
        mask,
        log_prefix: str = None,
    ):
        """
        Computes FAPE loss using identity frames (no orientation info).
        
        This is useful for contact map diffusion mode where we predict coordinates
        directly without orientation information from the backbone.
        
        Args:
            x_1: Ground truth coordinates, shape [b, n, 3].
            x_1_pred: Predicted coordinates, shape [b, n, 3].
            mask: Boolean mask, shape [b, n].
            log_prefix: Optional prefix for logging.
            
        Returns:
            FAPE loss per sample, shape [b].
        """
        from openfold.utils.loss import compute_fape
        from openfold.utils.rigid_utils import Rigid, Rotation
        
        frames_mask = mask.float()
        
        # Create identity rotation matrices for all residues
        rot = torch.eye(3, device=x_1.device, dtype=x_1.dtype)
        rot = rot.view(1, 1, 3, 3).expand(x_1.shape[0], x_1.shape[1], 3, 3)
        
        # Create rigid frames with identity rotations and CA translations
        pred_rotation = Rotation(rot_mats=rot, quats=None)
        target_rotation = Rotation(rot_mats=rot, quats=None)
        pred_frames = Rigid(rot=pred_rotation, trans=x_1_pred)
        target_frames = Rigid(rot=target_rotation, trans=x_1)
        
        fape = compute_fape(
            pred_frames=pred_frames,
            target_frames=target_frames,
            frames_mask=frames_mask,
            pred_positions=x_1_pred,
            target_positions=x_1,
            positions_mask=frames_mask,
            length_scale=10.0,
            l1_clamp_distance=10.0,
        )
        
        if log_prefix:
            self.log(
                f"{log_prefix}/fape_loss",
                torch.mean(fape),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
        
        return fape

    @abstractmethod
    def compute_auxiliary_loss(
        self, x_1, x_1_pred, x_t, t, mask, batch = None
    ):
        """
        Computes and logs auxiliary losses.

        Args:
            x_1: True clean sample.
            x_1_pred: Predicted clean sample.
            x_t: Sample at interpolation time t (used as input to predict clean sample).
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Auxiliary loss per sample in the batch.
        """

    @abstractmethod
    def detach_gradients(self, x):
        """Detaches gradients from sample x"""

    def validation_step(self, batch, batch_idx):
        """
        This is the validation step for both when generating proteins (dataloader_idx_1) and when evaluating the training
        loss on some validation data (dataloader_idx_2).

        dataloader_idx_1: The batch comes from the length dataset
        dataloader_idx_2: The batch contains actual data

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
            dataloader_idx: 0 or 1.
                0 means the batch comes from the length dataloader, contains no data, but the info of the samples to generate (nsamples, nres, dt)
                1 means the batch comes from the data dataloader, contains data from the dataset, we compute normal training loss
        """
        self.validation_step_data(batch, batch_idx)

    def validation_step_data(self, batch, batch_idx):
        """
        Evaluates the training loss, without auxiliary loss nor logging.
        This is done with the function `training_step` with batch_idx -1.
        """
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def predict_step(self, batch, batch_idx):
        """
        Makes predictions. Should call set_inf_cfg before calling this.

        Args:
            batch: data batch, contains no data, but the info of the samples
                to generate (nsamples, nres, dt)

        Returns:
            Dictionary with keys:
                - "coords_atom37": Generated coordinates in atom37 format, shape [b, n, 37, 3]
                - "contact_map": Generated contact map (if available), shape [b, n, n]
                - "cath_code": CATH codes for each sample
                - "cirpin_ids": CIRPIN IDs for each sample
        """
        sampling_args = self.inf_cfg.sampling_caflow

        cath_code = (
            _extract_cath_code(batch) if self.inf_cfg.get("fold_cond", False) else [["x.x.x.x"] for _ in range(batch["nsamples"])]
        )  # When using unconditional model, don't use cath_code
        residue_type = batch["residue_type"] if self.inf_cfg.get("seq_cond", False) else None
        guidance_weight = self.inf_cfg.get("guidance_weight", 1.0)
        autoguidance_ratio = self.inf_cfg.get("autoguidance_ratio", 0.0)
        
        mask = batch['mask'].squeeze(0) if 'mask' in batch else None
        if 'motif_seq_mask' in batch:
            fixed_sequence_mask = batch['motif_seq_mask'].squeeze(0).to(self.device)
            x_motif = batch['motif_structure'].squeeze(0).to(self.device)
            fixed_structure_mask = fixed_sequence_mask[:, :, None] * fixed_sequence_mask[:, None, :]
        else:
            fixed_sequence_mask, x_motif, fixed_structure_mask = None, None, None
            fixed_sequence_mask = None

        result = self.generate(
            nsamples=batch["nsamples"],
            n=batch["nres"],
            dt=batch["dt"].to(dtype=torch.float32),
            self_cond=self.inf_cfg.self_cond,
            cath_code=cath_code,
            residue_type=residue_type,
            guidance_weight=guidance_weight,
            autoguidance_ratio=autoguidance_ratio,
            dtype=torch.float32,
            schedule_mode=self.inf_cfg.schedule.schedule_mode,
            schedule_p=self.inf_cfg.schedule.schedule_p,
            sampling_mode=sampling_args["sampling_mode"],
            sc_scale_noise=sampling_args["sc_scale_noise"],
            sc_scale_score=sampling_args["sc_scale_score"],
            gt_mode=sampling_args["gt_mode"],
            gt_p=sampling_args["gt_p"],
            gt_clamp_val=sampling_args["gt_clamp_val"],
            mask = mask,
            x_motif = x_motif,
            fixed_sequence_mask = fixed_sequence_mask,
            fixed_structure_mask = fixed_structure_mask,
        )
        cirpin_ids = batch.get("cirpin_ids", [None] * len(cath_code))
        coords_atom37 = None
        if result.get("coords") is not None:
            coords_atom37 = self.samples_to_atom37(result["coords"])

        return {
            "coords_atom37": coords_atom37,
            "contact_map": result.get("contact_map"),
            "distogram": result.get("distogram"),
            "cath_code": cath_code,
            "cirpin_ids": cirpin_ids,
        }

    def generate(
        self,
        nsamples: int,
        n: int,
        dt: float,
        self_cond: bool,
        cath_code: List[List[str]],
        residue_type: List[List[int]],
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
        dtype: torch.dtype = None,
        schedule_mode: str = "uniform",
        schedule_p: float = 1.0,
        sampling_mode: str = "sc",
        sc_scale_noise: float = "1.0",
        sc_scale_score: float = "1.0",
        gt_mode: Literal["us", "tan"] = "us",
        gt_p: float = 1.0,
        gt_clamp_val: float = None,
        mask = None,
        x_motif = None,
        fixed_sequence_mask = None,
        fixed_structure_mask = None,
    ) -> Dict[str, Tensor]:
        """
        Generates samples by integrating ODE with learned vector field.

        Returns:
            Dictionary with keys:
                - "coords": Generated coordinates, shape [nsamples, n, 3]
                - "contact_map": Generated contact map (if contact_map_mode), shape [nsamples, n, n]
                - "distogram": Distogram logits (if available), shape [nsamples, n, n, num_buckets]
        """
        predict_clean_n_v_w_guidance = partial(
            self.predict_clean_n_v_w_guidance,
            guidance_weight=guidance_weight,
            autoguidance_ratio=autoguidance_ratio,
        )
        if mask is None:
            mask = torch.ones(nsamples, n).long().bool().to(self.device)

        contact_map_mode = getattr(self, "contact_map_mode", False)
        modality = "contact_map" if contact_map_mode else "coordinates"
        predict_coords = getattr(self.nn, "predict_coords", True)

        return self.fm.full_simulation(
            predict_clean_n_v_w_guidance,
            dt=dt,
            nsamples=nsamples,
            n=n,
            self_cond=self_cond,
            cath_code=cath_code,
            residue_type=residue_type,
            device=self.device,
            mask=mask,
            dtype=dtype,
            schedule_mode=schedule_mode,
            schedule_p=schedule_p,
            sampling_mode=sampling_mode,
            sc_scale_noise=sc_scale_noise,
            sc_scale_score=sc_scale_score,
            gt_mode=gt_mode,
            gt_p=gt_p,
            gt_clamp_val=gt_clamp_val,
            x_motif=x_motif,
            fixed_sequence_mask=fixed_sequence_mask,
            fixed_structure_mask=fixed_structure_mask,
            modality=modality,
            predict_coords=predict_coords,
        )



def _extract_cath_code(batch):
    cath_code = batch.get("cath_code", None)
    if cath_code:
        # Remove the additional tuple layer introduced during collate
        _cath_code = []
        for codes in cath_code:
            _cath_code.append(
                [code[0] if isinstance(code, tuple) else code for code in codes]
            )
        cath_code = _cath_code
    return cath_code
