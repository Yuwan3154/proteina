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
import time

from abc import abstractmethod
from functools import partial
from typing import Dict, List, Literal, Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Bool, Float
from loguru import logger
from PIL import Image
from torch import Tensor

from proteinfoundation.flow_matching.discrete_md4 import (
    GenMD4DiscreteDiffusion,
    MD4DiscreteDiffusion,
    _normalize_position_bias_mode,
)
from proteinfoundation.utils.ff_utils.pdb_utils import (
    mask_cath_code_by_level,
    mask_seq,
    write_prot_to_pdb,
)
# from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference, OpenFoldDistogramOnlyInference
import proteinfoundation.openfold_stub.np.residue_constants as rc
from proteinfoundation.openfold_stub.utils.loss import compute_fape
from proteinfoundation.openfold_stub.utils.rigid_utils import Rigid, Rotation

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
        self.discrete_diffusion = None
        self.discrete_diffusion_type = None
        self._init_discrete_diffusion()

        # Lazy OpenFold template inference helper
        self._template_inference = None
        self._logged_val_traj_epoch = -1
        self._self_cond_copy_last_step = None

    def configure_optimizers(self):
        if self.cfg_exp.training.finetune_seq_cond_lora_only:
            opt_params = []
            opt_params_names = []
            for name, param in self.named_parameters():
                if name.startswith("nn_sc."):
                    param.requires_grad = False
                    continue
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
            for name, param in self.named_parameters():
                if name.startswith("nn_sc."):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad],
                lr=self.cfg_exp.opt.lr,
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

    def _maybe_update_self_cond_copy(self) -> None:
        """Optionally sync a frozen self-conditioning model copy from the main model."""
        nn_sc = getattr(self, "nn_sc", None)
        if nn_sc is None:
            return
        update_every = int(self.cfg_exp.training.get("self_cond_copy_update_every", 1))
        if update_every <= 0:
            return
        step = int(getattr(self, "global_step", 0))
        last = self._self_cond_copy_last_step
        if last is None or (step - int(last)) >= update_every:
            with torch.no_grad():
                nn_sc.load_state_dict(self.nn.state_dict(), strict=True)
            self._self_cond_copy_last_step = step

    def on_after_backward(self) -> None:
        """Clears gradients when a non-finite loss was detected for this backward."""
        if getattr(self, "global_rank", 0) != 0:
            return
        if getattr(self, "_skip_update_due_to_nonfinite_loss", False):
            self.zero_grad()

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
        if "coords_pred" not in nn_out:
            return None

        nn_pred = nn_out["coords_pred"]
        t = batch["t"]  # [*]
        t_ext = t[..., None, None]  # [*, 1, 1]
        x_t = batch["x_t"]  # [*, n, 3]
        if self.cfg_exp.model.target_pred == "x_1" or self.contact_map_mode:
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
        nn_pred = nn_out["contact_map_pred"]
        t = batch["t"]  # [*]
        t_ext = t[..., None, None]  # [*, 1, 1]
        contact_map_t = batch["contact_map_t"]  # [*, n, n]
        if self.cfg_exp.model.target_pred == "c_1":
            contact_map_pred = nn_pred
        elif self.cfg_exp.model.target_pred == "v":
            contact_map_pred = contact_map_t + (1.0 - t_ext) * nn_pred
        else:
            raise IOError(
                f"Wrong parameterization chosen: {self.cfg_exp.model.target_pred} for contact map prediction"
            )
        return contact_map_pred

    def _nn_out_to_c_logits(self, nn_out: Dict, batch: Dict):
        """
        Returns clean contact-map logits, when available.

        This is used for losses that operate in logit space (e.g. BCEWithLogits when
        non_contact_value == 0).
        """
        if "contact_map_logits" not in nn_out:
            return None
        if self.cfg_exp.model.target_pred != "c_1":
            raise ValueError(
                "Contact-map mode currently expects target_pred == 'c_1' "
                f"(got {self.cfg_exp.model.target_pred})."
            )
        return nn_out["contact_map_logits"]

    def _contact_map_to_viz(self, c: torch.Tensor) -> torch.Tensor:
        """
        Converts a contact map in model space to [0, 1] for visualization.
        - If non_contact_value == 0: already in [0, 1]
        - If non_contact_value == -1: map from [-1, 1] to [0, 1] via (x + 1)/2
        """
        non_contact_value = getattr(self.nn, "non_contact_value")
        if non_contact_value == 0:
            return c
        elif non_contact_value == -1:
            return (c + 1.0) * 0.5
        else:
            raise ValueError(f"non_contact_value must be 0 or -1, got {non_contact_value}")

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
        return self.nn(batch)

    def _get_template_inference_module(self, num_bins: int):
        raise NotImplementedError("This method is not implemented")
        # if self._template_inference is None or self._template_inference.num_bins != num_bins:
        #     self._template_inference = OpenFoldTemplateInference(num_bins=num_bins).to(self.device)
        # return self._template_inference

    def _get_distogram_only_inference_module(self):
        raise NotImplementedError("This method is not implemented")
        # if getattr(self, "_distogram_only_inference", None) is None:
        #     model_name = self.cfg_exp.model.nn.get("openfold_model_name", "model_1_ptm")
        #     jax_params_path = self.cfg_exp.model.nn.get(
        #         "openfold_jax_params_path", "/home/ubuntu/params/params_model_1_ptm.npz"
        #     )
        #     self._distogram_only_inference = OpenFoldDistogramOnlyInference(
        #         model_name=model_name,
        #         jax_params_path=jax_params_path,
        #         device=self.device,
        #     )
        # return self._distogram_only_inference

    def _predict_structure_from_distogram(
        self,
        pair_logits: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Distogram-only template inference uses probabilities [B,L,L,39].
        # We allow either logits or probs; if logits, convert to probs.
        use_full_openfold = self.cfg_exp.model.nn.get("openfold_distogram_only", True)
        if use_full_openfold:
            distogram_probs = torch.softmax(pair_logits, dim=-1)

            module = self._get_distogram_only_inference_module()
            module = module.to(self.device)
            module.eval()
            with torch.no_grad():
                out = module(distogram_probs, residue_type, mask)
            atom37 = out["atom37"]
            ca = atom37[..., rc.atom_order["CA"], :]
            return ca, atom37

        module = self._get_template_inference_module(pair_logits.shape[-1])
        module = module.to(self.device)
        module.eval()
        with torch.no_grad():
            out = module(pair_logits, residue_type, mask)
        atom37 = out["atom37"]
        ca = atom37[..., rc.atom_order["CA"], :]
        return ca, atom37

    def predict_clean_n_v_w_guidance(
        self,
        batch: Dict,
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
        force_compile: bool = False,
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

        nn_out = self.nn(batch, force_compile=force_compile)
        x_pred = self._nn_out_to_x_clean(nn_out, batch)
        c_pred = self._nn_out_to_c_clean(nn_out, batch)
        c_logits = self._nn_out_to_c_logits(nn_out, batch)
        
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
                c_logits_ag = self._nn_out_to_c_logits(nn_out_ag, batch) if c_logits is not None else None
            else:
                x_pred_ag = torch.zeros_like(x_pred) if x_pred is not None else None
                c_pred_ag = torch.zeros_like(c_pred) if c_pred is not None else None
                c_logits_ag = torch.zeros_like(c_logits) if c_logits is not None else None

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
                c_logits_uncond = self._nn_out_to_c_logits(nn_out_uncond, uncond_batch) if c_logits is not None else None
            else:
                x_pred_uncond = torch.zeros_like(x_pred) if x_pred is not None else None
                c_pred_uncond = torch.zeros_like(c_pred) if c_pred is not None else None
                c_logits_uncond = torch.zeros_like(c_logits) if c_logits is not None else None

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
            if c_logits is not None:
                c_logits = guidance_weight * c_logits + (1 - guidance_weight) * (
                    autoguidance_ratio * c_logits_ag
                    + (1 - autoguidance_ratio) * c_logits_uncond
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
            # `c_pred` is already in model/data space (activated in the NN):
            # - non_contact_value == 0  -> in [0, 1] (sigmoid)
            # - non_contact_value == -1 -> in [-1, 1] (tanh)
            result["contact_map"] = c_pred
            # Only compute velocity in continuous contact map diffusion mode
            if (
                contact_map_mode
                and "contact_map_t" in batch
                and not self._discrete_diffusion_enabled()
            ):
                result["contact_map_v"] = self.fm.xt_dot(
                    c_pred,
                    batch["contact_map_t"],
                    batch["t"],
                    batch["mask"],
                    modality="contact_map",
                )
        
        if c_logits is not None:
            result["contact_map_logits"] = c_logits

        # Include distogram if available
        if "pair_logits" in nn_out:
            # Store probabilities for OpenFold distogram-only template inference.
            result["distogram"] = torch.softmax(nn_out["pair_logits"], dim=-1)

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
    
    @abstractmethod
    def extract_clean_contact_map(self, batch, mask):
        """
        Extracts clean contact map from the batch.

        Args:
            batch: batch from dataloader.
            mask: mask of the batch.

        Returns:
            Clean contact map of shape [b, n, n].
        """

    def _get_discrete_diffusion_cfg(self):
        cfg = self.cfg_exp.model.get("discrete_diffusion", None)
        if cfg is None or not cfg.get("enabled", False):
            return None
        return cfg

    def _init_discrete_diffusion(self):
        cfg = self._get_discrete_diffusion_cfg()
        if cfg is None:
            return
        if not self.cfg_exp.model.nn.get("contact_map_mode", False):
            raise ValueError(
                "discrete_diffusion requires contact_map_mode=True in model.nn config."
            )
        diffusion_type = cfg.get("type", "md4")
        position_bias = self.cfg_exp.training.get("position_bias", None)
        if diffusion_type == "md4":
            md4_cfg = cfg.get("md4", {})
            self.discrete_diffusion = MD4DiscreteDiffusion(
                vocab_size=md4_cfg.get("vocab_size", 2),
                noise_schedule_type=md4_cfg.get("noise_schedule", "cosine"),
                timesteps=md4_cfg.get("timesteps", 1000),
                cont_time=md4_cfg.get("cont_time", True),
                sampling_grid=md4_cfg.get("sampling_grid", "cosine"),
                eps=md4_cfg.get("eps", 1e-4),
                position_bias=position_bias,
            )
        elif diffusion_type == "genmd4":
            gen_cfg = cfg.get("genmd4", {})
            self.discrete_diffusion = GenMD4DiscreteDiffusion(
                vocab_size=gen_cfg.get("vocab_size", 2),
                noise_schedule_type=gen_cfg.get("noise_schedule", "poly"),
                power_init=gen_cfg.get("power_init", 1.0),
                t1=gen_cfg.get("t1", 1e-3),
                eps=gen_cfg.get("eps", 1e-4),
                position_bias=position_bias,
            )
        else:
            raise ValueError(
                f"Unknown discrete diffusion type {diffusion_type}. "
                "Use 'md4' or 'genmd4'."
            )
        self.discrete_diffusion_type = diffusion_type

    def _discrete_diffusion_enabled(self) -> bool:
        return self.discrete_diffusion is not None

    def _pair_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return mask[..., :, None] * mask[..., None, :]

    def _apply_pair_mask(self, pair_tensor: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        mask = pair_mask.float()
        if pair_tensor.dim() == mask.dim() + 1:
            mask = mask.unsqueeze(-1)
        return pair_tensor * mask

    def _contact_tokens_to_input(self, tokens: torch.Tensor) -> torch.Tensor:
        input_dim = int(getattr(self.nn, "contact_map_input_dim", 1))
        if input_dim == 1:
            non_contact_value = getattr(self.nn, "non_contact_value", 0)
            if non_contact_value == -1:
                contact = torch.ones_like(tokens, dtype=torch.float32)
                non_contact = -torch.ones_like(tokens, dtype=torch.float32)
                mask_val = torch.zeros_like(tokens, dtype=torch.float32)
                return torch.where(
                    tokens == self.discrete_diffusion.mask_token,
                    mask_val,
                    torch.where(tokens == 1, contact, non_contact),
                )
            return tokens.float()
        if input_dim == 3:
            num_classes = self.discrete_diffusion.vocab_size + 1
            return torch.nn.functional.one_hot(tokens, num_classes=num_classes).float()
        raise ValueError(
            f"Unsupported contact_map_input_dim {input_dim}. Expected 1 or 3."
        )

    def _contact_tokens_to_output(self, tokens: torch.Tensor) -> torch.Tensor:
        non_contact_value = getattr(self.nn, "non_contact_value", 0)
        if non_contact_value == -1:
            contact = torch.ones_like(tokens, dtype=torch.float32)
            non_contact = -torch.ones_like(tokens, dtype=torch.float32)
            return torch.where(tokens == 1, contact, non_contact)
        return tokens.float()

    def _set_contact_map_self_cond(self, batch: Dict, c_1: torch.Tensor, use_self_cond: bool):
        if use_self_cond:
            with torch.no_grad():
                self._maybe_update_self_cond_copy()
                sc_model = getattr(self, "nn_sc", None) or self.nn
                nn_out_sc = sc_model(batch)
                c_pred_sc = self._nn_out_to_c_clean(nn_out_sc, batch)
            if c_pred_sc is not None:
                batch["contact_map_sc"] = self.detach_gradients(c_pred_sc)
            else:
                batch["contact_map_sc"] = torch.zeros_like(c_1)
        else:
            batch["contact_map_sc"] = torch.zeros_like(c_1)

    def sample_t(self, shape):
        dist_name = self.cfg_exp.loss.t_distribution.name
        if self._discrete_diffusion_enabled() and dist_name not in ("uniform", "cosine", "cosine_cdf"):
            raise ValueError(
                f"Discrete diffusion only supports t_distribution uniform or cosine "
                f"(got {dist_name})."
            )
        if dist_name == "uniform":
            t_max = self.cfg_exp.loss.t_distribution.p2
            return torch.rand(shape, device=self.device) * t_max  # [*]
        elif dist_name in ("cosine", "cosine_cdf"):
            t = torch.rand(shape, device=self.device)
            t = torch.sin(0.5 * math.pi * t)
            t_max = self.cfg_exp.loss.t_distribution.get("p2", 1.0)
            return t * t_max
        elif dist_name == "logit-normal":
            mean = self.cfg_exp.loss.t_distribution.p1
            std = self.cfg_exp.loss.t_distribution.p2
            noise = torch.randn(shape, device=self.device) * std + mean  # [*]
            return torch.nn.functional.sigmoid(noise)  # [*]
        elif dist_name == "beta":
            p1 = self.cfg_exp.loss.t_distribution.p1
            p2 = self.cfg_exp.loss.t_distribution.p2
            dist = torch.distributions.beta.Beta(p1, p2)
            return dist.sample(shape).to(self.device)
        elif dist_name == "mix_up02_beta":
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
        t0 = time.time()
        rank = getattr(self, "global_rank", -1)
        logger.info(f"DEBUG_TRACE: [rank={rank}] training_step start batch_idx={batch_idx}")
        self._debug_last_batch_idx = int(batch_idx)
        self._debug_nonfinite_loss_ids = None
        self._skip_update_due_to_nonfinite_loss = False
        val_step = batch_idx == -1 or getattr(self, "_in_validation_loop", False)
        log_prefix = "validation_loss" if val_step else "train"
        
        # Extract inputs from batch (our dataloader)
        # This may apply augmentations, if requested in the config file
        x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        logger.info(f"DEBUG_TRACE: [rank={rank}] extract_clean_sample done batch_idx={batch_idx} t={time.time()-t0:.3f}s")
        
        # Center and mask input
        x_1 = self.fm._mask_and_zero_com(x_1, mask)
        logger.info(f"DEBUG_TRACE: [rank={rank}] _mask_and_zero_com done batch_idx={batch_idx} t={time.time()-t0:.3f}s")

        # Sample time
        t = self.sample_t(batch_shape)
        contact_map_mode = getattr(self, "contact_map_mode", False)
        predict_coords = getattr(self.nn, "predict_coords", True)
        discrete_enabled = contact_map_mode and self._discrete_diffusion_enabled()
        if discrete_enabled and self.cfg_exp.model.target_pred != "c_1":
            raise ValueError(
                "Discrete contact-map diffusion requires model.target_pred == 'c_1'."
            )

        # Ensure x_1_pred is always defined for downstream auxiliary loss calls
        x_1_pred = None
        c_tokens = None
        pair_mask = None
        c_1 = None
        zt = None
        zt_2 = None
        contact_map_t_1 = None

        if contact_map_mode:
            if discrete_enabled:
                if "contact_map" not in batch:
                    raise ValueError(
                        "contact_map not found in batch. Ensure ContactMapTransform is enabled."
                    )
                pair_mask = self._pair_mask(mask)
                c_tokens = (batch["contact_map"].float() > 0.5).long()
                c_tokens = c_tokens * pair_mask.long()
                c_1 = self._contact_tokens_to_output(c_tokens)
                c_1 = c_1 * pair_mask.float()
                if self.discrete_diffusion_type == "genmd4":
                    t1 = self.discrete_diffusion.t1
                    t = (1.0 - t1) * t + t1
                zt = self.discrete_diffusion.forward_sample(c_tokens, t)
                if self.discrete_diffusion_type == "genmd4":
                    zt_2 = self.discrete_diffusion.forward_sample(c_tokens, t)
                contact_map_t_1 = self._contact_tokens_to_input(zt)
                contact_map_t_1 = self._apply_pair_mask(contact_map_t_1, pair_mask)
                batch["contact_map_t"] = contact_map_t_1
                # x_t is a placeholder only (not used for diffusion in this mode)
                x_t = torch.zeros_like(x_1)
            else:
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
        logger.info(f"DEBUG_TRACE: pre-forward prep done batch_idx={batch_idx} t={time.time()-t0:.3f}s")

        # Fold conditional training
        if self.cfg_exp.training.fold_cond:
            logger.info(f"DEBUG_TRACE: [rank={rank}] fold_cond start batch_idx={batch_idx}")
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
            logger.info(f"DEBUG_TRACE: [rank={rank}] fold_cond end batch_idx={batch_idx}")
        else:
            if "cath_code" in batch:
                batch.pop("cath_code")

        # Sequence conditional training
        if self.cfg_exp.training.seq_cond:
            logger.info(f"DEBUG_TRACE: [rank={rank}] seq_cond start batch_idx={batch_idx}")
            # Get the sequence from the batch
            seq = batch["residue_type"]
            seq[seq == -1] = 20
            # Preserve the unmasked sequence for logging / IPA geometry (do this before masking)
            if "residue_type_unmasked" not in batch:
                print(f"DEBUG_TRACE: [rank={rank}] residue_type_unmasked not in batch creating clone")
                residue_type_unmasked = seq.clone()
                print(f"DEBUG_TRACE: [rank={rank}] residue_type_unmasked cloned batch_idx={batch_idx}")
                batch["residue_type_unmasked"] = residue_type_unmasked
                print(f"DEBUG_TRACE: [rank={rank}] residue_type_unmasked added to batch batch_idx={batch_idx}")

            # Mask the sequence
            for i in range(len(seq)):
                seq[i] = mask_seq(seq[i], self.cfg_exp.training.mask_seq_proportion)
            batch["residue_type"] = seq
            logger.info(f"DEBUG_TRACE: [rank={rank}] seq_cond end batch_idx={batch_idx}")
        else:
            # Keep residue_type if IPA coordinates are needed for contact_map_mode
            need_residue_type = (
                getattr(self, "contact_map_mode", False)
                and getattr(self.nn, "predict_coords", None) == "ipa"
            )
            if "residue_type" in batch and not need_residue_type:
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
        
        logger.info(f"DEBUG_TRACE: cond prep done batch_idx={batch_idx} t={time.time()-t0:.3f}s")

        # Prediction for self-conditioning
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            if contact_map_mode:
                # In contact-map mode, target_pred is 'c_1' and predict_clean()'s
                # coordinate path is not applicable. We only need the raw nn outputs
                # to extract contact_map_pred for self-conditioning.
                with torch.no_grad():
                    self._maybe_update_self_cond_copy()
                    sc_model = getattr(self, "nn_sc", None) or self.nn
                    logger.info(f"DEBUG_TRACE: starting self-cond forward batch_idx={batch_idx} t={time.time()-t0:.3f}s")
                    nn_out_sc = sc_model(batch)
                    logger.info(f"DEBUG_TRACE: finished self-cond forward batch_idx={batch_idx} t={time.time()-t0:.3f}s")
                    c_pred_sc = self._nn_out_to_c_clean(nn_out_sc, batch)
                if c_pred_sc is not None:
                    # `c_pred_sc` is already activated in model/data space.
                    batch["contact_map_sc"] = self.detach_gradients(c_pred_sc)
                else:
                    batch["contact_map_sc"] = torch.zeros_like(c_1)
            else:
                with torch.no_grad(): 
                    self._maybe_update_self_cond_copy()
                    sc_model = getattr(self, "nn_sc", None) or self.nn
                    logger.info(f"DEBUG_TRACE: starting self-cond forward batch_idx={batch_idx} t={time.time()-t0:.3f}s")
                    nn_out_sc = sc_model(batch)
                    logger.info(f"DEBUG_TRACE: finished self-cond forward batch_idx={batch_idx} t={time.time()-t0:.3f}s")
                    x_pred_sc = self._nn_out_to_x_clean(nn_out_sc, batch)
                if x_pred_sc is not None:
                    batch["x_sc"] = self.detach_gradients(x_pred_sc)
                else:
                    batch["x_sc"] = torch.zeros_like(x_1)
        else:
            if contact_map_mode:
                batch["contact_map_sc"] = torch.zeros_like(c_1)
            else:
                batch["x_sc"] = torch.zeros_like(x_1)

        # Main prediction
        logger.info(f"DEBUG_TRACE: starting main forward batch_idx={batch_idx} t={time.time()-t0:.3f}s")
        nn_out = self.predict_clean(batch)
        logger.info(f"DEBUG_TRACE: finished main forward batch_idx={batch_idx} t={time.time()-t0:.3f}s")

        def _sanitize_and_log_loss_vec(loss_vec: torch.Tensor, name: str) -> torch.Tensor:
            """
            If loss_vec has non-finite entries, log which samples are affected and
            replace those entries with 0 (detached) so training can continue.
            Also marks the step to skip the optimizer update for safety.
            """
            if loss_vec is None:
                return loss_vec
            if loss_vec.numel() == 0:
                return loss_vec
            nonfinite = ~(torch.isfinite(loss_vec))
            if not nonfinite.any():
                return loss_vec

            idx = nonfinite.nonzero(as_tuple=False).reshape(-1)
            ids = None
            if "id" in batch:
                batch_ids = batch["id"]
                idx_list = idx.detach().cpu().tolist()
                # Common cases:
                # - list/tuple of strings
                # - 1D tensor of ints/strings-like
                if isinstance(batch_ids, (list, tuple)):
                    if len(batch_ids) > 0:
                        ids = [str(batch_ids[int(i)]) for i in idx_list]
                elif torch.is_tensor(batch_ids):
                    # Convert to Python list first
                    ids_list = batch_ids.detach().cpu().tolist()
                    ids = [str(ids_list[int(i)]) for i in idx_list]

            self._skip_update_due_to_nonfinite_loss = True
            self._debug_nonfinite_loss_ids = ids

            logger.warning(
                "[nonfinite_loss_detected/{}] step={} batch_idx={} n_bad={} bad_indices={} ids={}",
                name,
                int(getattr(self, "global_step", -1)),
                int(getattr(self, "_debug_last_batch_idx", -1)),
                int(idx.numel()),
                idx.detach().cpu().tolist()[:32],
                ids[:32] if ids is not None else None,
            )

            # Replace non-finite entries with 0 (detached) so grads stay finite.
            zero = loss_vec.detach() * 0.0
            return torch.where(torch.isfinite(loss_vec), loss_vec, zero)
        if contact_map_mode:
            non_contact_value = getattr(self.nn, "non_contact_value")
            if non_contact_value not in (0, -1):
                raise ValueError(f"non_contact_value must be 0 or -1, got {non_contact_value}")
            c_1_pred = self._nn_out_to_c_clean(nn_out, batch)
            if c_1_pred is None:
                raise ValueError(
                    "contact_map_mode=True requires contact_map_pred in model output."
                )
            if discrete_enabled:
                c_logits = self._nn_out_to_c_logits(nn_out, batch)
                if c_logits is None:
                    raise ValueError(
                        "discrete diffusion requires contact_map_logits in model output."
                    )
                if self.discrete_diffusion_type == "genmd4":
                    if zt_2 is None:
                        zt_2 = self.discrete_diffusion.forward_sample(c_tokens, t)
                    contact_map_t_2 = self._contact_tokens_to_input(zt_2)
                    contact_map_t_2 = self._apply_pair_mask(contact_map_t_2, pair_mask)
                    batch["contact_map_t"] = contact_map_t_2
                    nn_out_2 = self.predict_clean(batch)
                    c_logits_2 = self._nn_out_to_c_logits(nn_out_2, batch)
                    if c_logits_2 is None:
                        raise ValueError(
                            "discrete diffusion requires contact_map_logits in model output."
                        )
                    loss_diff_1 = self.discrete_diffusion.diffusion_loss(
                        c_logits, c_tokens, zt, t, pair_mask
                    )
                    loss_diff_2 = self.discrete_diffusion.diffusion_loss(
                        c_logits_2, c_tokens, zt_2, t, pair_mask
                    )
                    rloo = self.discrete_diffusion.reinforce_loss(
                        t, c_tokens, zt, zt_2, loss_diff_1, loss_diff_2, pair_mask
                    )
                    loss_diff = 0.5 * (loss_diff_1 + loss_diff_2)
                    loss_diff_sg = loss_diff + rloo
                    recon_loss = self.discrete_diffusion.recon_loss(c_tokens, pair_mask)
                    prior_loss = self.discrete_diffusion.latent_loss(
                        c_tokens.shape[0], c_tokens.device
                    )
                    contact_map_loss = loss_diff_sg + recon_loss + prior_loss
                    if contact_map_t_1 is not None:
                        batch["contact_map_t"] = contact_map_t_1
                else:
                    loss_diff = self.discrete_diffusion.diffusion_loss(
                        c_logits, c_tokens, zt, t, pair_mask
                    )
                    recon_loss = self.discrete_diffusion.recon_loss(pair_mask)
                    prior_loss = self.discrete_diffusion.latent_loss(
                        c_tokens.shape[0], c_tokens.device
                    )
                    contact_map_loss = loss_diff + recon_loss + prior_loss
            else:
                contact_map_loss = self.compute_contact_map_loss(
                    c_1, c_1_pred, batch["contact_map_t"], t, mask, log_prefix=log_prefix
                )
            contact_map_loss = _sanitize_and_log_loss_vec(contact_map_loss, "contact_map_loss")
            train_loss = torch.mean(contact_map_loss)

            if predict_coords:
                x_1_pred = nn_out["coords_pred"]
                pred_frames_tensor7 = None
                if getattr(self.nn, "predict_coords", None) == "ipa":
                    ipa_fape_loss_frame = self.cfg_exp.training.get(
                        "ipa_fape_loss_frame", "predicted_frames"
                    )
                    if ipa_fape_loss_frame == "predicted_frames":
                        pred_frames_tensor7 = nn_out.get("frames_pred", None)
                        if pred_frames_tensor7 is None:
                            raise ValueError(
                                "ipa_fape_loss_frame='predicted_frames' requires "
                                "nn_out['frames_pred'] from the model forward."
                            )
                    elif ipa_fape_loss_frame == "frames_from_predicted_N_CA_C":
                        pred_frames_tensor7 = None
                    else:
                        raise ValueError(
                            "ipa_fape_loss_frame must be one of "
                            "('predicted_frames', 'frames_from_predicted_N_CA_C'), "
                            f"got {ipa_fape_loss_frame!r}"
                        )
                coord_loss = self.compute_fape_loss(
                    x_1,
                    x_1_pred,
                    mask,
                    log_prefix=log_prefix,
                    pred_frames_tensor7=pred_frames_tensor7,
                    residue_type=batch.get("residue_type_unmasked", batch.get("residue_type", None)),
                )
                coord_loss = _sanitize_and_log_loss_vec(coord_loss, "fape_loss")
                coord_weight = getattr(
                    self, "contact_map_coord_loss_weight", 0.0
                )
                train_loss = train_loss + (
                    coord_weight * torch.mean(coord_loss)
                )
        else:
            x_1_pred = self._nn_out_to_x_clean(nn_out, batch)
            # Compute losses
            fm_loss = self.compute_fm_loss(
                x_1, x_1_pred, x_t, t, mask, log_prefix=log_prefix
            )  # [*]
            fm_loss = _sanitize_and_log_loss_vec(fm_loss, "fm_loss")
            train_loss = torch.mean(fm_loss)
        
        if self.cfg_exp.loss.use_aux_loss:
            auxiliary_loss = self.compute_auxiliary_loss(
                x_1, x_1_pred, x_t, t, mask, nn_out=nn_out, log_prefix=log_prefix, batch=batch
            )  # [*] already includes loss weights
            auxiliary_loss = _sanitize_and_log_loss_vec(auxiliary_loss, "auxiliary_loss")
            train_loss = train_loss + torch.mean(auxiliary_loss)

        # If anything non-finite was detected in this backward, skip the update safely.
        if self._skip_update_due_to_nonfinite_loss:
            logger.warning(
                "[nonfinite_loss_skip_update] step={} batch_idx={} ids={}",
                int(getattr(self, "global_step", -1)),
                int(getattr(self, "_debug_last_batch_idx", -1)),
                self._debug_nonfinite_loss_ids[:32] if self._debug_nonfinite_loss_ids is not None else None,
            )

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

        logger.info(f"DEBUG_TRACE: loss computed batch_idx={batch_idx} t={time.time()-t0:.3f}s")

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
            if hasattr(self.nn, "get_flops"):
                nflops_step = self.nn.get_flops(mask)

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
        
        # Log structure visualization (train + validation)
        log_every_n = self.cfg_exp.log.get("log_structure_every_n_steps", 0)
        if log_every_n > 0:
            if val_step:
                log_id = f"{log_prefix}_{self.global_step}_{batch_idx}"
            else:
                log_id = f"{log_prefix}_{self.global_step}"
            if not hasattr(self, "_last_structure_log_step"):
                self._last_structure_log_step = {}
            already_logged = self._last_structure_log_step.get(log_prefix) == log_id
            should_log = False
            if val_step:
                should_log = batch_idx % log_every_n == 0
            else:
                should_log = self.global_step % log_every_n == 0
            if should_log and not already_logged:
                self._last_structure_log_step[log_prefix] = log_id
                c_pred_full = (
                    self._nn_out_to_c_clean(nn_out, batch)
                    if "contact_map_pred" in nn_out
                    else None
                )
                contact_map_pred = (
                    self._contact_map_to_viz(c_pred_full[0]) if c_pred_full is not None else None
                )
                self._log_structure_visualization(
                    # Log only the first sample. Passing the full batch here makes
                    # `write_prot_to_pdb` treat batch dim as MODEL index, producing a
                    # multi-model PDB with mismatched aatype/mask (confusing in WandB).
                    x_1_pred=x_1_pred[:1] if x_1_pred is not None else None,
                    contact_map_pred=contact_map_pred,
                    mask=mask[0],
                    log_prefix=log_prefix,
                    pair_logits=nn_out.get("pair_logits")[0] if "pair_logits" in nn_out else None,
                    residue_type=(
                        batch.get("residue_type_unmasked", batch.get("residue_type"))[0]
                        if batch.get("residue_type_unmasked", batch.get("residue_type")) is not None
                        else None
                    ),
                    use_template_inference=(
                        getattr(self.nn, "predict_coords", True) is False
                        and getattr(self, "contact_map_mode", False)
                        and self.cfg_exp.model.nn.get("predict_structure_from_distogram", False)
                    ),
                )
                gt_contact_map = (
                    self._contact_map_to_viz(c_1[0]) if c_1 is not None else None
                )
                self._log_structure_visualization(
                    x_1_pred=x_1[:1] if x_1 is not None else None,
                    contact_map_pred=gt_contact_map,
                    mask=mask[0],
                    log_prefix=log_prefix,
                    key_suffix="_gt",
                    residue_type=(
                        batch.get("residue_type_unmasked", batch.get("residue_type"))[0]
                        if batch.get("residue_type_unmasked", batch.get("residue_type"))
                        is not None
                        else None
                    ),
                )

        logger.info(f"DEBUG_TRACE: training_step end batch_idx={batch_idx} t={time.time()-t0:.3f}s")
        return train_loss
    
    def _log_structure_visualization(
        self,
        x_1_pred: torch.Tensor,
        contact_map_pred: torch.Tensor,
        mask: torch.Tensor,
        log_prefix: str,
        key_suffix: str = "",
        pair_logits: torch.Tensor = None,
        residue_type: torch.Tensor = None,
        use_template_inference: bool = False,
    ):
        """
        Logs predicted structure (as temporary PDB) and contact map visualizations.

        Args:
            x_1_pred: Predicted coordinates, shape [1, n, 3]
            contact_map_pred: Predicted contact map, shape [n, n] or None
            mask: Boolean mask, shape [n]
            residue_type: Residue type, shape [n]
            log_prefix: Prefix for log names ("train" or "validation_loss")
            key_suffix: Suffix appended to logged keys (e.g., "_gt")
        """
        if (
            self.logger is None
            or not hasattr(self.logger, "experiment")
            or not hasattr(self.logger.experiment, "log")
        ):
            return

        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        mask = mask.bool()

        # If coords not predicted, try template inference from distogram
        if (
            x_1_pred is None
            and use_template_inference
            and pair_logits is not None
            and residue_type is not None
        ):
            ca_coords, _ = self._predict_structure_from_distogram(
                pair_logits, residue_type, mask
            )
            x_1_pred = ca_coords

        if x_1_pred is not None:
            # Save first sample as temporary PDB for logging
            atom37 = self.samples_to_atom37(x_1_pred, residue_type=residue_type).float().detach().cpu().numpy()
            aatype = residue_type.long().detach().cpu().numpy()
            # Choose an atom mask that matches what the model actually predicts:
            # - CA-flow predicts CA only: mask only CA (avoid writing fake all-atom zeros)
            # - Backbone mode predicts N/CA/C only: mask only N/CA/C
            # - All-atom predictions: use residue-type atom existence mask
            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            if x_1_pred.dim() == 3:
                atom37_mask = np.zeros((aatype.shape[0], 37), dtype=np.float32)
                atom37_mask[:, rc.atom_order["CA"]] = mask_np
            elif x_1_pred.dim() == 4 and x_1_pred.shape[-2] == 3:
                atom37_mask = np.zeros((aatype.shape[0], 37), dtype=np.float32)
                atom37_mask[:, [rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]]] = mask_np[:, None]
            else:
                # Use explicit atom mask derived from residue types, not coordinate magnitudes.
                # This prevents CA atoms at (0,0,0) (common early in training) from being dropped,
                # which breaks WandB structure visualization.
                atom37_mask = rc.RESTYPE_ATOM37_MASK[aatype] * mask_np[:, None]
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
                temp_pdb_path = tmp_pdb.name
            try:
                write_prot_to_pdb(
                    atom37,
                    temp_pdb_path,
                    aatype=aatype,
                    atom37_mask=atom37_mask,
                    overwrite=True,
                    no_indexing=True,
                )
                self.logger.experiment.log(
                    {
                        f"{log_prefix}/structure{key_suffix}": wandb.Molecule(
                            temp_pdb_path
                        ),
                        "global_step": self.global_step,
                    }
                )
            finally:
                if os.path.exists(temp_pdb_path):
                    os.remove(temp_pdb_path)

        if contact_map_pred is None:
            return

        mask_np = mask.detach().cpu().numpy()
        # `contact_map_pred` is expected to already be in [0, 1] for visualization.
        contact_map_prob = contact_map_pred.float().clamp(0.0, 1.0).detach().cpu().numpy()
        pair_mask_np = mask_np[..., :, None] * mask_np[..., None, :]
        contact_map_np = contact_map_prob * pair_mask_np

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
                f"{log_prefix}/contact_map{key_suffix}": wandb.Image(img_contact),
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
        pred_frames_tensor7: Optional[torch.Tensor] = None,
        residue_type: Optional[torch.Tensor] = None,
    ):
        """
        Computes FAPE loss. Supports CA-only, backbone (N,CA,C), and all-atom inputs.
        
        Args:
            x_1: Ground truth coordinates, shape [b, n, 3] or [b, n, natom, 3].
            x_1_pred: Predicted coordinates, same shape as x_1.
            mask: Boolean mask, shape [b, n].
            log_prefix: Optional prefix for logging.
            
        Returns:
            FAPE loss per sample, shape [b].
        """
        if x_1_pred is None:
            raise ValueError("x_1_pred is required for FAPE loss.")

        frames_mask = mask.float()

        if x_1_pred.dim() == 4:
            n_true = x_1[..., 0, :]
            ca_true = x_1[..., 1, :]
            c_true = x_1[..., 2, :]

            # Target frames always from ground truth backbone (stable Gram–Schmidt, AF2 Alg. 21)
            target_frames = Rigid.from_3_points(
                p_neg_x_axis=c_true, origin=ca_true, p_xy_plane=n_true, eps=1e-8
            )

            # Predicted frames:
            # - If pred_frames_tensor7 is provided (StructureModule / IPA path), use those (OpenFold-style)
            # - Otherwise derive frames from predicted N/CA/C via Gram–Schmidt
            if pred_frames_tensor7 is not None:
                if pred_frames_tensor7.shape[-1] != 7:
                    raise ValueError(
                        f"pred_frames_tensor7 must have last dim 7, got shape {pred_frames_tensor7.shape}"
                    )
                pred_frames = Rigid.from_tensor_7(pred_frames_tensor7)
            else:
                # NOTE: This assumes the atom dimension is in atom37/atom14-like order where
                # N=0, CA=1, C=2 for atom37, and that the ground-truth uses the same ordering.
                n_pred = x_1_pred[..., 0, :]
                ca_pred = x_1_pred[..., 1, :]
                c_pred = x_1_pred[..., 2, :]
                pred_frames = Rigid.from_3_points(
                    p_neg_x_axis=c_pred, origin=ca_pred, p_xy_plane=n_pred, eps=1e-8
                )

            pred_positions = x_1_pred.reshape(x_1_pred.shape[0], -1, 3)
            target_positions = x_1.reshape(x_1.shape[0], -1, 3)
            # Positions mask:
            # - Prefer atom-existence mask from residue type (OpenFold-style)
            # - Fall back to residue mask expanded to natoms (legacy behavior)
            bsz, nres, natoms, _ = x_1_pred.shape
            if residue_type is not None:
                residue_type_safe = torch.clamp(
                    residue_type,
                    min=0,
                    max=rc.RESTYPE_ATOM37_MASK.shape[0] - 1,
                )
                if natoms == 37:
                    atom_exists = torch.as_tensor(
                        rc.RESTYPE_ATOM37_MASK,
                        device=x_1_pred.device,
                        dtype=x_1_pred.dtype,
                    )[residue_type_safe]  # [b, n, 37]
                    positions_mask = (
                        atom_exists * mask[..., None].to(dtype=x_1_pred.dtype)
                    ).reshape(bsz, -1)
                elif natoms == 14 and hasattr(rc, "RESTYPE_ATOM14_MASK"):
                    atom_exists = torch.as_tensor(
                        rc.RESTYPE_ATOM14_MASK,
                        device=x_1_pred.device,
                        dtype=x_1_pred.dtype,
                    )[residue_type_safe]  # [b, n, 14]
                    positions_mask = (
                        atom_exists * mask[..., None].to(dtype=x_1_pred.dtype)
                    ).reshape(bsz, -1)
                else:
                    positions_mask = (
                        mask[..., None]
                        .to(dtype=x_1_pred.dtype)
                        .expand(-1, -1, natoms)
                        .reshape(bsz, -1)
                    )
            else:
                positions_mask = (
                    mask[..., None]
                    .to(dtype=x_1_pred.dtype)
                    .expand(-1, -1, natoms)
                    .reshape(bsz, -1)
                )
        elif x_1_pred.dim() == 3:
            # CA-only fallback: identity frames
            rot = torch.eye(3, device=x_1.device, dtype=x_1.dtype)
            rot = rot.view(1, 1, 3, 3).expand(x_1.shape[0], x_1.shape[1], 3, 3)

            pred_rotation = Rotation(rot_mats=rot, quats=None)
            target_rotation = Rotation(rot_mats=rot, quats=None)
            pred_frames = Rigid(rots=pred_rotation, trans=x_1_pred)
            target_frames = Rigid(rots=target_rotation, trans=x_1)

            pred_positions = x_1_pred
            target_positions = x_1
            positions_mask = frames_mask
        else:
            raise ValueError(f"Unsupported x_1_pred shape for FAPE: {x_1_pred.shape}")

        fape = compute_fape(
            pred_frames=pred_frames,
            target_frames=target_frames,
            frames_mask=frames_mask,
            pred_positions=pred_positions,
            target_positions=target_positions,
            positions_mask=positions_mask,
            # Training-space coordinates are in nm (see ang_to_nm in data extraction).
            # OpenFold's defaults are 10Å clamp + 10Å scale, which is 1nm + 1nm here.
            length_scale=1.0,
            l1_clamp_distance=1.0,
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
            self._in_validation_loop = True
            loss = self.training_step(batch, batch_idx=batch_idx)
            self._in_validation_loop = False
            self.validation_output_data.append(loss.item())

        val_sampling_cfg = self.cfg_exp.get("validation_sampling", None)
        if (
            val_sampling_cfg
            and batch_idx == 0
            and self.global_step > 0
            and self._logged_val_traj_epoch != self.current_epoch
        ):
            self._run_validation_trajectory(batch, val_sampling_cfg)
            self._logged_val_traj_epoch = self.current_epoch

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def _run_validation_trajectory(self, batch, val_sampling_cfg):
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        dt = float(val_sampling_cfg.get("dt", 0.005))
        sampling_mode = val_sampling_cfg.get("sampling_mode", "sc")
        sc_scale_noise = float(val_sampling_cfg.get("sc_scale_noise", 0.45))
        sc_scale_score = float(val_sampling_cfg.get("sc_scale_score", 1.0))
        schedule_mode = val_sampling_cfg.get("schedule_mode", "uniform")
        schedule_p = float(val_sampling_cfg.get("schedule_p", 1.0))
        sampling_grid = val_sampling_cfg.get("sampling_grid", None)

        nsamples = 1
        mask = batch["mask"][:nsamples].to(self.device)
        n = mask.shape[-1]
        residue_type = batch.get("residue_type")
        if residue_type is not None:
            residue_type = residue_type[:nsamples].to(self.device)
        cath_code = batch.get("cath_code") if self.cfg_exp.training.get("fold_cond", False) else None
        if cath_code is not None and isinstance(cath_code, torch.Tensor):
            cath_code = cath_code[:nsamples]
        if cath_code is not None and isinstance(cath_code, (list, tuple)):
            cath_code = cath_code[:nsamples]

        x_motif = batch.get("motif_structure", None)
        if x_motif is not None:
            x_motif = x_motif[:nsamples].to(self.device)
        fixed_sequence_mask = batch.get("motif_seq_mask", None)
        if fixed_sequence_mask is not None:
            fixed_sequence_mask = fixed_sequence_mask[:nsamples].to(self.device)

        discrete_enabled = getattr(self, "contact_map_mode", False) and self._discrete_diffusion_enabled()
        prev_sampling_grid = None
        if discrete_enabled and sampling_grid is not None:
            if sampling_grid not in ("uniform", "cosine"):
                raise ValueError(
                    f"validation_sampling.sampling_grid must be 'uniform' or 'cosine', got {sampling_grid!r}"
                )
            prev_sampling_grid = getattr(self.discrete_diffusion, "sampling_grid", None)
            self.discrete_diffusion.sampling_grid = sampling_grid
        try:
            result = self.generate(
                nsamples=nsamples,
                n=n,
                dt=dt,
                self_cond=self.cfg_exp.training.self_cond,
                cath_code=cath_code,
                residue_type=residue_type,
                guidance_weight=1.0,
                autoguidance_ratio=0.0,
                dtype=torch.float32,
                schedule_mode=schedule_mode,
                schedule_p=schedule_p,
                sampling_mode=sampling_mode,
                sc_scale_noise=sc_scale_noise,
                sc_scale_score=sc_scale_score,
                gt_mode="us",
                gt_p=1.0,
                gt_clamp_val=None,
                mask=mask,
                x_motif=x_motif,
                fixed_sequence_mask=fixed_sequence_mask,
                fixed_structure_mask=None,
            )
        finally:
            if prev_sampling_grid is not None:
                self.discrete_diffusion.sampling_grid = prev_sampling_grid

        coords = result.get("coords")
        contact_map = result.get("contact_map")
        distogram = result.get("distogram")

        predict_from_dist = (
            val_sampling_cfg.get("predict_structure_from_distogram", False)
            or self.cfg_exp.model.nn.get("predict_structure_from_distogram", False)
        )
        # In contact-map mode, if the NN doesn't predict coords, default to enabling
        # template inference for validation sampling (so we can log a structure).
        if (
            not predict_from_dist
            and getattr(self, "contact_map_mode", False)
            and getattr(self.nn, "predict_coords", True) is False
        ):
            predict_from_dist = True
        if coords is None and predict_from_dist and distogram is not None and residue_type is not None:
            coords, _ = self._predict_structure_from_distogram(
                distogram,
                residue_type,
                mask,
            )

        self._log_structure_visualization(
            x_1_pred=coords,
            contact_map_pred=self._contact_map_to_viz(contact_map).squeeze(0) if contact_map is not None else None,
            mask=mask.squeeze(0),
            log_prefix="validation_sampling",
            pair_logits=distogram.squeeze(0) if distogram is not None else None,
            residue_type=residue_type.squeeze(0),
            use_template_inference=predict_from_dist,
        )

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag
        if self.discrete_diffusion is not None:
            sampling_grid = inf_cfg.get("sampling_grid", None)
            if sampling_grid is not None:
                if sampling_grid not in ("uniform", "cosine"):
                    raise ValueError(
                        "sampling_grid must be 'uniform' or 'cosine' "
                        f"(got {sampling_grid!r})."
                    )
                self.discrete_diffusion.sampling_grid = sampling_grid
            position_bias = inf_cfg.get("position_bias", None)
            if position_bias is not None:
                enabled = bool(position_bias.get("enabled", False))
                mode = _normalize_position_bias_mode(position_bias.get("mode", "polynomial"))
                w_min = float(position_bias.get("w_min", 0.2))
                w_max = float(position_bias.get("w_max", 5.0))
                k = float(position_bias.get("k", 30.0))
                self.discrete_diffusion.position_bias = dict(position_bias)
                self.discrete_diffusion.position_bias_enabled = enabled
                self.discrete_diffusion.position_bias_mode = mode
                self.discrete_diffusion.position_bias_w_min = w_min
                self.discrete_diffusion.position_bias_w_max = w_max
                self.discrete_diffusion.position_bias_k = k

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
        force_compile = getattr(self, "_force_compile", False)
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

        save_trajectory = bool(self.inf_cfg.get("save_trajectory", False))
        save_trajectory_gif = bool(self.inf_cfg.get("save_trajectory_gif", False))
        return_trajectory = save_trajectory or save_trajectory_gif
        trajectory_stride = int(self.inf_cfg.get("trajectory_stride", 1))
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
            mask=mask,
            x_motif=x_motif,
            fixed_sequence_mask=fixed_sequence_mask,
            fixed_structure_mask=fixed_structure_mask,
            force_compile=force_compile,
            return_trajectory=return_trajectory,
            trajectory_stride=trajectory_stride,
        )
        cirpin_ids = batch.get("cirpin_ids", [None] * len(cath_code))
        coords_atom37 = None
        if result.get("coords") is not None:
            coords_atom37 = self.samples_to_atom37(result["coords"])
        distogram = result.get("distogram")

        predict_from_dist = self.inf_cfg.get("predict_structure_from_distogram", False)
        if (
            coords_atom37 is None
            and predict_from_dist
            and distogram is not None
            and residue_type is not None
            and mask is not None
        ):
            ca_coords, atom37 = self._predict_structure_from_distogram(
                distogram,
                residue_type.to(self.device),
                mask.to(self.device),
            )
            coords_atom37 = atom37

        return {
            "coords_atom37": coords_atom37,
            "contact_map": result.get("contact_map"),
            "distogram": distogram,
            "cath_code": cath_code,
            "cirpin_ids": cirpin_ids,
            "trajectory_tokens": result.get("trajectory_tokens"),
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
        force_compile: bool = False,
        return_trajectory: bool = False,
        trajectory_stride: int = 1,
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
            force_compile=force_compile,
        )
        if torch.is_tensor(residue_type) and residue_type.dim() == 2 and residue_type.shape[0] == 1 and nsamples > 1:
            residue_type = residue_type.expand(nsamples, -1)
        if mask is None:
            mask = torch.ones(nsamples, n).long().bool().to(self.device)

        contact_map_mode = getattr(self, "contact_map_mode", False)
        discrete_enabled = contact_map_mode and self._discrete_diffusion_enabled()
        if discrete_enabled:
            if self.discrete_diffusion_type != "md4":
                raise ValueError("GenMD4 sampling is not implemented for inference.")
            timesteps = self.discrete_diffusion.timesteps
            pair_mask = self._pair_mask(mask)
            zt = self.discrete_diffusion.prior_sample(
                (nsamples, n, n), device=self.device
            )
            contact_map_sc = None
            distogram = None
            coords = None
            c_logits = None
            stride = max(1, int(trajectory_stride))
            trajectory_tokens = [] if return_trajectory else None
            for i in range(timesteps):
                s, t = self.discrete_diffusion.get_sampling_grid(i, timesteps)
                t_tensor = torch.full((nsamples,), t, device=self.device, dtype=torch.float32)
                contact_map_t = self._contact_tokens_to_input(zt)
                contact_map_t = self._apply_pair_mask(contact_map_t, pair_mask)
                nn_in = {
                    "x_t": torch.zeros(
                        nsamples, n, 3, device=self.device, dtype=dtype or torch.float32
                    ),
                    "contact_map_t": contact_map_t,
                    "t": t_tensor,
                    "mask": mask,
                }
                if cath_code is not None:
                    nn_in["cath_code"] = cath_code
                if residue_type is not None:
                    nn_in["residue_type"] = residue_type
                if fixed_sequence_mask is not None:
                    nn_in["fixed_sequence_mask"] = fixed_sequence_mask
                if fixed_structure_mask is not None:
                    nn_in["fixed_structure_mask"] = fixed_structure_mask
                if x_motif is not None:
                    nn_in["x_motif"] = x_motif
                if self_cond:
                    if contact_map_sc is None:
                        contact_map_sc = torch.zeros(
                            nsamples,
                            n,
                            n,
                            device=self.device,
                            dtype=contact_map_t.dtype,
                        )
                    nn_in["contact_map_sc"] = contact_map_sc
                result = predict_clean_n_v_w_guidance(nn_in)
                c_logits = result.get("contact_map_logits")
                if c_logits is None:
                    raise ValueError(
                        "discrete diffusion sampling requires contact_map_logits in model output."
                    )
                zt = self.discrete_diffusion.sample_step(zt, c_logits, s, t)
                if return_trajectory and (i % stride == 0):
                    trajectory_tokens.append(zt.detach().cpu())
                distogram = result.get("distogram")
                coords = result.get("coords")
                if self_cond:
                    c_pred = result.get("contact_map")
                    if c_pred is not None:
                        contact_map_sc = c_pred.detach()
            if c_logits is None:
                raise ValueError("discrete diffusion sampling produced no logits.")
            z0 = self.discrete_diffusion.decode(zt, c_logits)
            contact_map = self._contact_tokens_to_output(z0)
            contact_map = self._apply_pair_mask(contact_map, pair_mask)
            result = {"coords": coords, "contact_map": contact_map, "distogram": distogram}
            if return_trajectory:
                if trajectory_tokens:
                    result["trajectory_tokens"] = torch.stack(trajectory_tokens, dim=0)
                else:
                    result["trajectory_tokens"] = torch.empty(
                        (0, nsamples, n, n), dtype=torch.long
                    )
            return result
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
