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
import shutil
import subprocess
import tempfile
import time

from abc import abstractmethod
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

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
from proteinfoundation.datasets.cath_utils import (
    apply_fold_mask_to_indices,
    load_cath_mapping,
)
from proteinfoundation.utils.ff_utils.pdb_utils import (
    mask_ext_lig_blocky,
    mask_seq,
    write_prot_to_pdb,
)
from proteinfoundation.prediction_pipeline.usalign_tabular import (
    parse_usalign_pair_outfmt2,
)
# from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference, OpenFoldDistogramOnlyInference
import proteinfoundation.openfold_stub.np.residue_constants as rc
from proteinfoundation.openfold_stub.utils.loss import compute_fape
from proteinfoundation.openfold_stub.utils.rigid_utils import Rigid, Rotation

class ModelTrainerBase(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None):
        super(ModelTrainerBase, self).__init__()
        # Non-strict model-weight loading: lets a checkpoint resume even after new
        # submodules are wired in (e.g. a previously-dead feature-embedding gate) --
        # the new params get their fresh random init instead of crashing the resume.
        self.strict_loading = False
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

        # Lazy cache for the ext_lig "smart unknown" augmentation stats (see
        # _get_ext_lig_augment_stats / mask_ext_lig_blocky)
        self._ext_lig_augment_stats = None

        # NOTE on temp PDB files handed to wandb.Molecule() in _log_structure_visualization:
        # wandb does NOT read the file synchronously at construction time -- confirmed by two
        # separate live crashes this session, one of which survived a "delete on the next call"
        # deferral (i.e. even a full inter-call gap, >= log_structure_every_n_steps steps, wasn't
        # always enough -- wandb's own internal copy can be delayed further than that, e.g. by
        # its background queue). These temp files are therefore intentionally NEVER deleted by
        # this code anymore (see _log_structure_visualization) -- correctness beats tidiness here;
        # a handful of small PDB text files accumulating in node-local /tmp over a run is
        # negligible and the directory is wiped when the SLURM job's node session ends anyway.

        # Attributes re-written by classes that inherit from this one
        self.nn = None
        self.fm = None

        # For autoguidance, overridden in `self.configure_inference`
        self.nn_ag = None
        self.motif_conditioning = cfg_exp.training.get("motif_conditioning", False)
        self.discrete_diffusion = None
        self.discrete_diffusion_type = None
        self._init_discrete_diffusion()
        self.dssp_diffusion = None
        self.dssp_diffusion_type = None
        self._init_dssp_diffusion()

        # Lazy OpenFold template inference helper
        self._template_inference = None
        self._logged_val_traj_epoch = -1
        self._self_cond_copy_last_step = None

    def _create_optimizer(self, params, optimizer_type=None):
        """Create an optimizer instance based on the configured type.

        Args:
            params: Parameter list or iterable for Adam/AdamW.  Ignored when
                ``optimizer_type="muon"`` (parameter partitioning is done
                internally).
            optimizer_type: One of ``"adam"``, ``"adamw"``, ``"muon"``.
                If ``None``, read from ``self.cfg_exp.opt.optimizer``
                (default ``"adam"``).

        Returns:
            A ``torch.optim.Optimizer`` instance.
        """
        if optimizer_type is None:
            optimizer_type = self.cfg_exp.opt.get("optimizer", "adam")

        lr = self.cfg_exp.opt.lr

        if optimizer_type == "muon":
            from proteinfoundation.optim.param_groups import build_optimizer_param_groups
            from proteinfoundation.optim.muon import HybridMuonAdamW

            param_groups = build_optimizer_param_groups(self, self.cfg_exp.opt)
            optimizer = HybridMuonAdamW(param_groups)
        elif optimizer_type == "adamw":
            weight_decay = float(self.cfg_exp.opt.get("weight_decay", 0.0))
            adam_betas_raw = self.cfg_exp.opt.get("adam_betas", [0.9, 0.999])
            adam_betas = tuple(float(b) for b in adam_betas_raw)
            adam_eps = float(self.cfg_exp.opt.get("adam_eps", 1e-8))
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=adam_betas,
                eps=adam_eps,
            )
        else:  # "adam" (default — preserves original behaviour)
            optimizer = torch.optim.Adam(params, lr=lr)

        return optimizer

    def configure_optimizers(self):
        optimizer_type = self.cfg_exp.opt.get("optimizer", "adam")

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
            # LoRA finetune: always use Adam/AdamW (Muon not beneficial
            # for the small/1-D LoRA params).
            finetune_opt_type = optimizer_type if optimizer_type != "muon" else "adamw"
            optimizer = self._create_optimizer(opt_params, finetune_opt_type)
            print(f"Finetuning {opt_params_names}")
        else:
            for name, param in self.named_parameters():
                if name.startswith("nn_sc."):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = self._create_optimizer(trainable_params, optimizer_type)
        
        # Check if learning rate warmup is enabled
        if self.cfg_exp.opt.get("use_lr_warmup", False):
            warmup_steps = self.cfg_exp.opt.get("warmup_steps", 1000)
            warmup_start_lr_ratio = self.cfg_exp.opt.get("warmup_start_lr_ratio", 0.0)
            scheduler_type = self.cfg_exp.opt.get("lr_scheduler_type", "constant")
            
            # Calculate total training steps dynamically from max_epochs and dataloader
            total_steps = None
            if scheduler_type in ["linear_decay", "cosine"]:
                # Opt-in: pin the LR-anneal horizon to max_steps (the hard stop) so LR reaches 0 exactly at
                # training end. Avoids the max_epochs * len(dataloader) estimate below, which over-counts
                # steps/epoch (len(dataloader) != actual iterated batches) and mis-sizes the horizon.
                if self.cfg_exp.opt.get("lr_horizon_from_max_steps", False):
                    ms = getattr(self.trainer, "max_steps", -1) if (hasattr(self, 'trainer') and self.trainer is not None) else -1
                    if ms and ms > 0:
                        total_steps = int(ms)
                        logger.info(f"LR horizon total_training_steps = {total_steps} (pinned to max_steps; lr_horizon_from_max_steps=True)")
                # Try to get dataloader length and max_epochs (fallback when not pinned to max_steps)
                if total_steps is None and hasattr(self, 'trainer') and self.trainer is not None:
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
        # Alias mode: nn_sc shares parameter tensors with self.nn, so a periodic
        # state_dict copy would at best be Tensor.copy_(self) (a no-op that still
        # adds dispatcher overhead). Detect via tensor identity on the first
        # registered parameter and skip when params are aliased.
        nn_first = next(iter(self.nn.parameters()), None)
        sc_first = next(iter(nn_sc.parameters()), None)
        if nn_first is not None and sc_first is not None and nn_first is sc_first:
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
        """Clears gradients when a non-finite loss was detected for this backward.

        Also implements the FAPE/IPA debug logging — when
        ``opt.debug_log_grads`` is True, prints per-N-step gradient norms for
        ``ipa_linear_s`` / ``ipa_linear_z`` plus the captured activation
        gradients (``pair_rep`` going into the IPA pair projector, and the
        projector output itself). Use this to confirm the frozen-IPA-with-
        trainable-projectors setup actually receives gradient signal on the
        projector layers (see project_proteina_data_engineering_plan.md →
        FAPE/IPA debug).

        When ``opt.debug_log_unused_params`` is True, also dumps a one-shot
        listing of parameters that received `grad is None` or `grad == 0`
        after the FIRST backward. Used to debug the DDP
        find_unused_parameters requirement.
        """
        if getattr(self, "global_rank", 0) != 0:
            return
        if getattr(self, "_skip_update_due_to_nonfinite_loss", False):
            self.zero_grad()

        # Unused-params diagnostic: scan EVERY step. The first step prints a
        # full report. Subsequent steps print only newly-unused params (sticky
        # accumulator) so we catch stochastic cases (e.g. self_cond ON vs OFF,
        # aux_loss_t_lim threshold).
        if bool(self.cfg_exp.opt.get("debug_log_unused_params", False)):
            self._dump_unused_params_sticky()

        if not bool(self.cfg_exp.opt.get("debug_log_grads", False)):
            return
        # Cadence: 1 (every step) is useful for the smoke test; higher for prod.
        cadence = int(self.cfg_exp.opt.get("debug_log_grads_every", 100))
        if cadence < 1:
            cadence = 1
        if int(self.global_step) % cadence != 0:
            return

        parts = [f"[FAPE-IPA-debug step={int(self.global_step)}]"]
        # Projector weight + bias gradients (leaf-param grads accumulate after
        # backward). Reports both so a zero weight.grad with a nonzero bias.grad
        # can be spotted — that was the smoking gun for the IPA-mask sign bug
        # (see openfold_stub/model/structure_module.py:361).
        for name in ("ipa_linear_s", "ipa_linear_z"):
            lin = getattr(self.nn, name, None)
            if lin is None or getattr(lin, "weight", None) is None:
                parts.append(f"{name}=missing")
                continue
            wg = lin.weight.grad
            bg = lin.bias.grad if lin.bias is not None else None
            parts.append(
                f"{name}.weight |grad|={(wg.norm().item() if wg is not None else float('nan')):.4e} "
                f"{name}.bias |grad|={(bg.norm().item() if bg is not None else float('nan')):.4e}"
            )
        print(" ".join(parts))

    def _dump_unused_params_sticky(self) -> None:
        """Run-every-step unused-params tracker.

        Maintains two cumulative sets across steps:
          - `_dbg_none_ever`: parameter names whose `.grad` was `None` on at
            least one step. These are the DDP `find_unused_parameters`
            culprits.
          - `_dbg_seen_nonzero`: parameter names that have received a non-zero
            gradient at least once. Param NOT in this set means it never got a
            real gradient (always zero).
        """
        step = int(getattr(self, "global_step", 0))
        if not hasattr(self, "_dbg_none_ever"):
            self._dbg_none_ever = set()
            self._dbg_seen_nonzero = set()
            self._dbg_first_step_dumped = False
        new_none = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            g = p.grad
            if g is None:
                if name not in self._dbg_none_ever:
                    new_none.append(name)
                self._dbg_none_ever.add(name)
            else:
                if g.abs().sum().item() > 0:
                    self._dbg_seen_nonzero.add(name)
        if not self._dbg_first_step_dumped:
            self._dbg_first_step_dumped = True
            self._dump_unused_params()  # full report on step 0
        if new_none:
            print(f"[unused-params-debug step={step}] NEW grad-is-None params (n={len(new_none)}):")
            for nm in new_none[:20]:
                print(f"    NONE  {nm}")

    def _dump_unused_params(self) -> None:
        """List parameters that didn't receive a gradient on the first backward.

        Distinguishes three states:
          - `grad is None`: parameter was NOT touched by backward → DDP "unused"
          - `grad.abs().sum() == 0`: touched but gradient is identically zero
            (e.g. multiplied by 0.0 in a forward-graph tying trick)
          - `grad.abs().sum() > 0`: real gradient flow

        DDP without ``find_unused_parameters=True`` errors out on params in the
        first category (grad is None). Knowing which they are lets us either
        rewrite the forward to always touch them, or accept ``find_unused_parameters=True``.
        """
        from collections import defaultdict
        none_grad = []
        zero_grad = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                none_grad.append((name, tuple(p.shape)))
            elif p.grad.abs().sum().item() == 0.0:
                zero_grad.append((name, tuple(p.shape)))
        print(f"\n[unused-params-debug] === parameter gradient status after first backward ===")
        print(f"[unused-params-debug] total trainable params: "
              f"{sum(1 for _, p in self.named_parameters() if p.requires_grad)}")
        print(f"[unused-params-debug] grad is None: {len(none_grad)}")
        for nm, sh in none_grad[:40]:
            print(f"    NONE  {nm}  shape={sh}")
        if len(none_grad) > 40:
            print(f"    ... and {len(none_grad) - 40} more (suppressed)")
        # Group zero-grad by top-level module prefix to make the output readable.
        zero_by_prefix = defaultdict(list)
        for nm, sh in zero_grad:
            zero_by_prefix[nm.split(".")[0] + "." + (nm.split(".")[1] if "." in nm[len(nm.split(".")[0]):] else "")].append(nm)
        print(f"[unused-params-debug] grad == 0 (touched but zero): {len(zero_grad)}")
        for pfx, names in list(zero_by_prefix.items())[:20]:
            print(f"    ZERO  prefix={pfx!r}  count={len(names)}  example={names[0]}")
        print(f"[unused-params-debug] ====================================================\n")

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

    def _eval_forward_module(self, force_compile: bool = False):
        """Module to run a no-grad eval forward through.

        Alias nn_sc shares live weights with self.nn (EMA copies in place) but
        owns a separate Dynamo cache (_forward_impl_sc), so routing validation
        forwards through it gives compile speedup without adding grad/no-grad
        variants to the training graph's cache. Dedicated inference
        (force_compile=True) keeps self.nn's force_compile eval path.
        deepcopy/none modes and grad-enabled (training) fall back to self.nn.
        """
        if (
            not force_compile
            and not torch.is_grad_enabled()
            and getattr(self, "_nn_sc_aliased", False)
            and self.nn_sc is not None
        ):
            return self.nn_sc
        return self.nn

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
        return self._eval_forward_module()(batch)

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
        #         "openfold_jax_params_path", os.path.expanduser("~/openfold/openfold/resources/params/params_model_1_ptm.npz")
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
            distogram_probs = torch.softmax(pair_logits.contiguous(), dim=-1)

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

        # Zero out sinusoidal positional embedding during inference if configured
        if getattr(self, "_inf_zero_sin_pos_emb", False):
            batch["_zero_idx_emb"] = True

        # Joint-discontinuous: inject per-residue position/break feats (stashed in predict_step)
        _xt = batch.get("x_t")
        _nb = _xt.shape[0] if _xt is not None else None
        for _k, _attr in (("residue_pdb_idx", "_gen_residue_pdb_idx"), ("chain_breaks_per_residue", "_gen_chain_breaks")):
            _v = getattr(self, _attr, None)
            if _v is not None:
                if _v.dim() == 2 and _v.shape[0] == 1 and _nb and _nb > 1:
                    _v = _v.expand(_nb, -1)
                batch[_k] = _v.to(_xt.device) if _xt is not None else _v

        eval_nn = self._eval_forward_module(force_compile)
        nn_out = eval_nn(batch, force_compile=force_compile)
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
                    "cath_code" in batch or "cath_code_indices" in batch
                ), "Only support CFG when cath_code or cath_code_indices is provided"
                uncond_batch = batch.copy()
                uncond_batch.pop("cath_code", None)
                uncond_batch.pop("cath_code_indices", None)
                uncond_batch.pop("cath_code_indices_mask", None)
                nn_out_uncond = eval_nn(uncond_batch)
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
            result["distogram"] = torch.softmax(nn_out["pair_logits"].contiguous(), dim=-1)

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
        elif diffusion_type == "udlm":
            from proteinfoundation.flow_matching import UDLMDiscreteDiffusion
            udlm_cfg = cfg.get("udlm", cfg.get("md4", {}))
            self.discrete_diffusion = UDLMDiscreteDiffusion(
                vocab_size=udlm_cfg.get("vocab_size", 2),
                noise_schedule_type=udlm_cfg.get("noise_schedule", "cosine"),
                timesteps=udlm_cfg.get("timesteps", 1000),
                cont_time=udlm_cfg.get("cont_time", True),
                sampling_grid=udlm_cfg.get("sampling_grid", "cosine"),
                eps=udlm_cfg.get("eps", 1e-4),
                position_bias=position_bias,
                symmetrize=True,  # Contact maps are symmetric
            )
        else:
            raise ValueError(
                f"Unknown discrete diffusion type {diffusion_type}. "
                "Use 'md4', 'genmd4', or 'udlm'."
            )
        self.discrete_diffusion_type = diffusion_type

    def _get_dssp_diffusion_cfg(self):
        cfg = self.cfg_exp.model.get("dssp_diffusion", None)
        if cfg is None or not cfg.get("enabled", False):
            return None
        return cfg

    def _init_dssp_diffusion(self):
        """Initialize DSSP discrete diffusion (1D per-residue sequence)."""
        cfg = self._get_dssp_diffusion_cfg()
        if cfg is None:
            self.dssp_diffusion = None
            self.dssp_diffusion_type = None
            return
        diffusion_type = cfg.get("type", "md4")
        dssp_cfg = cfg.get(diffusion_type, cfg)
        vocab_size = int(dssp_cfg.get("vocab_size", 3))  # loop, helix, strand
        if diffusion_type == "md4":
            self.dssp_diffusion = MD4DiscreteDiffusion(
                vocab_size=vocab_size,
                noise_schedule_type=dssp_cfg.get("noise_schedule", "cosine"),
                timesteps=dssp_cfg.get("timesteps", 1000),
                cont_time=dssp_cfg.get("cont_time", True),
                sampling_grid=dssp_cfg.get("sampling_grid", "cosine"),
                eps=dssp_cfg.get("eps", 1e-4),
                symmetrize=False,  # DSSP is 1D, no symmetrization
            )
        elif diffusion_type == "udlm":
            from proteinfoundation.flow_matching import UDLMDiscreteDiffusion
            self.dssp_diffusion = UDLMDiscreteDiffusion(
                vocab_size=vocab_size,
                noise_schedule_type=dssp_cfg.get("noise_schedule", "cosine"),
                timesteps=dssp_cfg.get("timesteps", 1000),
                cont_time=dssp_cfg.get("cont_time", True),
                sampling_grid=dssp_cfg.get("sampling_grid", "cosine"),
                eps=dssp_cfg.get("eps", 1e-4),
                symmetrize=False,  # DSSP is 1D, no symmetrization
            )
        else:
            raise ValueError(
                f"Unknown DSSP diffusion type {diffusion_type}. "
                "Use 'md4' or 'udlm'."
            )
        self.dssp_diffusion_type = diffusion_type

    def _discrete_diffusion_enabled(self) -> bool:
        return self.discrete_diffusion is not None

    def _dssp_diffusion_enabled(self) -> bool:
        return getattr(self, "dssp_diffusion", None) is not None

    def _compute_discrete_input_dims(self) -> dict:
        """Compute NN input dimensions from diffusion config.

        Returns a dict with 'contact_map_input_dim' and 'dssp_input_dim' and
        'dssp_num_classes' derived from the active diffusion modules.

        For UDLM (uniform): vocab_size (no mask token).
        For MD4/GenMD4 (absorbing): vocab_size + 1 (extra mask token).
        """
        dims = {}
        if self._discrete_diffusion_enabled():
            dd = self.discrete_diffusion
            if dd.mask_token is not None:
                # Absorbing diffusion: one-hot over vocab_size + 1 (mask)
                dims["contact_map_input_dim"] = dd.vocab_size + 1
            else:
                # UDLM: one-hot over vocab_size only
                dims["contact_map_input_dim"] = dd.vocab_size
        if self._dssp_diffusion_enabled():
            dd = self.dssp_diffusion
            if dd.mask_token is not None:
                dims["dssp_input_dim"] = dd.vocab_size + 1
            else:
                dims["dssp_input_dim"] = dd.vocab_size
            # dssp_num_classes: always the actual vocab size (for output head)
            dims["dssp_num_classes"] = dd.vocab_size
        return dims

    def _pair_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return mask[..., :, None] * mask[..., None, :]

    def _apply_pair_mask(self, pair_tensor: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        mask = pair_mask.float()
        if pair_tensor.dim() == mask.dim() + 1:
            mask = mask.unsqueeze(-1)
        return pair_tensor * mask

    def _contact_tokens_to_input(self, tokens: torch.Tensor) -> torch.Tensor:
        input_dim = int(getattr(self.nn, "contact_map_input_dim", 1))
        is_uniform = self.discrete_diffusion_type == "udlm"
        if input_dim == 1:
            non_contact_value = getattr(self.nn, "non_contact_value", 0)
            if non_contact_value == -1:
                if is_uniform:
                    # Uniform: tokens are always valid vocab {0, 1}
                    contact = torch.ones_like(tokens, dtype=torch.float32)
                    non_contact = -torch.ones_like(tokens, dtype=torch.float32)
                    return torch.where(tokens == 1, contact, non_contact)
                else:
                    contact = torch.ones_like(tokens, dtype=torch.float32)
                    non_contact = -torch.ones_like(tokens, dtype=torch.float32)
                    mask_val = torch.zeros_like(tokens, dtype=torch.float32)
                    return torch.where(
                        tokens == self.discrete_diffusion.mask_token,
                        mask_val,
                        torch.where(tokens == 1, contact, non_contact),
                    )
            return tokens.float()
        if is_uniform:
            # Uniform mode: one-hot over vocab_size (no mask token)
            num_classes = self.discrete_diffusion.vocab_size
            result = torch.nn.functional.one_hot(tokens, num_classes=num_classes).float()
            # Pad to input_dim if needed
            if num_classes < input_dim:
                pad = torch.zeros(*result.shape[:-1], input_dim - num_classes,
                                  device=result.device, dtype=result.dtype)
                result = torch.cat([result, pad], dim=-1)
            return result
        if input_dim == 3:
            num_classes = self.discrete_diffusion.vocab_size + 1  # +1 for mask token
            return torch.nn.functional.one_hot(tokens, num_classes=num_classes).float()
        raise ValueError(
            f"Unsupported contact_map_input_dim {input_dim}. Expected 1 or 3."
        )

    def _dssp_tokens_to_input(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert DSSP diffusion tokens to network input (one-hot).

        For absorbing mode (mask_token exists): one-hot over vocab_size + 1
        For uniform mode (no mask token): one-hot over vocab_size
        """
        dssp_vocab = self.dssp_diffusion.vocab_size
        if self.dssp_diffusion.mask_token is not None:
            num_classes = dssp_vocab + 1  # +1 for mask token
        else:
            num_classes = dssp_vocab
        return torch.nn.functional.one_hot(
            tokens.long(), num_classes=num_classes
        ).float()

    def _dssp_tokens_to_output(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert DSSP tokens to clean output representation (long tensor)."""
        return tokens.long()

    def _contact_tokens_to_output(self, tokens: torch.Tensor) -> torch.Tensor:
        non_contact_value = getattr(self.nn, "non_contact_value", 0)
        if non_contact_value == -1:
            contact = torch.ones_like(tokens, dtype=torch.float32)
            non_contact = -torch.ones_like(tokens, dtype=torch.float32)
            return torch.where(tokens == 1, contact, non_contact)
        return tokens.float()

    def _set_self_cond(self, batch: Dict, target_clean: torch.Tensor,
                       contact_map_mode: bool, use_self_cond: bool):
        """Set self-conditioning input in batch (zeros or model prediction).

        Initializes the SC key to zeros first so all feature params are exercised,
        then optionally runs an SC forward to produce a prediction.

        When DSSP diffusion is enabled, also sets dssp_sc using the shared SC flag.
        """
        sc_key = "contact_map_sc" if contact_map_mode else "x_sc"
        batch[sc_key] = torch.zeros_like(target_clean)

        # Initialize DSSP SC to zeros if DSSP diffusion is enabled
        dssp_enabled = self._dssp_diffusion_enabled()
        if dssp_enabled:
            b, n_res = batch["mask"].shape
            dssp_vocab = self.dssp_diffusion.vocab_size
            batch["dssp_sc"] = torch.zeros(
                b, n_res, dssp_vocab,
                device=batch["mask"].device,
                dtype=target_clean.dtype,
            )

        if use_self_cond:
            with torch.no_grad():
                self._maybe_update_self_cond_copy()
                sc_model = getattr(self, "nn_sc", None) or self.nn
                nn_out_sc = sc_model(batch)
                if contact_map_mode:
                    pred_sc = self._nn_out_to_c_clean(nn_out_sc, batch)
                else:
                    pred_sc = self._nn_out_to_x_clean(nn_out_sc, batch)
            # PyTorch #105211: under an outer autocast context (Lightning bf16-mixed),
            # weights cast inside the no_grad SC pass get cached with requires_grad=False
            # and reuse in the next grad forward disconnects params from autograd. Only
            # bites when sc_model is self.nn (no-deepcopy); harmless otherwise.
            torch.clear_autocast_cache()
            if pred_sc is not None:
                batch[sc_key] = self.detach_gradients(pred_sc)
            # DSSP SC: use softmax of dssp logits as probability distribution
            if dssp_enabled:
                dssp_logits_sc = nn_out_sc.get("dssp_logits")
                if dssp_logits_sc is not None:
                    batch["dssp_sc"] = torch.softmax(
                        dssp_logits_sc, dim=-1
                    ).detach()

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

    def _get_ext_lig_augment_stats(self) -> dict:
        """Lazily load and cache the ext_lig 'smart unknown' augmentation artifacts
        (script_utils/precompute_ext_lig_augment_stats.py) -- real per-chain frac_present
        pool + adaptive by-residue-weighted run-length buckets."""
        if self._ext_lig_augment_stats is None:
            frac_path = self.cfg_exp.training.ext_lig_frac_present_stats_path
            runlen_path = self.cfg_exp.training.ext_lig_runlen_stats_path
            frac_present_pool = np.load(frac_path)
            runlen = np.load(runlen_path)
            self._ext_lig_augment_stats = {
                "frac_present_pool": frac_present_pool,
                "runlen_bucket_lo": runlen["bucket_lo"],
                "runlen_bucket_hi": runlen["bucket_hi"],
                "runlen_bucket_weight": runlen["bucket_weight"],
            }
            logger.info(
                f"Loaded ext_lig smart-augment stats: {len(frac_present_pool)} chains from "
                f"{frac_path}, {len(runlen['bucket_lo'])} runlen buckets from {runlen_path}"
            )
        return self._ext_lig_augment_stats

    def _log_ext_lig_augment_monitoring(
        self, before: Tensor, after: Tensor, mask: Tensor, augment_stats: dict
    ) -> None:
        """Before/after ext_lig label distribution + sampled block stats, so a
        misbehaving augmentation (wrong fraction, degenerate all-singleton blocks, etc.)
        is visible in wandb rather than silently degrading training."""
        valid = mask.bool()

        def frac(labels, value):
            return (labels[valid] == value).float().mean()

        log_kwargs = dict(on_step=True, on_epoch=False, prog_bar=False, logger=True,
                           batch_size=mask.shape[0], sync_dist=True)
        for name, value in [
            ("present_before", frac(before, 1)),
            ("absent_before", frac(before, 0)),
            ("unknown_before", frac(before, 2)),
            ("present_after", frac(after, 1)),
            ("absent_after", frac(after, 0)),
            ("unknown_after", frac(after, 2)),
            ("sampled_fraction", augment_stats["mean_fraction"]),
            ("n_pieces", augment_stats["mean_n_pieces"]),
            ("piece_length", augment_stats["mean_piece_length"]),
        ]:
            self.log(f"ext_lig_augment/{name}", value, **log_kwargs)

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
        self._debug_last_batch_idx = int(batch_idx)
        self._debug_nonfinite_loss_ids = None
        self._skip_update_due_to_nonfinite_loss = False
        val_step = batch_idx == -1 or getattr(self, "_in_validation_loop", False)
        log_prefix = "validation_loss" if val_step else "train"
        # Rate-limited diag heartbeat: every batch during val (rare, slow path),
        # every 100 batches during training. Use to spot which rank stalls before
        # the next gloo/nccl barrier when validation deadlocks.
        if val_step or (int(batch_idx) % 100 == 0):
            self._diag_log("training_step enter", f"val={val_step} batch_idx={batch_idx}")
        
        # Extract inputs from batch (our dataloader)
        # This may apply augmentations, if requested in the config file
        x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        
        # Center and mask input
        x_1 = self.fm._mask_and_zero_com(x_1, mask)

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
        dssp_enabled = self._dssp_diffusion_enabled()
        dssp_tokens = None
        dssp_zt = None

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
                zt = self.discrete_diffusion.forward_sample(c_tokens, t, pair_mask=pair_mask)
                if self.discrete_diffusion_type == "genmd4":
                    zt_2 = self.discrete_diffusion.forward_sample(c_tokens, t, pair_mask=pair_mask)
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
        
        # DSSP discrete diffusion: corrupt DSSP tokens using shared timestep t
        if dssp_enabled:
            dssp_target = batch.get("dssp_target")
            if dssp_target is None:
                raise ValueError(
                    "dssp_target not found in batch. Ensure DSSPTargetTransform is enabled."
                )
            # dssp_target is [b, n] with values in {0, 1, 2} and -1 for padding
            dssp_tokens = dssp_target.long()
            # Replace padding (-1) with 0 to avoid indexing errors; mask handles this
            dssp_tokens = dssp_tokens.clamp(min=0)
            dssp_tokens = dssp_tokens * mask.long()
            dssp_zt = self.dssp_diffusion.forward_sample(dssp_tokens, t, pair_mask=mask)
            dssp_zt = dssp_zt * mask.long()
            dssp_t_input = self._dssp_tokens_to_input(dssp_zt)
            dssp_t_input = dssp_t_input * mask[..., None].float()
            batch["dssp_t"] = dssp_t_input

        if self.motif_conditioning:
            batch.update(self.motif_factory(batch))
            x_1 = batch["x_1"] # we need this since we change x_1 based n the motif center
        # Add a few things to batch, needed for nn
        batch["t"] = t
        batch["mask"] = mask
        batch["x_t"] = x_t

        # Fold conditional training: apply progressive masking to cath_code_indices
        if self.cfg_exp.training.fold_cond:
            idx = batch.get("cath_code_indices")
            if idx is not None and isinstance(idx, torch.Tensor):
                cath_code_dir = self.cfg_exp.model.nn.get("cath_code_dir")
                if cath_code_dir:
                    _, _, _, nC, nA, nT = load_cath_mapping(cath_code_dir)
                    bs = idx.shape[0]
                    result = idx.clone()
                    for i in range(bs):
                        mask_T = random.random() < self.cfg_exp.training.mask_T_prob
                        mask_A = random.random() < self.cfg_exp.training.mask_A_prob
                        mask_C = random.random() < self.cfg_exp.training.mask_C_prob
                        result[i] = apply_fold_mask_to_indices(
                            idx[i : i + 1],
                            mask_T, mask_A, mask_C,
                            nC, nA, nT,
                        ).squeeze(0)
                    batch.cath_code_indices = result
            if "cath_code" in batch:
                batch.pop("cath_code")
        else:
            if "cath_code" in batch:
                batch.pop("cath_code")
            if "cath_code_indices" in batch:
                batch.pop("cath_code_indices")

        # Sequence conditional training
        if self.cfg_exp.training.seq_cond:
            # Get the sequence from the batch
            seq = batch["residue_type"]
            # Avoid in-place modification for better compilation support
            seq = torch.where(seq == -1, torch.tensor(20, device=seq.device, dtype=seq.dtype), seq)

            # Preserve the unmasked sequence for logging / IPA geometry (do this before masking)
            if "residue_type_unmasked" not in batch:
                residue_type_unmasked = seq.clone().detach()
                batch["residue_type_unmasked"] = residue_type_unmasked

            # Mask the sequence (vectorized)
            seq = mask_seq(seq, mask, self.cfg_exp.training.mask_seq_proportion, mask_value=20)
            
            batch["residue_type"] = seq
        else:
            # Keep residue_type if IPA coordinates are needed for contact_map_mode
            need_residue_type = (
                getattr(self, "contact_map_mode", False)
                and getattr(self.nn, "predict_coords", None) == "ipa"
            )
            if "residue_type" in batch and not need_residue_type:
                batch.pop("residue_type")

        # ext_lig masking (when model uses ext_lig embeddings)
        mask_extlig = self.cfg_exp.training.get("mask_extlig_proportion", 0.0)
        smart_extlig_augment = self.cfg_exp.training.get("ext_lig_smart_unknown_augment", False)
        if "ext_lig" in batch and getattr(self.cfg_exp.model.nn, "ext_lig_emb_dim", None):
            if smart_extlig_augment:
                ext_lig_before = batch["ext_lig"]
                stats = self._get_ext_lig_augment_stats()
                batch["ext_lig"], augment_stats = mask_ext_lig_blocky(
                    ext_lig_before,
                    mask,
                    stats["frac_present_pool"],
                    stats["runlen_bucket_lo"],
                    stats["runlen_bucket_hi"],
                    stats["runlen_bucket_weight"],
                    mask_value=2,
                )
                self._log_ext_lig_augment_monitoring(ext_lig_before, batch["ext_lig"], mask, augment_stats)
            elif mask_extlig > 0:
                batch["ext_lig"] = mask_seq(batch["ext_lig"], mask, mask_extlig, mask_value=2)

        # Zero out sinusoidal positional embedding (for fine-tuning without position info)
        if self.cfg_exp.training.get("zero_sin_pos_emb", False):
            batch["_zero_idx_emb"] = True

        # Self-conditioning: initialize SC key to zeros, optionally run SC forward
        target_clean = c_1 if contact_map_mode else x_1
        use_sc = random.random() > 0.5 and self.cfg_exp.training.self_cond
        self._set_self_cond(batch, target_clean, contact_map_mode, use_sc)

        # Main prediction
        nn_out = self.predict_clean(batch)

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
                        zt_2 = self.discrete_diffusion.forward_sample(c_tokens, t, pair_mask=pair_mask)
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

            # DSSP discrete diffusion loss
            if dssp_enabled and dssp_tokens is not None and dssp_zt is not None:
                dssp_logits = nn_out.get("dssp_logits")
                if dssp_logits is None:
                    raise ValueError(
                        "DSSP diffusion requires dssp_logits in model output. "
                        "Ensure dssp_diffusion_mode=True in nn config."
                    )
                dssp_diff_loss = self.dssp_diffusion.diffusion_loss(
                    dssp_logits, dssp_tokens, dssp_zt, t, pair_mask=mask.float()
                )
                dssp_recon_loss = self.dssp_diffusion.recon_loss(mask.float())
                dssp_prior_loss = self.dssp_diffusion.latent_loss(
                    dssp_tokens.shape[0], dssp_tokens.device
                )
                dssp_loss_total = dssp_diff_loss + dssp_recon_loss + dssp_prior_loss
                dssp_loss_total = _sanitize_and_log_loss_vec(dssp_loss_total, "dssp_diffusion_loss")
                dssp_loss_weight = float(self.cfg_exp.model.get("dssp_diffusion", {}).get("loss_weight", 1.0))
                self.log(
                    f"{log_prefix}/dssp_diffusion_loss",
                    torch.mean(dssp_loss_total),
                    on_step=True, on_epoch=True, prog_bar=False, logger=True,
                    batch_size=mask.shape[0], sync_dist=True, add_dataloader_idx=False,
                )
                train_loss = train_loss + dssp_loss_weight * torch.mean(dssp_loss_total)

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
                # Phase D: noised inputs (what the model actually saw this step).
                # x_t/contact_map_t/dssp_t are set during the flow-matching/diffusion
                # interpolation block earlier in training_step.
                x_t_local = batch.get("x_t")
                contact_map_t = batch.get("contact_map_t")
                contact_map_noised_viz = None
                if contact_map_t is not None:
                    try:
                        contact_map_noised_viz = self._contact_map_to_viz(contact_map_t[0])
                    except Exception:
                        contact_map_noised_viz = None
                dssp_t_local = batch.get("dssp_t")
                dssp_target_local = batch.get("dssp_target")
                dssp_logits_local = nn_out.get("dssp_logits") if isinstance(nn_out, dict) else None
                # Phase C: cheap single-step TMscore (predicted-x1 vs ground-truth x_1, no full trajectory).
                residue_type_first = (
                    batch.get("residue_type_unmasked", batch.get("residue_type"))[0]
                    if batch.get("residue_type_unmasked", batch.get("residue_type")) is not None
                    else None
                )
                if x_1_pred is not None and x_1 is not None and residue_type_first is not None:
                    tm = self._compute_tmscore_via_usalign(
                        x_pred=x_1_pred[:1].detach(),
                        x_gt=x_1[:1].detach(),
                        residue_type=residue_type_first,
                        mask=mask[0],
                    )
                    if tm is not None:
                        self.log(
                            f"{log_prefix}/tmscore_single_step",
                            float(tm["tm_gt_norm"]),
                            on_step=not val_step,
                            on_epoch=True,
                            logger=True,
                            batch_size=1,
                            sync_dist=False,
                            add_dataloader_idx=False,
                        )
                        self.log(
                            f"{log_prefix}/rmsd_single_step",
                            float(tm["rmsd"]),
                            on_step=not val_step,
                            on_epoch=True,
                            logger=True,
                            batch_size=1,
                            sync_dist=False,
                            add_dataloader_idx=False,
                        )
                # Shared payload: both calls below log panels for the SAME training
                # example/step, so accumulate into one dict and send a single
                # wandb.log() call (see `_log_structure_visualization` docstring) --
                # otherwise each panel would land on a different wandb step, making
                # the per-panel "Step" sliders in the WandB UI disagree even though
                # they all depict the same example.
                qual_viz_payload: Dict = {}
                self._log_structure_visualization(
                    # Log only the first sample. Passing the full batch here makes
                    # `write_prot_to_pdb` treat batch dim as MODEL index, producing a
                    # multi-model PDB with mismatched aatype/mask (confusing in WandB).
                    x_1_pred=x_1_pred[:1] if x_1_pred is not None else None,
                    contact_map_pred=contact_map_pred,
                    mask=mask[0],
                    log_prefix=log_prefix,
                    pair_logits=nn_out.get("pair_logits")[0] if "pair_logits" in nn_out else None,
                    residue_type=residue_type_first,
                    use_template_inference=(
                        getattr(self.nn, "predict_coords", True) is False
                        and getattr(self, "contact_map_mode", False)
                        and self.cfg_exp.model.nn.get("predict_structure_from_distogram", False)
                    ),
                    # In contact_map_mode the structure is NOT diffused -- x_t is a zeros
                    # placeholder (see training_step), so a "noised structure" is meaningless;
                    # don't log it for contact+dssp runs.
                    x_noised=(
                        x_t_local[:1]
                        if isinstance(x_t_local, torch.Tensor)
                        and x_t_local.numel() > 0
                        and not getattr(self, "contact_map_mode", False)
                        else None
                    ),
                    contact_map_noised=contact_map_noised_viz,
                    dssp_noised=dssp_t_local[0] if isinstance(dssp_t_local, torch.Tensor) and dssp_t_local.numel() > 0 else None,
                    dssp_pred=dssp_logits_local[0] if isinstance(dssp_logits_local, torch.Tensor) else None,
                    dssp_gt=dssp_target_local[0] if isinstance(dssp_target_local, torch.Tensor) and dssp_target_local.numel() > 0 else None,
                    payload=qual_viz_payload,
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
                    residue_type=residue_type_first,
                    payload=qual_viz_payload,
                )
                if len(qual_viz_payload) > 0:
                    self.logger.experiment.log(qual_viz_payload)

        return train_loss
    
    def _coords_to_temp_pdb(
        self,
        coords: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[str]:
        """Materialize a [n, 3] / [n, 3, 3] / [n, 14|37, 3] coords tensor to a tempfile PDB.

        Returns the path on success; None if no atoms survived the mask. Caller
        must delete the file. Mirrors the atom-mask logic in
        ``_log_structure_visualization`` so per-step USalign calls and the wandb
        Molecule logged for the same step are derived from the same atoms.
        """
        if coords is None:
            return None
        coords = coords[:1] if coords.dim() in (3, 4) and coords.shape[0] > 1 else coords
        if coords.dim() == 4:
            atom_dim = coords.shape[-2]
        else:
            atom_dim = None
        atom37 = self.samples_to_atom37(coords, residue_type=residue_type).float().detach().cpu().numpy()
        aatype = residue_type.long().detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(np.float32)
        if coords.dim() == 3:
            atom37_mask = np.zeros((aatype.shape[0], 37), dtype=np.float32)
            atom37_mask[:, rc.atom_order["CA"]] = mask_np
        elif coords.dim() == 4 and atom_dim == 3:
            atom37_mask = np.zeros((aatype.shape[0], 37), dtype=np.float32)
            atom37_mask[:, [rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]]] = mask_np[:, None]
        else:
            atom37_mask = rc.RESTYPE_ATOM37_MASK[aatype] * mask_np[:, None]
        if atom37_mask.sum() < 1.0:
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
            path = tmp_pdb.name
        write_prot_to_pdb(
            atom37, path, aatype=aatype, atom37_mask=atom37_mask,
            overwrite=True, no_indexing=True,
        )
        return path

    def _compute_tmscore_via_usalign(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
        timeout_s: float = 60.0,
    ) -> Optional[Dict[str, float]]:
        """Run USalign on the first sample of (x_pred, x_gt). Returns metrics dict or None.

        We use the GT-normalized score (TM2) as the canonical "TMscore vs ground
        truth"; the dict also exposes TM1 and RMSD so downstream logging can
        choose. Falls through gracefully if USalign isn't on PATH or the call
        fails — we never let logging break training.
        """
        usalign_bin = shutil.which("USalign") or os.path.expanduser("~/.local/bin/USalign")
        if not os.path.exists(usalign_bin):
            if not getattr(self, "_usalign_missing_warned", False):
                logger.warning(
                    f"USalign not found (checked PATH and {usalign_bin}) -- "
                    f"validation_sampling/tmscore_median will never be logged, which crashes "
                    f"the best-tmscore ModelCheckpoint callback the first time it fires. "
                    f"Install USalign or set log.checkpoint_best_tmscore_median=False."
                )
                self._usalign_missing_warned = True
            return None
        if x_pred is None or x_gt is None or residue_type is None or mask is None:
            return None
        # USalign mode: 5 = TMscore-style (residue-index alignment, fixed order — canonical
        # for structure prediction). 0 = USalign default (TMalign, sequence-independent
        # optimal alignment). The validation_sampling.tmscore_mode config knob picks.
        val_cfg = self.cfg_exp.get("validation_sampling", {}) or {}
        tmscore_mode = int(val_cfg.get("tmscore_mode", 5))
        pred_pdb = None
        gt_pdb = None
        try:
            pred_pdb = self._coords_to_temp_pdb(x_pred, residue_type, mask)
            gt_pdb = self._coords_to_temp_pdb(x_gt, residue_type, mask)
            if pred_pdb is None or gt_pdb is None:
                return None
            # One-shot unit-correctness assertion the first time this fires:
            # PDBs must have coordinates in Å (typical extent 30-300 Å).
            # Mismatched units (nm leaking through) would produce ~3-30 Å span; flag it.
            if not getattr(self, "_tmscore_unit_logged", False):
                try:
                    with open(pred_pdb) as f_pdb:
                        xs = []
                        for ln in f_pdb:
                            if ln.startswith("ATOM"):
                                try:
                                    xs.append(float(ln[30:38]))
                                except ValueError:
                                    pass
                        if xs:
                            span = max(xs) - min(xs)
                            logger.info(
                                f"[tmscore_unit_check] pred PDB x-span={span:.1f} Å "
                                f"(expect ~30-300 Å for proteins; <10 Å suggests nm leakage)"
                            )
                            self._tmscore_unit_logged = True
                except Exception:
                    pass
            argv = [usalign_bin, pred_pdb, gt_pdb, "-outfmt", "2"]
            if tmscore_mode != 0:
                argv += ["-TMscore", str(tmscore_mode)]
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s,
            )
            if result.returncode != 0:
                return None
            metrics = parse_usalign_pair_outfmt2(result.stdout)
            # parse_usalign_pair_outfmt2 returns keys: tms (=TM1, normalized to
            # structure 1 i.e. predicted), tms2 (=TM2, normalized to structure 2
            # i.e. ground truth), rms (RMSD), gdt (always NaN with -outfmt 2).
            return {
                "tm_pred_norm": float(metrics.get("tms", 0.0)),
                "tm_gt_norm": float(metrics.get("tms2", 0.0)),
                "rmsd": float(metrics.get("rms", 0.0)),
            }
        except Exception as e:
            logger.warning(f"USalign TMscore failed: {e!r}")
            return None
        finally:
            for p in (pred_pdb, gt_pdb):
                if p is not None and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

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
        x_noised: torch.Tensor = None,
        contact_map_noised: torch.Tensor = None,
        dssp_noised: torch.Tensor = None,
        dssp_pred: torch.Tensor = None,
        dssp_gt: torch.Tensor = None,
        payload: Optional[Dict] = None,
    ):
        """
        Logs predicted structure (as temporary PDB) and contact map visualizations.
        Expects x_1_pred coordinates in nm (training pipeline convention); samples_to_atom37
        converts nm -> Å for PDB output.

        All panels for one call are accumulated into a single dict and sent via one
        wandb.log() call, so they share the same wandb step (wandb auto-increments its
        internal step on every separate .log() call, which previously caused the
        structure/dssp/contact_map panels of the same training example to be spread
        across several different wandb steps).

        Args:
            x_1_pred: Predicted coordinates in nm, shape [1, n, 3]
            contact_map_pred: Predicted contact map, shape [n, n] or None
            mask: Boolean mask, shape [n]
            residue_type: Residue type, shape [n]
            log_prefix: Prefix for log names ("train" or "validation_loss")
            key_suffix: Suffix appended to logged keys (e.g., "_gt")
            payload: If given, accumulate entries into this dict instead of logging
                immediately -- the caller is responsible for calling
                self.logger.experiment.log(payload) once all panels are added, so
                that panels from multiple calls (e.g. pred pass + "_gt" pass) that
                belong to the same training example land on the same wandb step.
        """
        if (
            self.logger is None
            or not hasattr(self.logger, "experiment")
            or not hasattr(self.logger.experiment, "log")
        ):
            return

        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        # Accumulate all panels here and issue a single wandb.log() call at the end
        # (unless the caller supplied its own payload dict to merge multiple calls
        # into one log() -- see `payload` doc above).
        own_payload = payload is None
        if payload is None:
            payload = {}

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
                payload[f"{log_prefix}/structure{key_suffix}"] = wandb.Molecule(temp_pdb_path)
                payload["global_step"] = self.global_step
                payload["epoch"] = self.current_epoch
                # Intentionally NOT deleted here -- see the temp-PDB-files note in __init__
                # (wandb consumes this file at an unpredictable later time; deleting it here
                # races that read).
            except Exception:
                if os.path.exists(temp_pdb_path):
                    os.remove(temp_pdb_path)
                raise

        # Optional noised structure (Phase D — show what the model actually saw).
        if x_noised is not None and residue_type is not None:
            noised_pdb = self._coords_to_temp_pdb(x_noised, residue_type, mask)
            if noised_pdb is not None:
                try:
                    payload[f"{log_prefix}/structure_noised{key_suffix}"] = wandb.Molecule(noised_pdb)
                    payload["global_step"] = self.global_step
                    payload["epoch"] = self.current_epoch
                    # Intentionally not deleted -- see note above.
                except Exception:
                    if os.path.exists(noised_pdb):
                        try:
                            os.remove(noised_pdb)
                        except OSError:
                            pass
                    raise

        mask_np = mask.detach().cpu().numpy()
        pair_mask_np = mask_np[..., :, None] * mask_np[..., None, :]

        def _log_contact_image(cmap_tensor: torch.Tensor, key: str, title: str):
            cmap_prob = cmap_tensor.float().clamp(0.0, 1.0).detach().cpu().numpy()
            # UDLM contact maps come as 2-class softmax (..., 2) - take the
            # positive (contact=1) class for the heatmap. CB-distance/precomputed
            # maps come as plain 2D - no reduction needed.
            if cmap_prob.ndim == 3 and cmap_prob.shape[-1] == 2:
                cmap_prob = cmap_prob[..., 1]
            cmap_np = cmap_prob * pair_mask_np
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(cmap_np, cmap="viridis", aspect="auto", vmin=0, vmax=1)
            ax.set_xlabel("Residue j")
            ax.set_ylabel("Residue i")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="Contact probability")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            payload[key] = wandb.Image(img)
            payload["global_step"] = self.global_step
            payload["epoch"] = self.current_epoch

        if contact_map_pred is not None:
            _log_contact_image(
                contact_map_pred,
                f"{log_prefix}/contact_map{key_suffix}",
                f"Contact Map - Step {self.global_step}",
            )

        if contact_map_noised is not None:
            _log_contact_image(
                contact_map_noised,
                f"{log_prefix}/contact_map_noised{key_suffix}",
                f"Contact Map (noised) - Step {self.global_step}",
            )

        # DSSP visualization (Phase D). Renders 3-state DSSP as a horizontal strip:
        # 0=loop (gray), 1=helix (red), 2=strand (gold). -1 (ignore) is rendered black.
        def _dssp_to_arr(d: torch.Tensor) -> Optional[np.ndarray]:
            if d is None:
                return None
            d = d.detach().cpu()
            if d.dim() >= 2 and d.shape[-1] in (3, 4):
                # logits / probabilities — argmax to class indices
                d = d.argmax(dim=-1)
            if d.dim() == 0 or d.numel() == 0:
                return None
            return d.numpy().reshape(-1)

        def _log_dssp_strip(arr: Optional[np.ndarray], key: str, title: str):
            if arr is None:
                return
            arr = arr.copy()
            mask_1d = mask_np if mask_np.ndim == 1 else mask_np.reshape(-1)
            if mask_1d.size == arr.size:
                # Select the actual valid positions, not the first mask.sum() raw
                # positions -- those differ whenever the mask has an INTERNAL gap
                # (missing residue / chain break), not just right-side padding,
                # which previously misaligned this strip against the structure and
                # contact-map panels (both of which correctly skip only the gaps).
                arr_show = arr[mask_1d.astype(bool)]
            else:
                arr_show = arr
            if arr_show.size <= 0:
                return
            fig, ax = plt.subplots(figsize=(8, 1.0))
            cmap = plt.matplotlib.colors.ListedColormap(["black", "lightgray", "firebrick", "gold"])
            # remap: -1 -> 0 (black), 0/1/2 -> 1/2/3
            im_arr = np.where(arr_show < 0, 0, arr_show + 1).astype(np.int64)[None, :]
            ax.imshow(im_arr, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            ax.set_yticks([])
            ax.set_xlabel("Residue")
            ax.set_title(title)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            payload[key] = wandb.Image(img)
            payload["global_step"] = self.global_step
            payload["epoch"] = self.current_epoch

        _log_dssp_strip(
            _dssp_to_arr(dssp_noised),
            f"{log_prefix}/dssp_noised{key_suffix}",
            f"DSSP (noised) - Step {self.global_step}",
        )
        _log_dssp_strip(
            _dssp_to_arr(dssp_pred),
            f"{log_prefix}/dssp_pred{key_suffix}",
            f"DSSP (predicted) - Step {self.global_step}",
        )
        _log_dssp_strip(
            _dssp_to_arr(dssp_gt),
            f"{log_prefix}/dssp_gt{key_suffix}",
            f"DSSP (ground truth) - Step {self.global_step}",
        )

        # Single wandb.log() call for all panels accumulated above -- keeps them
        # on the same wandb step. Skipped when the caller passed its own `payload`
        # (it will log once after merging in panels from other calls too).
        if own_payload and len(payload) > 0:
            self.logger.experiment.log(payload)

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

    def _diag_log(self, where: str, extra: str = "") -> None:
        """Rank-aware diagnostic log that does NOT cross DDP barriers.

        Use to pinpoint where ranks stall during DDP collectives. Every rank
        writes through loguru's plain handler (no sync_dist, no torchmetrics).
        Post-hoc, grep ``[diag rank=N]`` per N to see how far each rank got
        before the timeout — the first rank that stopped logging is the one
        holding the barrier.
        """
        rank = getattr(self, "global_rank", -1)
        step = getattr(self, "global_step", -1)
        logger.info(f"[diag rank={rank} step={step} t={time.time():.3f}] {where} {extra}")

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
        self._diag_log("validation_step enter", f"batch_idx={batch_idx}")
        self.validation_step_data(batch, batch_idx)
        self._diag_log("validation_step exit", f"batch_idx={batch_idx}")

    @staticmethod
    def _concat_val_batches(batches: List) -> dict:
        """Concatenate a list of PyG-Batch / dict batches along dim 0.

        Tensors are torch.cat'd; lists are extended; nested dicts (e.g.
        ``mask_dict``) are recursively concatenated by key; everything else
        falls back to the first batch's value. All input batches MUST share the
        same padded length N (true under PaddingTransform with fixed max_size).

        Returns a plain dict (good enough for both ``extract_clean_sample`` —
        which does ``batch["coords"]`` — and ``_run_validation_trajectory`` —
        which does ``batch.get(...)``).
        """
        if not batches:
            return {}

        def _cat_values(vals):
            if not vals:
                return None
            if all(isinstance(v, torch.Tensor) for v in vals):
                # All tensors must agree on every dim except 0. If they don't,
                # something's wrong (e.g. unequal padding) — keep the first.
                try:
                    return torch.cat(vals, dim=0)
                except RuntimeError:
                    return vals[0]
            if all(isinstance(v, dict) for v in vals):
                # Recurse on union of keys.
                union: Dict = {}
                for k in {kk for v in vals for kk in v.keys()}:
                    sub = [v[k] for v in vals if k in v]
                    union[k] = _cat_values(sub)
                return union
            if all(isinstance(v, (list, tuple)) for v in vals):
                merged: List = []
                for v in vals:
                    merged.extend(list(v))
                return merged
            return vals[0]

        out: Dict = {}
        keys = list(batches[0].keys())
        for k in keys:
            # PyG Batch internal counters that don't make sense to concat.
            if k in ("batch", "ptr"):
                continue
            vals = [b[k] for b in batches if k in b.keys()]
            if not vals:
                continue
            out[k] = _cat_values(vals)
        return out

    def validation_step_data(self, batch, batch_idx):
        """
        Evaluates the training loss, without auxiliary loss nor logging.
        This is done with the function `training_step` with batch_idx -1.

        After the per-batch loss, fire ``_run_validation_trajectory`` AT MOST ONCE
        per validation epoch (gated by ``_logged_val_traj_epoch``). Two paths:

        - ``tmscore_n_samples == 0`` (stage-1 default): qualitative-only. Rank-0
          generates one sample with fresh noise + variable-length sampling and
          logs structure / contact map / sampled DSSP to wandb. Non-rank-0
          ranks return immediately. No TM-score, no USalign.
        - ``tmscore_n_samples > 0`` (stage-2 default 8): TM-score path. Each
          rank generates ``tmscore_n_samples // world_size`` samples; per-rank
          USalign(sampled, GT) scores are gathered to rank-0 in
          ``on_validation_epoch_end_data``. Rank-0 also renders one sample for
          a qualitative wandb panel.

        ``batch`` is not used by ``_run_validation_trajectory`` itself (val
        sampling uses fresh noise) — it is still consumed by the training_step
        above for the validation loss. We pass it through to the trajectory
        helper so the GT DSSP / structure can be shown alongside the sample
        when available, but the sampling itself doesn't condition on it.
        """
        self._diag_log("val_step_data: pre training_step", f"batch_idx={batch_idx}")
        with torch.no_grad():
            self._in_validation_loop = True
            loss = self.training_step(batch, batch_idx=batch_idx)
            self._in_validation_loop = False
            self.validation_output_data.append(loss.item())
        self._diag_log("val_step_data: post training_step", f"batch_idx={batch_idx}")

        val_sampling_cfg = self.cfg_exp.get("validation_sampling", None)
        if not val_sampling_cfg or self.global_step <= 0:
            return
        if self._logged_val_traj_epoch == self.current_epoch:
            return  # already fired this epoch
        tmscore_n = int(val_sampling_cfg.get("tmscore_n_samples", 0))
        # Mark the epoch on ALL ranks BEFORE the call so subsequent val
        # batches across all ranks skip the trajectory consistently. Without
        # this, the previous code only set the flag on rank 0 (inside the
        # for-loop), so ranks 1-3 re-entered the buffer/chunk path on every
        # subsequent val batch.
        self._logged_val_traj_epoch = self.current_epoch
        if tmscore_n == 0:
            # Stage-1: validation inference trajectory DISABLED. Only the
            # per-batch val loss (computed above) runs; we skip the full
            # generate() that otherwise ran every val epoch with no value for
            # stage-1 (no seq conditioning -> GT comparison is meaningless).
            # _run_validation_trajectory's qualitative_only=True branch is now dormant.
            self._diag_log("val_step_data: trajectory disabled (tmscore_n=0)", "")
            return
        else:
            # TM-score path: distribute samples across ranks. With world=4 and
            # tmscore_n=8, each rank does 2 samples in one generate() call —
            # forwards are shared across the per-rank batch.
            world_size = max(1, int(getattr(self.trainer, "world_size", 1)))
            per_rank = max(1, tmscore_n // world_size)
            self._diag_log(
                "val_step_data: traj enter distributed",
                f"world={world_size} per_rank={per_rank} total={per_rank * world_size}",
            )
            self._run_validation_trajectory(
                batch,
                val_sampling_cfg,
                nsamples=per_rank,
                qualitative_only=False,
            )
            self._diag_log("val_step_data: traj done distributed", "")

    @staticmethod
    def _slice_concat_batch(batch: dict, start: int, end: int) -> dict:
        """Slice a concat'd val-traj batch dict along dim 0. Tensors are sliced;
        lists are sliced as Python lists; nested dicts are recursed; other types
        pass through unchanged (they don't carry a batch dim)."""
        def _slice(v):
            if isinstance(v, torch.Tensor):
                return v[start:end] if v.dim() >= 1 and v.shape[0] >= end else v
            if isinstance(v, list):
                return v[start:end] if len(v) >= end else v
            if isinstance(v, dict):
                return {kk: _slice(vv) for kk, vv in v.items()}
            return v
        return {k: _slice(v) for k, v in batch.items()}

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_start(self):
        # Reset the per-epoch TMscore accumulator. This runs on every rank, but
        # _run_validation_trajectory only appends on rank 0 (the only rank that
        # holds a logger), so the aggregation is rank-0 local.
        self._validation_tmscore_results: List[Dict[str, float]] = []
        # Per-epoch contact-map metrics accumulator (only populated in contact_map_mode
        # with discrete diffusion). Same per-sample-dict pattern as tmscore.
        self._validation_contact_results: List[Dict[str, float]] = []
        # Buffer for batched val-trajectory inference: the first
        # tmscore_n_samples val batches are accumulated and concatenated, then
        # one batched generate() call shares the trajectory's NN forward passes
        # across all of them.
        self._val_traj_buffer: List[dict] = []

    def on_validation_epoch_end_data(self):
        # Stage-2 distributed path accumulates per-rank TM-scores + contact
        # metrics in ``_validation_tmscore_results`` / ``_validation_contact_results``.
        # Gather every rank's lists to a single flat list before the rank-0
        # aggregation. NCCL doesn't have a native object collective so we use
        # the gloo subgroup Lightning creates automatically alongside the
        # NCCL default group; if that's missing we fall back to using only
        # the local list (rank-0 still has its share).
        local_tm = list(getattr(self, "_validation_tmscore_results", []) or [])
        local_cm = list(getattr(self, "_validation_contact_results", []) or [])
        results: List[Dict[str, float]] = local_tm
        contact_results: List[Dict[str, float]] = local_cm
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                tm_gathered: List = [None] * world_size
                cm_gathered: List = [None] * world_size
                dist.all_gather_object(tm_gathered, local_tm)
                dist.all_gather_object(cm_gathered, local_cm)
                results = [r for sub in tm_gathered for r in (sub or [])]
                contact_results = [r for sub in cm_gathered for r in (sub or [])]
                self._diag_log(
                    "val_end: gather",
                    f"world={world_size} tm_total={len(results)} cm_total={len(contact_results)}",
                )
        except (RuntimeError, ImportError) as e:
            logger.warning(
                f"on_validation_epoch_end_data: dist.all_gather_object failed ({e!r}); "
                f"falling back to local rank-only aggregation."
            )

        # Aggregate per-sample TMscores accumulated by _run_validation_trajectory.
        # `results` is already identical on every rank (all_gather_object above), so the
        # metric computation + self.log() below run on EVERY rank, not just rank 0 --
        # ModelCheckpoint's monitor check (e.g. checkpoint_best_tmscore_median) runs
        # per-rank against that rank's OWN callback_metrics, so if only rank 0 ever
        # logged this key (the previous rank_zero_only=True), every non-zero rank would
        # permanently lack it and crash with "could not find the monitored key" the moment such a
        # callback is enabled (confirmed root cause of a real production crash,
        # 2026-07-05/06 -- the earlier "gloo Connection closed" symptom on rank 0 was a
        # DOWNSTREAM consequence of the other ranks dying first on this exact exception,
        # not an independent network issue). Only the direct wandb dict-log (images/raw
        # payload, only meaningful on rank 0's real wandb experiment) stays rank-0-gated.
        if results:
            tm_gt = np.array([r["tm_gt_norm"] for r in results], dtype=np.float64)
            tm_pred = np.array([r["tm_pred_norm"] for r in results], dtype=np.float64)
            rmsd = np.array([r["rmsd"] for r in results], dtype=np.float64)
            payload = {
                "validation_sampling/tmscore_mean": float(tm_gt.mean()),
                "validation_sampling/tmscore_median": float(np.median(tm_gt)),
                "validation_sampling/tmscore_min": float(tm_gt.min()),
                "validation_sampling/tmscore_max": float(tm_gt.max()),
                "validation_sampling/tmscore_pred_norm_mean": float(tm_pred.mean()),
                "validation_sampling/rmsd_mean": float(rmsd.mean()),
                "validation_sampling/n_samples": int(len(results)),
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
            if (
                self.logger is not None
                and hasattr(self.logger, "experiment")
                and not (hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero)
            ):
                self.logger.experiment.log(payload)
            # Also publish through Lightning's metric system, on EVERY rank, so
            # ModelCheckpoint sees these values in callback_metrics regardless of which
            # rank it's checking. Skip "global_step"/"epoch" — Lightning attaches its
            # own versions of those keys to every logged metric.
            for k, v in payload.items():
                if k in ("global_step", "epoch"):
                    continue
                self.log(
                    k, float(v),
                    on_step=False, on_epoch=True,
                    prog_bar=False, logger=False,
                    rank_zero_only=False, sync_dist=False,
                    add_dataloader_idx=False,
                )

        # Aggregate per-sample contact-map metrics (only present in contact_map_mode
        # with discrete diffusion). Mean over the per-sample dicts (gathered
        # across ranks above). Each metric key is logged independently so a
        # sample missing one metric (e.g. no long-range pairs at all) doesn't
        # poison the rest. Same fix as the tmscore block above: contact_results is
        # already identical on every rank (all_gather_object earlier in this
        # function), so the metric computation + self.log() run on EVERY rank, not
        # just rank 0 -- otherwise a ModelCheckpoint monitor keyed on any of these
        # values would crash on every non-zero rank the same way the tmscore one did
        # (see that block's comment for the full story). Only the direct wandb
        # dict-log stays rank-0-gated (needs a real wandb experiment object).
        if contact_results:
            # Collect all metric keys across the samples (some are optional).
            all_keys = set()
            for r in contact_results:
                all_keys.update(r.keys())
            payload_c: Dict[str, float] = {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
            for k in sorted(all_keys):
                vals = [r[k] for r in contact_results if k in r]
                if not vals:
                    continue
                arr = np.array(vals, dtype=np.float64)
                payload_c[f"validation_sampling/{k}_mean"] = float(arr.mean())
                payload_c[f"validation_sampling/{k}_median"] = float(np.median(arr))
            payload_c["validation_sampling/contact_n_samples"] = int(len(contact_results))
            if (
                self.logger is not None
                and hasattr(self.logger, "experiment")
                and not (hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero)
            ):
                self.logger.experiment.log(payload_c)
            for k, v in payload_c.items():
                if k in ("global_step", "epoch"):
                    continue
                self.log(
                    k, float(v),
                    on_step=False, on_epoch=True,
                    prog_bar=False, logger=False,
                    rank_zero_only=False, sync_dist=False,
                    add_dataloader_idx=False,
                )
        self._validation_tmscore_results = []
        self._validation_contact_results = []
        self.validation_output_data = []

    def _resolve_validation_length_bounds(self, val_sampling_cfg) -> tuple:
        """Return (min_length, max_length) for validation trajectory sampling, or (None, None).

        Order: explicit ``validation_sampling.{min,max}_length`` → datamodule dataselector
        → optional ``cfg_exp.dataset_length_bounds``.
        """
        mn = val_sampling_cfg.get("min_length")
        mx = val_sampling_cfg.get("max_length")
        if mn is not None and mx is not None:
            return int(mn), int(mx)
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            ds = getattr(dm, "dataselector", None)
            if ds is not None:
                m = getattr(ds, "min_length", None)
                x = getattr(ds, "max_length", None)
                if m is not None and x is not None:
                    return int(m), int(x)
        db = self.cfg_exp.get("dataset_length_bounds")
        if db is not None and db.get("min_length") is not None and db.get("max_length") is not None:
            return int(db["min_length"]), int(db["max_length"])
        return None, None

    def _use_variable_length_validation_sampling(self, val_sampling_cfg, min_l, max_l) -> bool:
        """Whether to sample a random length in [min_l, max_l] for validation trajectories."""
        flag = val_sampling_cfg.get("variable_length_sampling", None)
        if flag is False:
            return False
        if flag is True:
            if min_l is None or max_l is None or max_l < min_l:
                logger.warning(
                    "variable_length_sampling=True but min/max length could not be resolved; "
                    "using full padded width."
                )
                return False
            return True
        # Default when flag is omitted: enable when bounds are available.
        return min_l is not None and max_l is not None and max_l >= min_l

    def _run_validation_trajectory(
        self,
        batch,
        val_sampling_cfg,
        nsamples: int = 1,
        qualitative_only: bool = True,
    ):
        """Run reverse-diffusion sampling and log/accumulate val metrics.

        Two paths (selected by caller via ``qualitative_only``):

        - ``qualitative_only=True`` (stage-1): only rank-0 runs ``generate()``,
          logs one sample's structure / contact map / DSSP to wandb. Non-rank-0
          ranks return immediately so we don't burn 3x compute on duplicate
          generations that nobody can render (only rank-0 has a wandb logger).
        - ``qualitative_only=False`` (stage-2): every rank runs ``generate()``
          on its share of ``nsamples`` samples and locally accumulates per-sample
          TM-scores + contact-map metrics into ``self._validation_tmscore_results``
          / ``self._validation_contact_results``. These per-rank lists are
          gathered to rank-0 in ``on_validation_epoch_end_data``. Rank-0 also
          renders one sample for a qualitative wandb panel.

        ``nsamples`` is the per-rank sample count (caller already divides
        total by world_size). Sliced from ``batch`` so the GT/residue_type
        used for TM-score scoring and CATH/motif conditioning is per-sample
        consistent.
        """
        is_rank0 = not (
            hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero
        )

        if qualitative_only and not is_rank0:
            # Stage-1: only rank-0 generates; others skip to save compute.
            # The caller-side per-rank sync barrier (the sync_dist=True
            # self.log calls inside the next validation_step_data invocation)
            # is what eventually re-aligns the ranks.
            self._diag_log("traj: skipped (qualitative non-rank-0)", "")
            return
        if is_rank0 and (self.logger is None or not hasattr(self.logger, "experiment")):
            # Rank-0 without a logger means we couldn't render anyway.
            self._diag_log("traj: skipped (no logger)", "")
            return

        self._diag_log("traj: enter", f"nsamples={nsamples} qualitative={qualitative_only}")
        _traj_t0 = time.time()
        dt = float(val_sampling_cfg.get("dt", 0.005))
        sampling_mode = val_sampling_cfg.get("sampling_mode", "sc")
        sc_scale_noise = float(val_sampling_cfg.get("sc_scale_noise", 0.45))
        sc_scale_score = float(val_sampling_cfg.get("sc_scale_score", 1.0))
        schedule_mode = val_sampling_cfg.get("schedule_mode", "uniform")
        schedule_p = float(val_sampling_cfg.get("schedule_p", 1.0))
        sampling_grid = val_sampling_cfg.get("sampling_grid", None)

        # Slice the val batch to ``nsamples`` per-sample tensors. The val
        # dataloader yields batches of size val_batch_size (= train batch_size,
        # 4 in the user's config); the caller's ``nsamples`` is at most this
        # (1 for qualitative; 2 = 8/world_size for stage-2 distributed). If
        # the batch is smaller than requested, we clamp to what's available
        # and log a warning.
        mask_all = batch["mask"].to(self.device)
        avail = int(mask_all.shape[0])
        if nsamples > avail:
            logger.warning(
                f"validation_sampling: requested nsamples={nsamples} > val "
                f"batch size {avail}; clamping to {avail}."
            )
            nsamples = avail
        mask = mask_all[:nsamples]
        n = mask.shape[-1]
        # Use unmasked sequence for validation sampling (training masks residue_type for seq_cond)
        residue_type = batch.get("residue_type_unmasked", batch.get("residue_type"))
        if residue_type is not None:
            residue_type = residue_type[:nsamples].to(self.device)
        log_visualization = is_rank0  # render only on rank-0 (only it has wandb)
        accumulate_metric = not qualitative_only  # stage-2 accumulates TM-score + contact
        # Prediction-eval semantics: at inference we don't condition on a known fold.
        # Build fresh all-unknown CATH from scratch instead of reading the buffered
        # cath_code_indices: with tmscore_n_samples>1 the val-traj buffer concatenates
        # multiple val batches along dim 0, but cath_code_indices has shape
        # [B, max_labels, 3] with per-batch-varying max_labels — so torch.cat fails on
        # the inner dim, _cat_values silently returns vals[0], and the model receives
        # B=batch_size_val (not B=nsamples) for the fold-embedding path. Constructing
        # [nsamples, 1, 3] all-null indices here sidesteps that entirely, and matches
        # the prediction-eval intent anyway. (TODO: also log a parallel set of val
        # metrics with real CATH conditioning; see project_proteina_val_metrics_todo.)
        cath_code = None
        cath_code_indices = None
        cath_code_indices_mask = None
        if self.cfg_exp.training.get("fold_cond", False):
            cath_code_dir = self.cfg_exp.model.nn.get("cath_code_dir")
            if cath_code_dir is not None:
                _, _, _, nC, nA, nT = load_cath_mapping(cath_code_dir)
                cath_code_indices = torch.zeros(
                    (nsamples, 1, 3), device=self.device, dtype=torch.long
                )
                cath_code_indices[:, 0, 0] = nC
                cath_code_indices[:, 0, 1] = nA
                cath_code_indices[:, 0, 2] = nT
                # cath_code_indices_mask is True for PADDED (invalid) positions; one
                # valid null label per sample → all-False mask.
                cath_code_indices_mask = torch.zeros(
                    (nsamples, 1), device=self.device, dtype=torch.bool
                )

        # Extract motif conditioning: support both (motif_structure, motif_seq_mask) from
        # inference/GenMotifDataset and (x_motif, fixed_sequence_mask, fixed_structure_mask)
        # from motif_factory during PDB training/validation.
        x_motif = None
        fixed_sequence_mask = None
        fixed_structure_mask = None
        if batch.get("motif_structure") is not None and batch.get("motif_seq_mask") is not None:
            x_motif = batch["motif_structure"][:nsamples].to(self.device)
            fixed_sequence_mask = batch["motif_seq_mask"][:nsamples].to(self.device)
            fixed_structure_mask = fixed_sequence_mask[:, :, None] * fixed_sequence_mask[:, None, :]
        elif batch.get("x_motif") is not None and batch.get("fixed_sequence_mask") is not None:
            x_motif = batch["x_motif"][:nsamples].to(self.device)
            fixed_sequence_mask = batch["fixed_sequence_mask"][:nsamples].to(self.device)
            fs_mask = batch.get("fixed_structure_mask")
            if fs_mask is not None:
                fixed_structure_mask = fs_mask[:nsamples].to(self.device)
            else:
                fixed_structure_mask = fixed_sequence_mask[:, :, None] * fixed_sequence_mask[:, None, :]

        # Variable-length validation trajectory: sample L ~ Uniform(min_length, max_length).
        # Inputs stay padded to max length (fixed shapes for compile); only ``mask`` is set so
        # positions [L:] are invalid (False), matching the sampled sequence length L.
        min_l, max_l = self._resolve_validation_length_bounds(val_sampling_cfg)
        if self._use_variable_length_validation_sampling(val_sampling_cfg, min_l, max_l):
            n_pad = n
            L = random.randint(min_l, max_l)
            L = min(L, n_pad)  # cannot exceed padded tensor width
            if L < min_l:
                logger.warning(
                    f"validation_sampling: padded width {n_pad} < min_length={min_l}; using L={L}."
                )
            mask = torch.zeros(nsamples, n_pad, dtype=torch.bool, device=self.device)
            mask[:, :L] = True
            # n, residue_type, cath_*, motif tensors unchanged — full padded width for static shapes
            logger.info(
                f"validation_sampling: variable length L={L} (bounds [{min_l}, {max_l}]), "
                f"mask valid prefix on padded width {n_pad}"
            )

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
                cath_code_indices=cath_code_indices,
                cath_code_indices_mask=cath_code_indices_mask,
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
                fixed_structure_mask=fixed_structure_mask,
                # Keep val trajectory's pos-emb behavior consistent with training:
                # if training zeros sinusoidal pos embs, val must too (and vice versa).
                zero_sin_pos_emb=bool(self.cfg_exp.training.get("zero_sin_pos_emb", False)),
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

        # Extract GT once (used for both visualization and per-sample TMscore).
        # `extract_clean_sample` returns the same `x_1` the training loop uses as
        # the prediction target, so logging it here gives the user a side-by-side
        # comparison of "what the model produced" vs "what it was trying to hit"
        # for the same residue_type / fold-conditioning / random-noise init.
        x_1_gt = None
        gt_mask = mask
        if coords is not None or log_visualization:
            try:
                x_1_gt, gt_mask, _, _, _ = self.extract_clean_sample(batch)
            except Exception as e:
                logger.warning(f"validation_sampling: failed to extract GT: {e!r}")
                x_1_gt = None
                gt_mask = mask

        # Log visualization for the FIRST sample only (consistent with the prior
        # single-sample behaviour). Use index 0 of the batched results — squeeze(0)
        # is wrong for nsamples > 1, so use [0] indexing instead.
        if log_visualization and coords is not None:
            cmap_first = None
            if contact_map is not None:
                _viz = self._contact_map_to_viz(contact_map)
                cmap_first = _viz[0] if _viz.dim() >= 2 else _viz
            # SAMPLED DSSP from generate() — this is what the diffusion model
            # produced (decoded long indices in {0=loop, 1=helix, 2=strand}).
            # Lets the user qualitatively check whether the joint
            # contact-map / structure / DSSP sample looks coherent (e.g. is
            # the helix annotation consistent with the dihedral profile of
            # the sampled CA trace). The visualization helper renders this
            # under `validation_sampling/dssp_pred`.
            dssp_sampled = result.get("dssp") if isinstance(result, dict) else None
            dssp_pred_first = (
                dssp_sampled[0]
                if isinstance(dssp_sampled, torch.Tensor) and dssp_sampled.numel() > 0
                else None
            )
            # GT DSSP from the batch (present whenever DSSPTargetTransform runs;
            # values in {0,1,2,-1}). Render alongside the sampled DSSP for
            # side-by-side comparison under `validation_sampling/dssp_gt`.
            dssp_target_raw = batch.get("dssp_target") if isinstance(batch, dict) else None
            dssp_gt_first = (
                dssp_target_raw[0]
                if isinstance(dssp_target_raw, torch.Tensor) and dssp_target_raw.numel() > 0
                else None
            )
            self._log_structure_visualization(
                x_1_pred=coords[:1],
                contact_map_pred=cmap_first,
                mask=mask[0],
                log_prefix="validation_sampling",
                pair_logits=distogram[0] if distogram is not None else None,
                residue_type=residue_type[0] if residue_type is not None else None,
                use_template_inference=predict_from_dist,
                dssp_pred=dssp_pred_first,
                dssp_gt=dssp_gt_first,
            )

            # GT visualization (target structure + GT contact map). Same sample
            # index 0, so the user can visually compare the prediction quality
            # against the actual target. DSSP is already covered above as
            # `dssp_gt`, so we don't pass it again here (which would log under
            # the awkward `dssp_gt_gt` key).
            gt_contact_map_viz = None
            if getattr(self, "contact_map_mode", False):
                try:
                    c_1_gt = self.extract_clean_contact_map(batch, mask)
                    if c_1_gt is not None and c_1_gt.numel() > 0:
                        gt_contact_map_viz = self._contact_map_to_viz(c_1_gt[0])
                except Exception as e:
                    logger.warning(
                        f"validation_sampling: failed to extract GT contact map: {e!r}"
                    )
                    gt_contact_map_viz = None
            if x_1_gt is not None:
                self._log_structure_visualization(
                    x_1_pred=x_1_gt[:1],
                    contact_map_pred=gt_contact_map_viz,
                    mask=mask[0],
                    log_prefix="validation_sampling",
                    key_suffix="_gt",
                    residue_type=residue_type[0] if residue_type is not None else None,
                )

        # Per-sample USalign TMscore loop. Even though we batched the trajectory's
        # NN forward passes, USalign itself runs per-pair on temp PDBs (it's a
        # subprocess, not a tensor op) — so we loop here.
        if accumulate_metric and coords is not None and residue_type is not None:
            if x_1_gt is not None:
                if not hasattr(self, "_validation_tmscore_results") or self._validation_tmscore_results is None:
                    self._validation_tmscore_results = []
                gt_mask_per = gt_mask if gt_mask.dim() > 1 else gt_mask.unsqueeze(0).expand(nsamples, -1)
                for s in range(nsamples):
                    if not bool(mask[s].any()):
                        continue
                    tm = self._compute_tmscore_via_usalign(
                        x_pred=coords[s:s + 1].detach(),
                        x_gt=x_1_gt[s:s + 1].detach(),
                        residue_type=residue_type[s] if residue_type.dim() > 1 else residue_type,
                        mask=gt_mask_per[s] if gt_mask_per.dim() > 1 else gt_mask_per,
                    )
                    if tm is not None:
                        self._validation_tmscore_results.append(tm)

        # Per-sample contact-map prediction-quality metrics. Only in contact_map_mode
        # with discrete diffusion (we need final-step logits from generate). All metrics
        # operate on the upper triangle with sequence separation >= 6 (the standard
        # contact-prediction convention — short-range trivial pairs are excluded).
        if (
            accumulate_metric
            and getattr(self, "contact_map_mode", False)
            and result is not None
            and result.get("contact_map_logits") is not None
        ):
            c_logits_pred = result["contact_map_logits"]  # [B, n, n]
            try:
                c_1_gt_full = self.extract_clean_contact_map(batch, mask)
            except Exception as e:
                logger.warning(
                    f"validation_sampling: failed to extract GT contact map for metrics: {e!r}"
                )
                c_1_gt_full = None
            if c_1_gt_full is not None:
                if (
                    not hasattr(self, "_validation_contact_results")
                    or self._validation_contact_results is None
                ):
                    self._validation_contact_results = []
                for s in range(nsamples):
                    if not bool(mask[s].any()):
                        continue
                    metrics_s = self._compute_contact_map_metrics(
                        logits=c_logits_pred[s],
                        gt=c_1_gt_full[s],
                        mask_1d=mask[s],
                    )
                    if metrics_s is not None:
                        self._validation_contact_results.append(metrics_s)
        self._diag_log(
            "traj: exit",
            f"elapsed={time.time() - _traj_t0:.2f}s nsamples={nsamples}",
        )

    @staticmethod
    def _compute_contact_map_metrics(
        logits: torch.Tensor,
        gt: torch.Tensor,
        mask_1d: torch.Tensor,
        min_sep: int = 6,
        long_range_sep: int = 24,
        medium_range_sep: int = 12,
    ) -> Optional[Dict[str, float]]:
        """Per-sample contact-map prediction-quality metrics.

        Args:
            logits: [n, n] real-valued (sigmoid → contact probability) — binary
                contact model (vocab_size=2 in MD4/UDLM).
            gt: [n, n] in {0, 1} (contact / no-contact) — masked outside valid pairs.
            mask_1d: [n] bool — valid residues.
            min_sep: minimum sequence separation for any pair to count (default 6 —
                the standard cutoff in contact-prediction literature; short-range
                pairs are trivial).
            long_range_sep: |i-j| >= 24 → long-range (CASP standard).
            medium_range_sep: 12 <= |i-j| < 24 → medium-range.

        Returns a dict of scalar metrics, or None if no valid pairs exist:
            contact_bce: mean BCE over valid upper-triangle pairs (sep >= min_sep)
            contact_accuracy: pixel-wise accuracy of (sigmoid(logits) >= 0.5) vs gt
            contact_recall, contact_precision, contact_f1: at threshold 0.5
            contact_precision_at_L,  _at_L2,  _at_L5: top-L, L/2, L/5 precision
                (k = valid_length / divisor, ranked by sigmoid(logits)) — pairs in
                the upper triangle with sep >= min_sep
            contact_long_range_precision_at_L5: same but pairs with sep >= long_range_sep
            contact_medium_range_precision_at_L5: medium-range only
        """
        logits = logits.float().detach()
        gt = gt.float().detach()
        n = int(logits.shape[-1])
        mask_1d = mask_1d.bool().detach()
        L_real = int(mask_1d.sum().item())
        if L_real < min_sep + 1:
            return None
        # Pair mask: both residues valid AND in upper triangle AND sep >= min_sep
        idx = torch.arange(n, device=logits.device)
        sep = (idx[None, :] - idx[:, None]).abs()
        pair_valid = mask_1d[:, None] & mask_1d[None, :]
        upper = idx[None, :] > idx[:, None]
        eval_mask = pair_valid & upper & (sep >= min_sep)
        long_mask = eval_mask & (sep >= long_range_sep)
        med_mask = eval_mask & (sep >= medium_range_sep) & (sep < long_range_sep)
        n_pairs = int(eval_mask.sum().item())
        if n_pairs <= 0:
            return None
        probs = torch.sigmoid(logits)
        # --- BCE over all valid upper-triangle pairs (sep >= min_sep) ---
        bce_full = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, gt, reduction="none"
        )
        bce_mean = (bce_full * eval_mask.float()).sum() / max(n_pairs, 1)
        # --- Pixel-wise accuracy at threshold 0.5 ---
        pred_bin = (probs >= 0.5).float()
        correct = (pred_bin == gt).float() * eval_mask.float()
        accuracy = correct.sum() / max(n_pairs, 1)
        # --- Threshold-0.5 precision/recall/F1 over eval pairs ---
        tp = (pred_bin * gt * eval_mask.float()).sum()
        fp = (pred_bin * (1 - gt) * eval_mask.float()).sum()
        fn = ((1 - pred_bin) * gt * eval_mask.float()).sum()
        precision = (tp / (tp + fp).clamp_min(1.0)).item() if (tp + fp).item() > 0 else 0.0
        recall = (tp / (tp + fn).clamp_min(1.0)).item() if (tp + fn).item() > 0 else 0.0
        f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) > 0 else 0.0
        # --- Top-K precision: rank pairs by prob, take top-K, count true contacts ---
        def _topk_precision(mask_for_k: torch.Tensor, k: int) -> Optional[float]:
            if k <= 0:
                return None
            flat_probs = probs[mask_for_k]
            flat_gt = gt[mask_for_k]
            if flat_probs.numel() == 0:
                return None
            topk = min(k, flat_probs.numel())
            top_idx = torch.topk(flat_probs, topk).indices
            return float(flat_gt[top_idx].mean().item())

        prec_L = _topk_precision(eval_mask, L_real)
        prec_L2 = _topk_precision(eval_mask, max(L_real // 2, 1))
        prec_L5 = _topk_precision(eval_mask, max(L_real // 5, 1))
        long_prec_L5 = _topk_precision(long_mask, max(L_real // 5, 1))
        med_prec_L5 = _topk_precision(med_mask, max(L_real // 5, 1))
        out = {
            "contact_bce": float(bce_mean.item()),
            "contact_accuracy": float(accuracy.item()),
            "contact_precision": float(precision),
            "contact_recall": float(recall),
            "contact_f1": float(f1),
            "n_pairs": int(n_pairs),
            "L_real": int(L_real),
        }
        if prec_L is not None:
            out["contact_precision_at_L"] = prec_L
        if prec_L2 is not None:
            out["contact_precision_at_L2"] = prec_L2
        if prec_L5 is not None:
            out["contact_precision_at_L5"] = prec_L5
        if long_prec_L5 is not None:
            out["contact_long_range_precision_at_L5"] = long_prec_L5
        if med_prec_L5 is not None:
            out["contact_medium_range_precision_at_L5"] = med_prec_L5
        return out

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag
        self._inf_zero_sin_pos_emb = bool(inf_cfg.get("zero_sin_pos_emb", False))
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
        # Also propagate sampling_grid to DSSP diffusion if present
        if self.dssp_diffusion is not None:
            dssp_sampling_grid = inf_cfg.get("dssp_sampling_grid", inf_cfg.get("sampling_grid", None))
            if dssp_sampling_grid is not None:
                if dssp_sampling_grid not in ("uniform", "cosine"):
                    raise ValueError(
                        "dssp_sampling_grid must be 'uniform' or 'cosine' "
                        f"(got {dssp_sampling_grid!r})."
                    )
                self.dssp_diffusion.sampling_grid = dssp_sampling_grid

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
        """
        force_compile = getattr(self, "_force_compile", False)
        verbose = getattr(self, "_verbose", False)
        sampling_args = self.inf_cfg.sampling_caflow

        cath_code_raw = (
            _extract_cath_code(batch) if self.inf_cfg.get("fold_cond", False) else None
        )
        cath_code_indices = batch.get("cath_code_indices")
        cath_code_indices_mask = batch.get("cath_code_indices_mask")
        if cath_code_indices is None and cath_code_raw is not None:
            cath_code_dir = self.cfg_exp.model.nn.get("cath_code_dir")
            multilabel_mode = self.cfg_exp.model.nn.get("multilabel_mode", "sample")
            if cath_code_dir:
                from proteinfoundation.datasets.cath_utils import cath_code_strings_to_indices_for_model
                cath_code_indices, cath_code_indices_mask = cath_code_strings_to_indices_for_model(
                    cath_code_raw, cath_code_dir, multilabel_mode, device=self.device
                )
        if cath_code_indices is None:
            cath_code_raw = cath_code_raw or [["x.x.x.x"] for _ in range(batch["nsamples"])]
        residue_type = batch["residue_type"] if self.inf_cfg.get("seq_cond", False) else None
        self._gen_residue_pdb_idx = batch.get("residue_pdb_idx")
        self._gen_chain_breaks = batch.get("chain_breaks_per_residue")
        guidance_weight = self.inf_cfg.get("guidance_weight", 1.0)
        autoguidance_ratio = self.inf_cfg.get("autoguidance_ratio", 0.0)
        
        mask = batch['mask'].squeeze(0) if 'mask' in batch else None
        # Extract motif: (motif_structure, motif_seq_mask) from GenMotifDataset or
        # (x_motif, fixed_sequence_mask, fixed_structure_mask) from motif_factory.
        if batch.get('motif_structure') is not None and batch.get('motif_seq_mask') is not None:
            x_motif = batch['motif_structure'].squeeze(0).to(self.device)
            fixed_sequence_mask = batch['motif_seq_mask'].squeeze(0).to(self.device)
            fixed_structure_mask = fixed_sequence_mask[:, :, None] * fixed_sequence_mask[:, None, :]
        elif batch.get('x_motif') is not None and batch.get('fixed_sequence_mask') is not None:
            x_motif = batch['x_motif'].squeeze(0).to(self.device)
            fixed_sequence_mask = batch['fixed_sequence_mask'].squeeze(0).to(self.device)
            fs_mask = batch.get('fixed_structure_mask')
            fixed_structure_mask = (
                fs_mask.squeeze(0).to(self.device) if fs_mask is not None
                else fixed_sequence_mask[:, :, None] * fixed_sequence_mask[:, None, :]
            )
        else:
            x_motif = fixed_sequence_mask = fixed_structure_mask = None

        save_trajectory = bool(self.inf_cfg.get("save_trajectory", False))
        save_trajectory_gif = bool(self.inf_cfg.get("save_trajectory_gif", False))
        return_trajectory = save_trajectory or save_trajectory_gif
        trajectory_stride = int(self.inf_cfg.get("trajectory_stride", 1))
        result = self.generate(
            nsamples=batch["nsamples"],
            n=batch["nres"],
            dt=batch["dt"].to(dtype=torch.float32),
            self_cond=self.inf_cfg.self_cond,
            cath_code=cath_code_raw if cath_code_indices is None else None,
            cath_code_indices=cath_code_indices,
            cath_code_indices_mask=cath_code_indices_mask,
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
            verbose=verbose,
        )
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
            "cath_code": cath_code_raw,
            "trajectory_tokens": result.get("trajectory_tokens"),
        }

    def generate(
        self,
        nsamples: int,
        n: int,
        dt: float,
        self_cond: bool,
        cath_code: Optional[List[List[str]]] = None,
        cath_code_indices: Optional[torch.Tensor] = None,
        cath_code_indices_mask: Optional[torch.Tensor] = None,
        residue_type: Optional[List[List[int]]] = None,
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
        verbose: bool = False,
        zero_sin_pos_emb: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Generates samples by integrating ODE with learned vector field.

        Args:
            zero_sin_pos_emb: when True, set ``nn_in["_zero_idx_emb"] = True`` so
                ``SeqPosEmbFeature`` zeros out the sinusoidal positional embedding
                at every trajectory step. Must match the training-time setting
                (``training.zero_sin_pos_emb``) to keep train/val/inference
                forward semantics consistent.

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
        dssp_diff_enabled = self._dssp_diffusion_enabled()
        if discrete_enabled or dssp_diff_enabled:
            if discrete_enabled and self.discrete_diffusion_type not in ("md4", "udlm"):
                raise ValueError("GenMD4 sampling is not implemented for inference.")
            # Use contact map timesteps as primary; fall back to DSSP if no contact diffusion
            if discrete_enabled:
                timesteps = self.discrete_diffusion.timesteps
            else:
                timesteps = self.dssp_diffusion.timesteps
            pair_mask = self._pair_mask(mask) if discrete_enabled else None
            # Prior samples
            zt = None
            dssp_zt = None
            if discrete_enabled:
                zt = self.discrete_diffusion.prior_sample(
                    (nsamples, n, n), device=self.device
                )
            if dssp_diff_enabled:
                dssp_zt = self.dssp_diffusion.prior_sample(
                    (nsamples, n), device=self.device
                )
            contact_map_sc = None
            dssp_sc = None
            distogram = None
            coords = None
            c_logits = None
            dssp_logits = None
            stride = max(1, int(trajectory_stride))
            trajectory_tokens = [] if return_trajectory else None
            for i in range(timesteps):
                # Use the appropriate diffusion module for grid
                if discrete_enabled:
                    s, t = self.discrete_diffusion.get_sampling_grid(i, timesteps)
                else:
                    s, t = self.dssp_diffusion.get_sampling_grid(i, timesteps)
                t_tensor = torch.full((nsamples,), t, device=self.device, dtype=torch.float32)

                nn_in = {
                    "x_t": torch.zeros(
                        nsamples, n, 3, device=self.device, dtype=dtype or torch.float32
                    ),
                    "t": t_tensor,
                    "mask": mask,
                }
                if zero_sin_pos_emb:
                    nn_in["_zero_idx_emb"] = True

                # Contact map input
                if discrete_enabled:
                    contact_map_t = self._contact_tokens_to_input(zt)
                    contact_map_t = self._apply_pair_mask(contact_map_t, pair_mask)
                    nn_in["contact_map_t"] = contact_map_t

                # DSSP input
                if dssp_diff_enabled:
                    dssp_t_input = self._dssp_tokens_to_input(dssp_zt)
                    dssp_t_input = dssp_t_input * mask[..., None].float()
                    nn_in["dssp_t"] = dssp_t_input

                if cath_code_indices is not None:
                    nn_in["cath_code_indices"] = cath_code_indices
                    if cath_code_indices_mask is not None:
                        nn_in["cath_code_indices_mask"] = cath_code_indices_mask
                elif cath_code is not None:
                    nn_in["cath_code"] = cath_code
                if residue_type is not None:
                    nn_in["residue_type"] = residue_type
                if fixed_sequence_mask is not None:
                    nn_in["fixed_sequence_mask"] = fixed_sequence_mask
                if fixed_structure_mask is not None:
                    nn_in["fixed_structure_mask"] = fixed_structure_mask
                if x_motif is not None:
                    nn_in["x_motif"] = x_motif

                # Self-conditioning (shared flag for both tracks)
                if self_cond:
                    if discrete_enabled:
                        if contact_map_sc is None:
                            contact_map_sc = torch.zeros(
                                nsamples, n, n,
                                device=self.device,
                                dtype=(dtype or torch.float32),
                            )
                        nn_in["contact_map_sc"] = contact_map_sc
                    if dssp_diff_enabled:
                        dssp_vocab = self.dssp_diffusion.vocab_size
                        if dssp_sc is None:
                            dssp_sc = torch.zeros(
                                nsamples, n, dssp_vocab,
                                device=self.device,
                                dtype=(dtype or torch.float32),
                            )
                        nn_in["dssp_sc"] = dssp_sc

                result = predict_clean_n_v_w_guidance(nn_in)

                # Contact map sampling step
                if discrete_enabled:
                    c_logits = result.get("contact_map_logits")
                    if c_logits is None:
                        raise ValueError(
                            "discrete diffusion sampling requires contact_map_logits in model output."
                        )
                    zt = self.discrete_diffusion.sample_step(zt, c_logits, s, t, pair_mask=pair_mask)

                # DSSP sampling step
                if dssp_diff_enabled:
                    dssp_logits = result.get("dssp_logits")
                    if dssp_logits is not None:
                        dssp_zt = self.dssp_diffusion.sample_step(dssp_zt, dssp_logits, s, t, pair_mask=mask)

                if return_trajectory and (i % stride == 0) and zt is not None:
                    trajectory_tokens.append(zt.detach().cpu())
                distogram = result.get("distogram")
                coords = result.get("coords")

                # Update self-conditioning
                if self_cond:
                    if discrete_enabled:
                        c_pred = result.get("contact_map")
                        if c_pred is not None:
                            contact_map_sc = c_pred.detach()
                    if dssp_diff_enabled and dssp_logits is not None:
                        dssp_sc = torch.softmax(dssp_logits, dim=-1).detach()

            # Decode final results
            gen_result = {"coords": coords, "distogram": distogram}
            if discrete_enabled and c_logits is not None:
                z0 = self.discrete_diffusion.decode(zt, c_logits)
                contact_map = self._contact_tokens_to_output(z0)
                contact_map = self._apply_pair_mask(contact_map, pair_mask)
                gen_result["contact_map"] = contact_map
                # Expose the final-step logits so the val trajectory can compute
                # contact-map prediction-quality metrics (BCE, top-L precision, etc.)
                # against the GT contact map. Detach: we never need gradients here
                # (val runs under no_grad in the trainer anyway).
                gen_result["contact_map_logits"] = c_logits.detach()
            if dssp_diff_enabled and dssp_logits is not None:
                dssp_z0 = self.dssp_diffusion.decode(dssp_zt, dssp_logits)
                gen_result["dssp"] = self._dssp_tokens_to_output(dssp_z0)
            if return_trajectory:
                if trajectory_tokens:
                    gen_result["trajectory_tokens"] = torch.stack(trajectory_tokens, dim=0)
                else:
                    shape = (0, nsamples, n, n) if discrete_enabled else (0, nsamples, n)
                    gen_result["trajectory_tokens"] = torch.empty(shape, dtype=torch.long)
            return gen_result
        modality = "contact_map" if contact_map_mode else "coordinates"
        predict_coords = getattr(self.nn, "predict_coords", True)

        return self.fm.full_simulation(
            predict_clean_n_v_w_guidance,
            dt=dt,
            nsamples=nsamples,
            n=n,
            self_cond=self_cond,
            cath_code=cath_code,
            cath_code_indices=cath_code_indices,
            cath_code_indices_mask=cath_code_indices_mask,
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
            verbose=verbose,
            zero_sin_pos_emb=zero_sin_pos_emb,
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
