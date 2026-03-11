import os
import logging
import random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from openfold.model.template import TemplatePairStack, TemplatePointwiseAttention
from openfold.model.embedders import TemplatePairEmbedder
from openfold.model.structure_module import StructureModule
import openfold.np.residue_constants as rc
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.data.data_pipeline import (
    make_dummy_msa_feats,
    make_sequence_features_with_custom_template,
    make_sequence_features,
)
from openfold.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


class OpenFoldTemplateInference(nn.Module):
    """
    Full OpenFold AlphaFold model inference using a distogram-only template.

    This is intended for Proteina contact-map diffusion logging/inference:
    given a predicted distogram probability tensor [B, L, L, 39] and the
    target sequence, produce atom37 coordinates.

    Supports:
      - torch.compile on compute-intensive submodules (evoformer, structure module)
        via enable_compilation() / disable_compilation()
      - Batched inference via forward_batched() for multiple samples
    """

    def __init__(
        self,
        *,
        model_name: str = "model_1_ptm",
        jax_params_path: str = None,
        device: Optional[torch.device] = None,
        rm_template_sequence: Optional[bool] = False,
        skip_template_alignment: bool = False,
        max_recycling_iters: Optional[int] = None,
        compile_model: bool = False,
        use_mlm: bool = False,
        use_deepspeed_evoformer_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        if jax_params_path is None:
            jax_params_path = os.path.join(
                os.path.expanduser("~/openfold/openfold/resources/params"),
                f"params_{model_name}.npz",
            )
        self.jax_params_path = jax_params_path
        self.cfg = model_config(
            model_name,
            use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )
        self.cfg.data.common.use_templates = True
        if max_recycling_iters is not None:
            self.cfg.data.common.max_recycling_iters = int(max_recycling_iters)

        # When use_mlm=False, disable the masked-language-model style random
        # corruption of MSA positions (masked_msa_replace_fraction).  This is
        # the main source of non-determinism in the feature pipeline and is
        # unnecessary for structure prediction from distograms.
        if not use_mlm:
            self.cfg.data.predict.masked_msa_replace_fraction = 0.0
            self.cfg.data.eval.masked_msa_replace_fraction = 0.0
        self.use_mlm = use_mlm

        self.model = AlphaFold(self.cfg)
        self.model.eval()
        import_jax_weights_(self.model, jax_params_path, version=model_name.replace("model_", ""))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise ValueError(
                "OpenFoldTemplateInference requires CUDA (OpenFold attention CUDA kernels are enabled). "
                f"Got device={device}."
            )
        self.device = device
        self.model = self.model.to(self.device)
        self.feature_pipeline = FeaturePipeline(self.cfg.data)
        self.rm_template_sequence = rm_template_sequence
        self.skip_template_alignment = skip_template_alignment
        self._template_use_unit_vector_default = None
        if hasattr(self.model, "template_embedder") and hasattr(self.model.template_embedder, "config"):
            self._template_use_unit_vector_default = self.model.template_embedder.config.use_unit_vector

        # Compilation state
        self._compiled = False
        self._original_forwards = {}

        if compile_model:
            self.enable_compilation()

    # ------------------------------------------------------------------
    # torch.compile support
    # ------------------------------------------------------------------

    @property
    def compiled(self) -> bool:
        return self._compiled

    def enable_compilation(self, dynamic: bool = False) -> None:
        """Compile compute-intensive submodules with torch.compile.

        Targets the evoformer blocks, extra-MSA blocks, template pair stack,
        and structure-module components (IPA, transition, angle resnet).
        The outer recycling loop and auxiliary heads are left uncompiled to
        avoid graph-break issues with .item(), .cpu(), and custom CUDA kernels.

        Args:
            dynamic: If True, use dynamic shapes to avoid recompilation when
                sequence lengths change between calls.  Slightly slower per
                call but avoids repeated compilation overhead.
        """
        if self._compiled:
            return

        compile_kwargs = dict(fullgraph=False, dynamic=dynamic)

        # --- Evoformer blocks (48 blocks – main compute bottleneck) ---
        for i, block in enumerate(self.model.evoformer.blocks):
            key = f"evoformer_block_{i}"
            self._original_forwards[key] = block.forward
            block.forward = torch.compile(block.forward, **compile_kwargs)

        # --- Extra-MSA stack blocks ---
        if hasattr(self.model, "extra_msa_stack"):
            for i, block in enumerate(self.model.extra_msa_stack.blocks):
                key = f"extra_msa_block_{i}"
                self._original_forwards[key] = block.forward
                block.forward = torch.compile(block.forward, **compile_kwargs)

        # --- Template pair stack blocks ---
        if (
            hasattr(self.model, "template_embedder")
            and hasattr(self.model.template_embedder, "template_pair_stack")
        ):
            tps = self.model.template_embedder.template_pair_stack
            for i, block in enumerate(tps.blocks):
                key = f"tps_block_{i}"
                self._original_forwards[key] = block.forward
                block.forward = torch.compile(block.forward, **compile_kwargs)

        # --- Structure module components ---
        # NOTE: The IPA is excluded because it uses attn_core_inplace_cuda, a
        # custom C++/CUDA extension that torch.compile cannot trace.  Graph
        # breaks around the kernel cause unacceptable numerical divergence.
        sm = self.model.structure_module
        for name in ("transition", "angle_resnet"):
            submod = getattr(sm, name, None)
            if submod is not None:
                key = f"sm_{name}"
                self._original_forwards[key] = submod.forward
                submod.forward = torch.compile(submod.forward, **compile_kwargs)

        self._compiled = True
        logger.info("OpenFold submodule compilation enabled (dynamic=%s)", dynamic)

    def disable_compilation(self) -> None:
        """Restore original (uncompiled) forward methods."""
        if not self._compiled:
            return

        for i, block in enumerate(self.model.evoformer.blocks):
            key = f"evoformer_block_{i}"
            if key in self._original_forwards:
                block.forward = self._original_forwards[key]

        if hasattr(self.model, "extra_msa_stack"):
            for i, block in enumerate(self.model.extra_msa_stack.blocks):
                key = f"extra_msa_block_{i}"
                if key in self._original_forwards:
                    block.forward = self._original_forwards[key]

        if (
            hasattr(self.model, "template_embedder")
            and hasattr(self.model.template_embedder, "template_pair_stack")
        ):
            tps = self.model.template_embedder.template_pair_stack
            for i, block in enumerate(tps.blocks):
                key = f"tps_block_{i}"
                if key in self._original_forwards:
                    block.forward = self._original_forwards[key]

        sm = self.model.structure_module
        for name in ("transition", "angle_resnet"):
            key = f"sm_{name}"
            if key in self._original_forwards:
                getattr(sm, name).forward = self._original_forwards[key]

        self._original_forwards.clear()
        self._compiled = False
        logger.info("OpenFold submodule compilation disabled")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _restype_idx_to_str(restype_idx: torch.Tensor) -> str:
        seq = []
        for x in restype_idx.detach().cpu().tolist():
            if x < 0:
                x = rc.restype_num
            if x > rc.restype_num:
                x = rc.restype_num
            seq.append(rc.restypes_with_x[x])
        return "".join(seq)

    def _set_template_use_unit_vector(self, zero_template_unit_vector: bool) -> None:
        if self._template_use_unit_vector_default is None:
            return
        if zero_template_unit_vector:
            self.model.template_embedder.config.use_unit_vector = False
        else:
            self.model.template_embedder.config.use_unit_vector = self._template_use_unit_vector_default

    @staticmethod
    def _apply_template_overrides(
        batch: dict,
        *,
        mask_template_aatype: bool,
        zero_template_torsion_angles: bool,
    ) -> None:
        if mask_template_aatype and "template_aatype" in batch:
            batch["template_aatype"] = torch.full_like(batch["template_aatype"], rc.restype_num)
        if zero_template_torsion_angles:
            for key in ("template_torsion_angles_sin_cos", "template_alt_torsion_angles_sin_cos"):
                if key in batch:
                    batch[key] = torch.zeros_like(batch[key])

    @staticmethod
    def _make_template_stub_features(
        sequence: str,
        mask: np.ndarray,
        rm_template_sequence: bool,
    ) -> dict:
        num_res = len(sequence)
        sequence_features = make_sequence_features(
            sequence=sequence,
            description="distogram_template",
            num_res=num_res,
        )
        msa_features = make_dummy_msa_feats(sequence)
        template_seq = ("X" * num_res) if rm_template_sequence else sequence
        template_aatype_one_hot = rc.sequence_to_onehot(
            sequence=template_seq,
            mapping=rc.restype_order_with_x,
            map_unknown_to_x=True,
        ).astype(np.float32)
        template_aatype = np.zeros(
            (1, num_res, len(rc.restypes_with_x_and_gap)), dtype=np.float32
        )
        template_aatype[..., : template_aatype_one_hot.shape[-1]] = template_aatype_one_hot[None, ...]
        mask_f = mask.astype(np.float32)
        template_features = {
            "template_aatype": template_aatype,
            "template_all_atom_mask": np.ones(
                (1, num_res, rc.atom_type_num), dtype=np.float32
            ),
            "template_all_atom_positions": np.zeros(
                (1, num_res, rc.atom_type_num, 3), dtype=np.float32
            ),
            "template_pseudo_beta_mask": mask_f[None, ...],
            "template_pseudo_beta": np.zeros((1, num_res, 3), dtype=np.float32),
            "template_domain_names": np.array([b"distogram_template"], dtype=object),
            "template_sequence": np.array([template_seq.encode("utf-8")], dtype=object),
            "template_sum_probs": np.array([[1.0]], dtype=np.float32),
        }
        return {**sequence_features, **msa_features, **template_features}

    @staticmethod
    def _inject_template_dgram_probs(
        batch: dict,
        distogram_probs: torch.Tensor,
    ) -> None:
        if distogram_probs is None or "template_aatype" not in batch:
            return
        dgram = distogram_probs
        if dgram.dim() == 4:
            dgram = dgram[0]
        if dgram.dim() != 3:
            raise ValueError(f"Expected distogram_probs [L, L, B], got shape {tuple(dgram.shape)}")
        num_res = batch["template_aatype"].shape[1]
        if dgram.shape[0] != num_res or dgram.shape[1] != num_res:
            raise ValueError(
                f"Distogram size {tuple(dgram.shape)} does not match template length {num_res}"
            )
        num_templates = batch["template_aatype"].shape[0]
        dgram = dgram.unsqueeze(0).expand(num_templates, -1, -1, -1)
        num_ensembles = batch["template_aatype"].shape[-1] if batch["template_aatype"].dim() > 2 else 1
        dgram = dgram.unsqueeze(-1).expand(-1, -1, -1, -1, num_ensembles)
        device = batch["template_aatype"].device
        dtype = (
            batch["template_all_atom_positions"].dtype
            if "template_all_atom_positions" in batch
            else distogram_probs.dtype
        )
        batch["template_dgram_probs"] = dgram.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Single-sample batch construction
    # ------------------------------------------------------------------

    def build_batch(
        self,
        distogram_probs: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
        *,
        template_mode: str = "distogram_only",
        template_mmcif_path: Optional[str] = None,
        template_chain_id: Optional[str] = None,
        kalign_binary_path: Optional[str] = None,
        mask_template_aatype: bool = False,
        zero_template_unit_vector: bool = False,
        zero_template_torsion_angles: bool = False,
        _pad_to_length: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Build a feature dict for a single sample.

        Args:
            distogram_probs: [1, L, L, 39] float probabilities
            residue_type: [1, L] ints in [0..20] (20=unknown)
            mask: [1, L] bool/float
            _pad_to_length: (internal) Pad features to this length for
                batching with variable-length sequences.  When None the
                features are truncated to the true (unmasked) length.
            seed: (optional) If set, seed all RNG sources before running
                the feature pipeline so that processing is deterministic.
                The feature pipeline otherwise introduces randomness via
                make_masked_msa (15% random MSA masking) even with a
                single-sequence MSA.
        Returns:
            dict of tensors (no batch dim) ready for the AlphaFold model.
        """
        b, n = residue_type.shape[:2]
        if b != 1:
            raise ValueError(
                f"build_batch expects a single sample (batch dim 1), got {b}. "
                "Use build_batch_multi / forward_batched for multiple samples."
            )

        # Compute true (unmasked) length
        if mask.dtype != torch.float32:
            mask_f = mask.float()
        else:
            mask_f = mask
        l = int(mask_f[0].sum().item())
        if l <= 0:
            raise ValueError("Mask has no valid residues")

        # Target length: true length, or padded for batching
        target_l = l if _pad_to_length is None else max(l, _pad_to_length)

        # Truncate residue_type to true length, optionally pad with X (20)
        residue_type_trunc = residue_type[:, :l]
        if target_l > l:
            pad_rt = torch.full(
                (1, target_l - l), rc.restype_num,
                dtype=residue_type.dtype, device=residue_type.device,
            )
            residue_type = torch.cat([residue_type_trunc, pad_rt], dim=1)
        else:
            residue_type = residue_type_trunc

        # Mask: 1.0 for real residues, 0.0 for padding
        mask_f = torch.zeros((1, target_l), dtype=torch.float32, device=residue_type.device)
        mask_f[0, :l] = 1.0

        # Sequence string (pad with X)
        seq = self._restype_idx_to_str(residue_type[0][:l])
        if target_l > l:
            seq += "X" * (target_l - l)

        template_mode = template_mode.lower()
        if template_mode == "full_template_zero_coords":
            zero_template_unit_vector = True
            zero_template_torsion_angles = True

        # Truncate (and pad) distogram to target_l
        if distogram_probs is not None:
            distogram_probs = distogram_probs[:, :l, :l, :]
            if target_l > l:
                pad = target_l - l
                distogram_probs = F.pad(distogram_probs, (0, 0, 0, pad, 0, pad))

        if template_mode == "distogram_only":
            if distogram_probs is None:
                raise ValueError("distogram_probs is required for template_mode=distogram_only")
            mask_np = mask_f[0].detach().cpu().numpy()
            if template_mmcif_path is not None:
                if kalign_binary_path is None:
                    raise ValueError("kalign_binary_path is required for custom-template distogram mode")
                if template_chain_id is None:
                    raise ValueError("template_chain_id is required for custom-template distogram mode")
                pdb_id = os.path.splitext(os.path.basename(template_mmcif_path))[0]
                raw = make_sequence_features_with_custom_template(
                    sequence=seq,
                    mmcif_path=template_mmcif_path,
                    pdb_id=pdb_id,
                    chain_id=template_chain_id,
                    kalign_binary_path=kalign_binary_path,
                    rm_template_sequence=self.rm_template_sequence,
                    skip_alignment=self.skip_template_alignment,
                )
            else:
                raw = self._make_template_stub_features(
                    sequence=seq,
                    mask=mask_np,
                    rm_template_sequence=self.rm_template_sequence,
                )
            zero_template_unit_vector = True
            zero_template_torsion_angles = True
        elif template_mode in ("full_template", "full_template_zero_coords"):
            if template_mmcif_path is None:
                raise ValueError("template_mmcif_path is required for full template modes")
            if kalign_binary_path is None:
                raise ValueError("kalign_binary_path is required for full template modes")
            if template_chain_id is None:
                raise ValueError("template_chain_id is required for full template modes")
            pdb_id = os.path.splitext(os.path.basename(template_mmcif_path))[0]
            raw = make_sequence_features_with_custom_template(
                sequence=seq,
                mmcif_path=template_mmcif_path,
                pdb_id=pdb_id,
                chain_id=template_chain_id,
                kalign_binary_path=kalign_binary_path,
                rm_template_sequence=self.rm_template_sequence,
                skip_alignment=self.skip_template_alignment,
            )
        else:
            raise ValueError(
                "Unknown template_mode. Expected one of: distogram_only, full_template, full_template_zero_coords"
            )
        # Seed all RNG sources so the feature pipeline is deterministic.
        # The primary source of randomness is make_masked_msa (randomly
        # replaces 15% of MSA positions) and ensemble_seed in
        # input_pipeline.process_tensors_from_config, both of which use
        # unseeded global RNG by default.
        if seed is not None:
            _random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        feats = self.feature_pipeline.process_features(raw, mode="predict", is_multimer=False)

        batch = tensor_tree_map(lambda x: x.to(self.device), feats)

        # When padding was applied, the feature pipeline sets seq_mask /
        # msa_mask to all-ones for the padded length.  Override them so the
        # model correctly ignores padding positions.
        if target_l > l:
            self._apply_padding_masks(batch, true_length=l, padded_length=target_l)

        if template_mode == "distogram_only":
            self._inject_template_dgram_probs(batch, distogram_probs)
        self._set_template_use_unit_vector(zero_template_unit_vector)
        self._apply_template_overrides(
            batch,
            mask_template_aatype=mask_template_aatype,
            zero_template_torsion_angles=zero_template_torsion_angles,
        )
        return batch

    @staticmethod
    def _apply_padding_masks(
        batch: dict,
        true_length: int,
        padded_length: int,
    ) -> None:
        """Zero out masks for padding positions (indices >= true_length).

        The feature pipeline creates masks assuming the full sequence is valid.
        After padding for batching, we override seq_mask, msa_mask, and
        extra_msa_mask so the model ignores padded positions.
        """
        # Map from feature key -> dimension index where N_res appears.
        # The last dimension is always the recycling dim for processed features.
        mask_keys = {
            "seq_mask": 0,           # [N_res, R]
            "msa_mask": 1,           # [N_seq, N_res, R]
            "extra_msa_mask": 1,     # [N_extra, N_res, R]
        }
        for key, res_dim in mask_keys.items():
            if key not in batch:
                continue
            t = batch[key]
            # Build a slice that zeros positions [true_length:] along res_dim
            slices = [slice(None)] * t.dim()
            slices[res_dim] = slice(true_length, None)
            t[tuple(slices)] = 0.0

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def build_batch_multi(
        self,
        distogram_probs_list: List[torch.Tensor],
        residue_type_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        **kwargs,
    ) -> dict:
        """Build a batched feature dict from multiple samples.

        Each input element should have shape [1, L_i, ...] (single-sample
        with potentially different lengths L_i).  Samples are padded to the
        maximum true length and stacked along a new leading batch dimension.

        Args:
            distogram_probs_list: list of [1, L_i, L_i, 39] tensors
            residue_type_list:    list of [1, L_i] tensors
            mask_list:            list of [1, L_i] tensors
            **kwargs: forwarded to build_batch (template_mode, etc.)

        Returns:
            dict of tensors with a leading batch dimension [B, ...].
        """
        n_samples = len(distogram_probs_list)
        if n_samples == 0:
            raise ValueError("Empty sample list")
        if not (n_samples == len(residue_type_list) == len(mask_list)):
            raise ValueError("All input lists must have the same length")

        # Compute true (unmasked) lengths to determine padding target
        lengths = []
        for mask in mask_list:
            mask_f = mask.float() if mask.dtype != torch.float32 else mask
            lengths.append(int(mask_f[0].sum().item()) if mask_f.dim() >= 2 else int(mask_f.sum().item()))
        max_l = max(lengths)

        # Build each single-sample batch, padded to max_l
        batches = []
        for i in range(n_samples):
            single = self.build_batch(
                distogram_probs_list[i],
                residue_type_list[i],
                mask_list[i],
                _pad_to_length=max_l,
                **kwargs,
            )
            batches.append(single)

        # Stack along a new leading batch dimension
        collated = {}
        for key in batches[0].keys():
            collated[key] = torch.stack([b[key] for b in batches], dim=0)
        return collated

    def forward_batched(
        self,
        distogram_probs_list: List[torch.Tensor],
        residue_type_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        **kwargs,
    ) -> dict:
        """Run batched inference on multiple samples.

        Accepts lists of per-sample tensors (each with batch dim 1), pads to
        the longest sequence, stacks into a true batch, and runs a single
        forward pass through the model.

        Args:
            distogram_probs_list: list of [1, L_i, L_i, 39] tensors
            residue_type_list:    list of [1, L_i] tensors
            mask_list:            list of [1, L_i] tensors
            **kwargs: forwarded to build_batch (template_mode, etc.)

        Returns:
            dict of model outputs.  Coordinate tensors have shape
            [B, L_max, ...]; use the per-sample masks to extract valid
            residues.
        """
        batch = self.build_batch_multi(
            distogram_probs_list,
            residue_type_list,
            mask_list,
            **kwargs,
        )
        with torch.no_grad():
            out = self.model(batch)
        return out

    # ------------------------------------------------------------------
    # Original single-sample forward (backward-compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        distogram_probs: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
        *,
        template_mode: str = "distogram_only",
        template_mmcif_path: Optional[str] = None,
        template_chain_id: Optional[str] = None,
        kalign_binary_path: Optional[str] = None,
        mask_template_aatype: bool = False,
        zero_template_unit_vector: bool = False,
        zero_template_torsion_angles: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            distogram_probs: [B, L, L, 39] float probabilities
            residue_type: [B, L] ints in [0..20] (20=unknown)
            mask: [B, L] bool/float
            seed: (optional) Seed for deterministic feature pipeline processing.
        Returns:
            dict with atom37 coordinates [B, L, 37, 3]
        """
        batch = self.build_batch(
            distogram_probs,
            residue_type,
            mask,
            template_mode=template_mode,
            template_mmcif_path=template_mmcif_path,
            template_chain_id=template_chain_id,
            kalign_binary_path=kalign_binary_path,
            mask_template_aatype=mask_template_aatype,
            zero_template_unit_vector=zero_template_unit_vector,
            zero_template_torsion_angles=zero_template_torsion_angles,
            seed=seed,
        )
        with torch.no_grad():
            out = self.model(batch)
        return out
