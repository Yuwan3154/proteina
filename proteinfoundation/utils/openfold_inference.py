import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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


class OpenFoldTemplateInference(nn.Module):
    """
    Full OpenFold AlphaFold model inference using a distogram-only template.

    This is intended for Proteina contact-map diffusion logging/inference:
    given a predicted distogram probability tensor [B, L, L, 39] and the
    target sequence, produce atom37 coordinates.
    """

    def __init__(
        self,
        *,
        model_name: str = "model_1_ptm",
        jax_params_path: str = "/home/ubuntu/params/params_model_1_ptm.npz",
        device: Optional[torch.device] = None,
        rm_template_sequence: Optional[bool] = False,
        skip_template_alignment: bool = False,
        max_recycling_iters: Optional[int] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.jax_params_path = jax_params_path
        self.cfg = model_config(model_name)
        self.cfg.data.common.use_templates = True
        if max_recycling_iters is not None:
            self.cfg.data.common.max_recycling_iters = int(max_recycling_iters)

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
    ):
        """
        Args:
            distogram_probs: [B, L, L, 39] float probabilities
            residue_type: [B, L] ints in [0..20] (20=unknown)
            mask: [B, L] bool/float
        Returns:
            dict with atom37 coordinates [B, L, 37, 3]
        """
        b, n = residue_type.shape[:2]
        if b != 1:
            raise ValueError(f"Expected batch size 1 for distogram-only inference, got {b}")

        # Truncate to true length for speed
        if mask.dtype != torch.float32:
            mask_f = mask.float()
        else:
            mask_f = mask
        l = int(mask_f[0].sum().item())
        if l <= 0:
            raise ValueError("Mask has no valid residues")

        residue_type = residue_type[:, :l]
        mask_f = torch.ones((1, l), dtype=torch.float32, device=residue_type.device)

        seq = self._restype_idx_to_str(residue_type[0])
        template_mode = template_mode.lower()
        if template_mode == "full_template_zero_coords":
            zero_template_unit_vector = True
            zero_template_torsion_angles = True

        if distogram_probs is not None:
            distogram_probs = distogram_probs[:, :l, :l, :]

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
        feats = self.feature_pipeline.process_features(raw, mode="predict", is_multimer=False)

        batch = tensor_tree_map(lambda x: x.to(self.device), feats)
        if template_mode == "distogram_only":
            self._inject_template_dgram_probs(batch, distogram_probs)
        self._set_template_use_unit_vector(zero_template_unit_vector)
        self._apply_template_overrides(
            batch,
            mask_template_aatype=mask_template_aatype,
            zero_template_torsion_angles=zero_template_torsion_angles,
        )
        return batch

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
    ):
        """
        Args:
            distogram_probs: [B, L, L, 39] float probabilities
            residue_type: [B, L] ints in [0..20] (20=unknown)
            mask: [B, L] bool/float
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
        )
        with torch.no_grad():
            out = self.model(batch)
        return out

