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
from openfold.data.data_pipeline import make_dummy_msa_feats, make_sequence_features_with_distogram_template, make_sequence_features
from openfold.utils.tensor_utils import tensor_tree_map


class OpenFoldTemplateInference(nn.Module):
    """
    Minimal OpenFold-based structure generator that maps a predicted distogram
    (pair logits) + residue types + mask to atom37 coordinates using the
    template pair stack and structure module.
    """

    def __init__(
        self,
        num_bins: int = 39,
        c_t: int = 64,
        c_z: int = 128,
        c_s: int = 384,
        c_ipa: int = 16,
        c_resnet: int = 128,
        no_heads_ipa: int = 12,
        no_qk_points: int = 4,
        no_v_points: int = 8,
        no_blocks: int = 8,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.chunk_size = chunk_size

        # Template embedding pieces
        # TemplatePairEmbedder expects the concatenated template_pair_feat as input.
        # We build a simplified feature set from the predicted distogram.
        self.template_pair_embedder = TemplatePairEmbedder(
            c_in=self._template_pair_feat_dim(num_bins),
            c_out=c_t,
        )
        self.template_pair_stack = TemplatePairStack(
            c_t=c_t,
            c_hidden_tri_att=c_t,
            c_hidden_tri_mul=c_t,
            no_blocks=2,
            no_heads=4,
            pair_transition_n=2,
            dropout_rate=0.0,
            blocks_per_ckpt=None,
            inf=1e8,
        )
        self.template_pointwise_att = TemplatePointwiseAttention(
            c_t=c_t,
            c_z=c_z,
            c_hidden=c_t,
            no_heads=4,
            inf=1e8,
        )

        # Pair and single projections
        self.pair_linear = nn.Linear(num_bins, c_z)
        self.single_linear = nn.Linear(rc.restype_num + 1, c_s)

        # Structure module mirrors AlphaFold defaults
        self.structure_module = StructureModule(
            c_s=c_s,
            c_z=c_z,
            c_ipa=c_ipa,
            c_resnet=c_resnet,
            no_heads_ipa=no_heads_ipa,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            dropout_rate=0.0,
            no_blocks=no_blocks,
            no_transition_layers=1,
            no_resnet_blocks=2,
            no_angles=7,
            trans_scale_factor=10.0,
            epsilon=1e-8,
            inf=1e8,
        )

    @staticmethod
    def _template_pair_feat_dim(num_bins: int) -> int:
        # dgram bins + mask + two residue one-hots + 3 unit-vector slots + mask
        return (
            num_bins
            + 1
            + 2 * (rc.restype_num + 2)
            + 3
            + 1
        )

    @staticmethod
    def _build_template_pair_feat(
        pair_probs: torch.Tensor,
        residue_type: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a template_pair_feat analog from distogram probabilities.

        pair_probs: [b, n, n, num_bins] probability over distance bins
        residue_type: [b, n] ints
        pair_mask: [b, n, n] bool/float
        """
        device = pair_probs.device
        dtype = pair_probs.dtype
        template_mask_2d = pair_mask.unsqueeze(-1).to(dtype)  # [b, n, n, 1]

        aatype = residue_type.clamp(min=0, max=rc.restype_num)
        aatype_one_hot = F.one_hot(aatype, rc.restype_num + 2).to(dtype)
        b, n_res = aatype_one_hot.shape[:2]

        aatype_i = aatype_one_hot[:, :, None, :].expand(b, n_res, n_res, -1)
        aatype_j = aatype_one_hot[:, None, :, :].expand(b, n_res, n_res, -1)

        to_concat = (
            pair_probs,                  # [b, n, n, num_bins]
            template_mask_2d,            # [b, n, n, 1]
            aatype_i,                    # [b, n, n, 22]
            aatype_j,                    # [b, n, n, 22]
            torch.zeros(                 # unit vectors placeholder
                pair_probs.shape[:-1] + (3,),
                device=device,
                dtype=dtype,
            ),
            template_mask_2d,            # mask again
        )

        act = torch.cat(to_concat, dim=-1) * template_mask_2d
        return act

    @staticmethod
    def _atom14_to_atom37(
        atom14: torch.Tensor,
        residue_type: torch.Tensor,
    ) -> torch.Tensor:
        restype_atom14_to_atom37 = torch.tensor(
            rc.RESTYPE_ATOM14_TO_ATOM37, device=atom14.device, dtype=torch.long
        )
        restype_atom37_mask = torch.tensor(
            rc.RESTYPE_ATOM37_MASK, device=atom14.device, dtype=atom14.dtype
        )
        rt = residue_type.clamp(min=0, max=20)
        map14_to_37 = restype_atom14_to_atom37[rt]  # [b, n, 14]
        atom37 = atom14.new_zeros(atom14.shape[:-2] + (37, 3))
        atom37 = atom37.scatter(
            2, map14_to_37.unsqueeze(-1).expand_as(atom37[..., :14, :]), atom14
        )
        mask37 = restype_atom37_mask[rt].unsqueeze(-1)
        return atom37 * mask37

    def forward(
        self,
        pair_logits: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Args:
            pair_logits: [b, n, n, num_bins]
            residue_type: [b, n]
            mask: [b, n] boolean/float
        Returns:
            dict with atom14, atom37 coordinates and pair/single reps used.
        """
        pair_probs = torch.softmax(pair_logits, dim=-1)
        mask_f = mask.to(pair_logits.dtype)
        pair_mask = mask_f[..., None] * mask_f[..., None, :]

        template_pair_feat = self._build_template_pair_feat(
            pair_probs, residue_type, pair_mask
        )
        # Add template dimension
        template_pair_feat = template_pair_feat.unsqueeze(-4)

        template_mask_pair = pair_mask.unsqueeze(-3).to(pair_logits.dtype)
        template_embed = self.template_pair_embedder(template_pair_feat)
        template_stack = self.template_pair_stack(
            template_embed,
            mask=template_mask_pair,
            chunk_size=self.chunk_size,
            _mask_trans=True,
        )

        z_base = self.pair_linear(pair_probs)
        template_mask = torch.ones(
            template_stack.shape[:-3], device=pair_logits.device, dtype=pair_logits.dtype
        )
        z_update = self.template_pointwise_att(
            template_stack,
            z_base,
            template_mask=template_mask,
            chunk_size=self.chunk_size,
        )
        z = (z_base + z_update) * pair_mask.unsqueeze(-1)

        rt_one_hot = F.one_hot(
            residue_type.clamp(min=0, max=rc.restype_num),
            rc.restype_num + 1,
        ).to(pair_logits.dtype)
        s = self.single_linear(rt_one_hot)
        s = s * mask_f[..., None]

        structure_outputs = self.structure_module(
            {"single": s, "pair": z},
            aatype=residue_type.clamp(min=0, max=rc.restype_num),
            mask=mask_f,
            inplace_safe=False,
            _offload_inference=False,
        )

        atom14 = structure_outputs["positions"][-1]
        atom37 = self._atom14_to_atom37(atom14, residue_type)

        return {
            "atom14": atom14,
            "atom37": atom37,
            "pair": z,
            "single": s,
        }


class OpenFoldDistogramOnlyInference(nn.Module):
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
        template_sequence_all_x: bool = False,
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
                "OpenFoldDistogramOnlyInference requires CUDA (OpenFold attention CUDA kernels are enabled). "
                f"Got device={device}."
            )
        self.device = device
        self.model = self.model.to(self.device)
        self.feature_pipeline = FeaturePipeline(self.cfg.data)
        self.template_sequence_all_x = template_sequence_all_x

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

    def forward(
        self,
        distogram_probs: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
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
        distogram_probs = distogram_probs[:, :l, :l, :]
        mask_f = torch.ones((1, l), dtype=torch.float32, device=distogram_probs.device)

        seq = self._restype_idx_to_str(residue_type[0])
        dist_np = distogram_probs[0].detach().cpu().numpy()
        mask_np = mask_f[0].detach().cpu().numpy()
        print("distogram_probs.shape", distogram_probs.shape)
        print("dist_np.shape", dist_np.shape)
        print("mask_np.shape", mask_np.shape)
        print("seq", seq)
        
        raw = make_sequence_features_with_distogram_template(
            sequence=seq,
            distogram_probs=dist_np,
            mask=mask_np,
            pdb_id="distogram_template",
            template_sequence_all_x=self.template_sequence_all_x,
        )
        feats = self.feature_pipeline.process_features(raw, mode="predict", is_multimer=False)

        batch = tensor_tree_map(lambda x: x.to(self.device), feats)
        with torch.no_grad():
            out = self.model(batch)

        atom_pos = out["final_atom_positions"]
        # OpenFold may return:
        # - atom37: [B, L, 37, 3]
        # - atom37 without batch: [L, 37, 3]
        # - CA-only: [B, L, 3]
        # - CA-only without batch: [L, 3]
        # Normalize to atom37: [B, L, 37, 3].
        if atom_pos.dim() == 4:
            if atom_pos.shape[-2] != rc.atom_type_num or atom_pos.shape[-1] != 3:
                raise ValueError(f"Unexpected final_atom_positions shape: {tuple(atom_pos.shape)}")
            atom37 = atom_pos
        elif atom_pos.dim() == 3:
            if atom_pos.shape[-2] == rc.atom_type_num and atom_pos.shape[-1] == 3:
                # [L, 37, 3] -> [1, L, 37, 3]
                atom37 = atom_pos[None, ...]
            elif atom_pos.shape[-1] == 3 and atom_pos.shape[-2] != rc.atom_type_num:
                # [B, L, 3] -> [B, L, 37, 3] with only CA filled
                bsz, l, _ = atom_pos.shape
                atom37 = atom_pos.new_zeros((bsz, l, rc.atom_type_num, 3))
                atom37[:, :, rc.atom_order["CA"], :] = atom_pos
            else:
                raise ValueError(f"Unexpected final_atom_positions shape: {tuple(atom_pos.shape)}")
        elif atom_pos.dim() == 2:
            # [L, 3] -> [1, L, 37, 3] with only CA filled
            if atom_pos.shape[-1] != 3:
                raise ValueError(f"Unexpected final_atom_positions shape: {tuple(atom_pos.shape)}")
            l, _ = atom_pos.shape
            atom37 = atom_pos.new_zeros((1, l, rc.atom_type_num, 3))
            atom37[0, :, rc.atom_order["CA"], :] = atom_pos
        else:
            raise ValueError(f"Unexpected final_atom_positions ndim: {atom_pos.dim()} shape={tuple(atom_pos.shape)}")
        return {"atom37": atom37}

