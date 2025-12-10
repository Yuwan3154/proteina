import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from openfold.model.template import TemplatePairStack, TemplatePointwiseAttention
from openfold.model.embedders import TemplatePairEmbedder
from openfold.model.structure_module import StructureModule
import openfold.np.residue_constants as rc


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

