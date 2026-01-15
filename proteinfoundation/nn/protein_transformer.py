# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Dict, Literal, Optional

import einops
import torch
from torch.utils.checkpoint import checkpoint

from proteinfoundation.openfold_stub.model.msa import MSARowAttentionWithPairBias
from proteinfoundation.openfold_stub.model.pair_transition import PairTransition
from proteinfoundation.openfold_stub.model.structure_module import (
    InvariantPointAttention,
    StructureModule,
)
from proteinfoundation.openfold_stub.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from proteinfoundation.openfold_stub.utils.rigid_utils import Rigid
import proteinfoundation.openfold_stub.np.residue_constants as rc

# Try to import cuequivariance for faster triangle multiplicative updates
try:
    import cuequivariance_torch as cuet
    CUEQUIVARIANCE_AVAILABLE = True
except ImportError:
    CUEQUIVARIANCE_AVAILABLE = False

from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.pair_bias_attn.pair_bias_attn import PairBiasAttention
from proteinfoundation.nn.alphafold3_pytorch_utils.modules import (
    AdaptiveLayerNorm,
    AdaptiveLayerNormOutputScale,
    Transition,
)


class MultiHeadAttention(torch.nn.Module):
    """Typical multi-head self-attention attention using pytorch's module."""

    def __init__(self, dim_token, nheads, dropout=0.0):
        super().__init__()

        self.to_q = torch.nn.Linear(dim_token, dim_token)
        self.to_kv = torch.nn.Linear(dim_token, 2 * dim_token, bias=False)

        self.mha = torch.nn.MultiheadAttention(
            embed_dim=dim_token,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input sequence, shape [b, n, dim_token]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence, shape [b, n, dim_token]
        """
        query = self.to_q(x)  # [b, n, dim_token]
        key, value = self.to_kv(x).chunk(2, dim=-1)  # Each [b, n, dim_token]
        return (
            self.mha(
                query=query,
                key=key,
                value=value,
                key_padding_mask=~mask,  # Indicated what should be ignores with True, that's why the ~
                need_weights=False,
                is_causal=False,
            )[0]
            * mask[..., None]
        )  # [b, n, dim_token]


class MultiHeadBiasedAttention(torch.nn.Module):
    """Multi-head self-attention with pair bias, based on openfold."""

    def __init__(self, dim_token, dim_pair, nheads, dropout=0.0):
        super().__init__()

        self.row_attn_pair_bias = MSARowAttentionWithPairBias(
            c_m=dim_token,
            c_z=dim_pair,
            c_hidden=int(dim_token // nheads),  # Per head dimension
            no_heads=nheads,
        )

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [b, n, dim_token]
            pair_rep: Pair representation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token]
        """
        # Add extra dimension for MSA, unused here but required by openfold
        x = einops.rearrange(x, "b n d -> b () n d")  # [b, 1, n, dim_token]
        mask = einops.rearrange(mask, "b n -> b () n") * 1.0  # float [b, 1, n]
        x = self.row_attn_pair_bias(x, pair_rep, mask)  # [b, 1, n, dim_token]
        x = x * mask[..., None]
        x = einops.rearrange(
            x, "b () n c -> b n c"
        )  # Remove extra dimension [b, n, dim_token]
        return x


class MultiHeadAttentionADALN(torch.nn.Module):
    """Typical multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, nheads, dim_cond, dropout=0.0):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = MultiHeadAttention(
            dim_token=dim_token, nheads=nheads, dropout=dropout
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        x = self.adaln(x, cond, mask)
        x = self.mha(x, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiHeadBiasedAttentionADALN(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, dropout=0.0):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = MultiHeadBiasedAttention(
            dim_token=dim_token, dim_pair=dim_pair, nheads=nheads, dropout=dropout
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        x = self.adaln(x, cond, mask)
        x = self.mha(x, pair_rep, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling aplied to output."""

    def __init__(self, *, dim, dim_cond, expansion_factor=4):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.transition = Transition(
            dim=dim, expansion_factor=expansion_factor, layer_norm=False
        )
        self.scale_output = AdaptiveLayerNormOutputScale(dim=dim, dim_cond=dim_cond)

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        x = self.adaln(x, cond, mask)  # [b, n, dim]
        x = self.transition(x, mask)  # [b, n, dim]
        x = self.scale_output(x, cond, mask)  # [b, n, dim]
        return x * mask[..., None]  # [b, n, dim]


class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        dim_cond,
        residual_mha,
        residual_transition,
        parallel_mha_transition,
        use_attn_pair_bias,
        use_qkln,
        dropout=0.0,
        expansion_factor=4,
    ):
        super().__init__()
        self.parallel = parallel_mha_transition
        self.use_attn_pair_bias = use_attn_pair_bias

        # If parallel do not allow both layers to have a residual connection since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadBiasedAttentionADALN_MM(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            dim_cond=dim_cond,
            use_qkln=use_qkln,
        )

        self.transition = TransitionADALN(
            dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor
        )

    def _apply_mha(self, x, pair_rep, cond, mask):
        x_attn = self.mhba(x, pair_rep, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored), shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, cond, mask) + self._apply_transition(
                x, cond, mask
            )
        else:
            x = self._apply_mha(x, pair_rep, cond, mask)
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]


class PairReprUpdate(torch.nn.Module):
    """Layer to update the pair representation."""

    def __init__(
        self,
        token_dim,
        pair_dim,
        expansion_factor_transition=2,
        use_tri_mult=False,
        tri_mult_c=196,
        use_cueq: bool = True,
    ):
        super().__init__()

        self.use_tri_mult = use_tri_mult
        # Explicit runtime gate for cuequivariance (in addition to availability/shape checks)
        self.use_cueq = bool(use_cueq)
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)

        if use_tri_mult:
            tri_mult_c = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TriangleMultiplicationOutgoing(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
            self.tri_mult_in = TriangleMultiplicationIncoming(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
            # Check if cuequivariance can be used (requires c_z == c_hidden and both multiples of 32)
            self._cuequivariance_compatible = (
                CUEQUIVARIANCE_AVAILABLE
                and pair_dim == tri_mult_c
                and pair_dim % 32 == 0
            )
        self.transition_out = PairTransition(
            c_z=pair_dim, n=expansion_factor_transition
        )

    def _apply_mask(self, pair_rep, pair_mask):
        """
        pair_rep has shape [b, n, n, pair_dim]
        pair_mask has shape [b, n, n]
        """
        return pair_rep * pair_mask[..., None]

    def _run_triangle_mult_cuet(
        self,
        x: torch.Tensor,
        tri_mult_module: torch.nn.Module,
        direction: Literal["outgoing", "incoming"],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run triangle multiplicative update using cuequivariance_torch.
        
        This function extracts weights from an OpenFold TriangleMultiplication module
        and runs the optimized cuequivariance implementation.
        
        Args:
            x: Input tensor of shape [B, N, N, D]
            tri_mult_module: OpenFold TriangleMultiplicationOutgoing or TriangleMultiplicationIncoming
            direction: "outgoing" or "incoming"
            mask: Optional mask tensor of shape [B, N, N]
        
        Returns:
            Output tensor of shape [B, N, N, D]
        """
        # Extract weights from OpenFold module
        # Input normalization
        norm_in_weight = tri_mult_module.layer_norm_in.weight
        norm_in_bias = tri_mult_module.layer_norm_in.bias
        
        # Input projections: stack a and b projections
        # p_in combines linear_a_p and linear_b_p: [2*c_hidden, c_z]
        p_in_weight = torch.cat([
            tri_mult_module.linear_a_p.weight,
            tri_mult_module.linear_b_p.weight
        ], dim=0)
        p_in_bias = torch.cat([
            tri_mult_module.linear_a_p.bias,
            tri_mult_module.linear_b_p.bias
        ], dim=0)
        
        # Input gating: stack a and b gates: [2*c_hidden, c_z]
        g_in_weight = torch.cat([
            tri_mult_module.linear_a_g.weight,
            tri_mult_module.linear_b_g.weight
        ], dim=0)
        g_in_bias = torch.cat([
            tri_mult_module.linear_a_g.bias,
            tri_mult_module.linear_b_g.bias
        ], dim=0)
        
        # Output normalization
        norm_out_weight = tri_mult_module.layer_norm_out.weight
        norm_out_bias = tri_mult_module.layer_norm_out.bias
        
        # Output projection: linear_z maps from c_hidden to c_z
        p_out_weight = tri_mult_module.linear_z.weight
        p_out_bias = tri_mult_module.linear_z.bias
        
        # Output gating: linear_g maps c_z to c_z
        g_out_weight = tri_mult_module.linear_g.weight
        g_out_bias = tri_mult_module.linear_g.bias
        
        return cuet.triangle_multiplicative_update(
            x=x,
            direction=direction,
            mask=mask,
            norm_in_weight=norm_in_weight,
            norm_in_bias=norm_in_bias,
            p_in_weight=p_in_weight,
            p_in_bias=p_in_bias,
            g_in_weight=g_in_weight,
            g_in_bias=g_in_bias,
            norm_out_weight=norm_out_weight,
            norm_out_bias=norm_out_bias,
            p_out_weight=p_out_weight,
            p_out_bias=p_out_bias,
            g_out_weight=g_out_weight,
            g_out_bias=g_out_bias,
        )

    def _tri_mult_out_cuet(self, pair_rep, pair_mask):
        """Wrapper for cuequivariance triangle mult outgoing (for checkpointing)."""
        return self._run_triangle_mult_cuet(pair_rep, self.tri_mult_out, "outgoing", pair_mask)
    
    def _tri_mult_in_cuet(self, pair_rep, pair_mask):
        """Wrapper for cuequivariance triangle mult incoming (for checkpointing)."""
        return self._run_triangle_mult_cuet(pair_rep, self.tri_mult_in, "incoming", pair_mask)

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [b, n, token_dim]
            pair_rep: Input pair representation, shape [b, n, n, pair_dim]
            mask: binary mask, shape [b, n]

        Returns:
            Updated pair representation, shape [b, n, n, pair_dim].
        """
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        x = x * mask[..., None]  # [b, n, token_dim]
        x_proj_1, x_proj_2 = self.linear_x(self.layer_norm_in(x)).chunk(
            2, dim=-1
        )  # [b, n, pair_dim] each
        pair_rep = (
            pair_rep + x_proj_1[:, None, :, :] + x_proj_2[:, :, None, :]
        )  # [b, n, n, pair_dim]
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        if self.use_tri_mult:
            if self.use_cueq and CUEQUIVARIANCE_AVAILABLE and self._cuequivariance_compatible:
                # Use fast cuequivariance implementation (requires c_z == c_hidden and both % 32 == 0)
                pair_rep = pair_rep + checkpoint(
                    self._tri_mult_out_cuet, *(pair_rep, pair_mask), use_reentrant=False
                )
                pair_rep = self._apply_mask(pair_rep, pair_mask)
                pair_rep = pair_rep + checkpoint(
                    self._tri_mult_in_cuet, *(pair_rep, pair_mask), use_reentrant=False
                )
                pair_rep = self._apply_mask(pair_rep, pair_mask)
            else:
                # Fall back to OpenFold implementation
                pair_rep = pair_rep + checkpoint(
                    self.tri_mult_out, *(pair_rep, pair_mask * 1.0), use_reentrant=False
                )
                pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
                pair_rep = pair_rep + checkpoint(
                    self.tri_mult_in, *(pair_rep, pair_mask * 1.0), use_reentrant=False
                )
                pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        # NOTE: Avoid non-reentrant checkpointing here. PairTransition has shown
        # unstable saved-tensor metadata under checkpointing, leading to
        # CheckpointError in backward. This transition is relatively cheap
        # vs the triangle multiplicative updates, so we run it without ckpt.
        pair_rep = pair_rep + self.transition_out(pair_rep, pair_mask * 1.0)
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        return pair_rep


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, **kwargs):
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            use_residue_type_emb=kwargs.get("residue_type_emb_init_pair", False),
            use_cirpin_emb=kwargs.get("cirpin_emb_init_pair", False),
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = FeatureFactory(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    use_residue_type_emb=kwargs.get("residue_type_emb_cond_pair", False),
                    use_cirpin_emb=kwargs.get("cirpin_emb_cond_pair", False),
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(
                    dim=dim_feats_out, dim_cond=dim_cond_pair
                )

    def forward(self, batch_nn):
        mask = batch_nn["mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
            repr = self.adaln(repr, cond, pair_mask)
        return repr


class ProteinTransformerAF3(torch.nn.Module):
    """
    Final neural network mimicking the one used in AF3 diffusion. It consists of:

    (1) Input preparation
    (1.a) Initial sequence representation from features
    (1.b) Embed coordaintes and add to initial sequence representation
    (1.c) Conditioning variables from features

    (2) Main trunk
    (2.a) A sequence of layers similar to algorithm 23 of AF3 (multi head attn, transition) using adaptive layer norm
    and adaptive output scaling (also from adaptive layer norm paper)

    (3) Recovering 3D coordinates
    (3.a) A layer that takes as input tokens and produces coordinates
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(ProteinTransformerAF3, self).__init__()
        self.use_attn_pair_bias = kwargs["use_attn_pair_bias"]
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_coors_on_the_fly = kwargs.get(
            "update_coors_on_the_fly", False
        )
        self.update_seq_with_coors = None
        self.update_pair_repr = kwargs.get(
            "update_pair_repr", False
        )
        self.update_pair_repr_every_n = kwargs.get(
            "update_pair_repr_every_n", 2
        )
        self.use_tri_mult = kwargs.get("use_tri_mult", False)
        self.feats_pair_cond = kwargs.get("feats_pair_cond", [])
        self.use_qkln = kwargs.get("use_qkln", False)
        self.use_torch_compile = bool(kwargs.get("use_torch_compile", True))
        self.num_buckets_predict_pair = kwargs.get(
            "num_buckets_predict_pair", None
        )

        # Contact map diffusion mode
        self.contact_map_mode = kwargs.get("contact_map_mode", False)
        if self.contact_map_mode:
            # Validation: contact map mode requires triangle multiplicative updates
            # for proper pairwise distance reasoning
            if not self.update_pair_repr:
                raise ValueError(
                    "contact_map_mode=True requires update_pair_repr=True. "
                    "Pair representation updates are needed for contact map prediction."
                )
            if not self.use_tri_mult:
                raise ValueError(
                    "contact_map_mode=True requires use_tri_mult=True. "
                    "Triangle multiplicative updates are essential for contact map "
                    "diffusion as they enable propagation of pairwise distance "
                    "constraints through the triangle inequality."
                )
            if not self.update_pair_repr:
                raise ValueError(
                    "contact_map_mode=True requires update_pair_repr=True. "
                    "Pair representation updates are needed for contact map prediction."
                )

        # Registers
        self.num_registers = kwargs.get("num_registers", None)
        if self.num_registers is None or self.num_registers <= 0:
            self.num_registers = 0
            self.registers = None
        else:
            self.num_registers = int(self.num_registers)
            self.registers = torch.nn.Parameter(
                torch.randn(self.num_registers, self.token_dim) / 20.0
            )

        # Coordinate encoder/decoder (configurable, default: disabled in contact map mode)
        predict_coords_cfg = kwargs.get("predict_coords", not self.contact_map_mode)
        if self.contact_map_mode:
            if isinstance(predict_coords_cfg, bool):
                predict_coords_cfg = "linear" if predict_coords_cfg else None
            elif isinstance(predict_coords_cfg, str):
                predict_coords_cfg = predict_coords_cfg.lower()
                if predict_coords_cfg == "none":
                    predict_coords_cfg = None
                elif predict_coords_cfg not in ("ipa", "linear"):
                    raise ValueError(
                        f"Invalid predict_coords mode for contact_map_mode: {predict_coords_cfg}"
                    )
            elif predict_coords_cfg is None:
                predict_coords_cfg = None
            else:
                raise ValueError(
                    f"predict_coords must be bool or one of ['ipa','linear','none'], got {type(predict_coords_cfg)}"
                )
        else:
            predict_coords_cfg = bool(predict_coords_cfg)
        self.predict_coords = predict_coords_cfg
        non_contact_value = kwargs.get("non_contact_value", 0)
        if non_contact_value not in (0, -1):
            raise ValueError(f"non_contact_value must be 0 or -1, got {non_contact_value}")
        self.non_contact_value = non_contact_value
        if self.contact_map_mode:
            # To encode corrupted contact map
            self.linear_contact_embed = torch.nn.Linear(1, kwargs["pair_repr_dim"], bias=False)
        else:
            # To encode corrupted 3d positions
            self.linear_3d_embed = torch.nn.Linear(3, kwargs["token_dim"], bias=False)

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_init_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            use_residue_type_emb=kwargs.get("residue_type_emb_init_seq", False),
            use_cirpin_emb=kwargs.get("cirpin_emb_init_seq", False),
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            use_residue_type_emb=kwargs.get("residue_type_emb_cond_seq", False),
            use_cirpin_emb=kwargs.get("cirpin_emb_cond_seq", False),
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        if self.use_attn_pair_bias:
            self.pair_repr_builder = PairReprBuilder(
                feats_repr=kwargs["feats_pair_repr"],
                feats_cond=kwargs["feats_pair_cond"],
                dim_feats_out=kwargs["pair_repr_dim"],
                dim_cond_pair=kwargs["dim_cond"],
                **kwargs,
            )
        else:
            # If no pair bias no point in having a pair representation
            self.update_pair_repr = False

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=kwargs["token_dim"],
                    dim_pair=kwargs["pair_repr_dim"],
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=kwargs["residual_mha"],
                    residual_transition=kwargs["residual_transition"],
                    parallel_mha_transition=kwargs["parallel_mha_transition"],
                    use_attn_pair_bias=kwargs["use_attn_pair_bias"],
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            tri_mult_c=kwargs["tri_mult_c"],
                            use_cueq=kwargs.get("use_cueq", True),
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )
            # For distogram pair prediction
            if self.num_buckets_predict_pair is not None:
                self.pair_head_prediction = torch.nn.Sequential(
                    torch.nn.LayerNorm(kwargs["pair_repr_dim"]),
                    torch.nn.Linear(
                        kwargs["pair_repr_dim"], self.num_buckets_predict_pair
                    ),
                )

        # Contact map prediction head (for contact map diffusion mode)
        if self.contact_map_mode:
            self.contact_map_decoder = torch.nn.Sequential(
                torch.nn.LayerNorm(kwargs["pair_repr_dim"]),
                torch.nn.Linear(kwargs["pair_repr_dim"], 1, bias=False),
            )

        self.coors_3d_decoder = None
        if not self.contact_map_mode:
            self.coors_3d_decoder = torch.nn.Sequential(
                torch.nn.LayerNorm(kwargs["token_dim"]),
                torch.nn.Linear(kwargs["token_dim"], 3, bias=False),
            )
        elif self.predict_coords == "ipa":
            sm_cfg = kwargs.get("structure_module_cfg", {})
            self.structure_module_cfg = {
                "c_s": kwargs["token_dim"],
                "c_z": kwargs["pair_repr_dim"],
                "c_ipa": sm_cfg.get("c_ipa", 16),
                "c_resnet": sm_cfg.get("c_resnet", 128),
                "no_heads_ipa": sm_cfg.get("no_heads_ipa", 12),
                "no_qk_points": sm_cfg.get("no_qk_points", 4),
                "no_v_points": sm_cfg.get("no_v_points", 8),
                "dropout_rate": sm_cfg.get("dropout_rate", 0.1),
                "no_blocks": sm_cfg.get("no_blocks", 8),
                "no_transition_layers": sm_cfg.get("no_transition_layers", 1),
                "no_resnet_blocks": sm_cfg.get("no_resnet_blocks", 2),
                "no_angles": sm_cfg.get("no_angles", 7),
                "trans_scale_factor": sm_cfg.get("trans_scale_factor", 10),
                "epsilon": sm_cfg.get("epsilon", 1e-12),
                "inf": sm_cfg.get("inf", 1e5),
            }
            self.coors_3d_decoder = StructureModule(**self.structure_module_cfg)
            self.register_buffer(
                "restype_atom14_to_atom37",
                torch.from_numpy(rc.RESTYPE_ATOM14_TO_ATOM37).to(dtype=torch.long),
                # torch.tensor(rc.RESTYPE_ATOM14_TO_ATOM37, dtype=torch.long),
            )
            self.register_buffer(
                "restype_atom14_mask",
                torch.from_numpy(rc.RESTYPE_ATOM14_MASK).to(dtype=torch.float32),
                # torch.tensor(rc.RESTYPE_ATOM14_MASK, dtype=torch.float32),
            )
            # Use atom37<-atom14 gather mapping to avoid scatter overwriting N/CA/C with
            # dummy indices (OpenFold uses gather-style conversion for this reason).
            self.register_buffer(
                "restype_atom37_to_atom14",
                torch.from_numpy(rc.RESTYPE_ATOM37_TO_ATOM14).to(dtype=torch.long),
            )
            self.register_buffer(
                "restype_atom37_mask",
                torch.from_numpy(rc.RESTYPE_ATOM37_MASK).to(dtype=torch.float32),
            )
        elif self.predict_coords == "linear":
            self.coors_3d_decoder = torch.nn.Sequential(
                torch.nn.LayerNorm(kwargs["token_dim"]),
                torch.nn.Linear(kwargs["token_dim"], 9, bias=False),
            )

    def _extend_w_registers(self, seqs, pair, mask, cond_seq):
        """
        Extends the sequence representation, pair representation, mask and indices with registers.

        Args:
            - seqs: sequence representation, shape [b, n, dim_token]
            - pair: pair representation, shape [b, n, n, dim_pair]
            - mask: binary mask, shape [b, n]
            - cond_seq: tensor of shape [b, n, dim_cond]

        Returns:
            All elements above extended with registers / zeros.
        """
        if self.num_registers == 0:
            return seqs, pair, mask, cond_seq  # Do nothing

        b, n, _ = seqs.shape
        dim_pair = pair.shape[-1]
        r = self.num_registers
        dim_cond = cond_seq.shape[-1]

        # Concatenate registers to sequence
        reg_expanded = self.registers[None, :, :]  # [1, r, dim_token]
        reg_expanded = reg_expanded.expand(b, -1, -1)  # [b, r, dim_token]
        seqs = torch.cat([reg_expanded, seqs], dim=1)  # [b, r+n, dim_token]

        # Extend mask
        true_tensor = torch.ones(b, r, dtype=torch.bool, device=seqs.device)  # [b, r]
        mask = torch.cat([true_tensor, mask], dim=1)  # [b, r+n]

        # Extend pair representation with zeros; pair has shape [b, n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        # [b, n, n, pair_dim] -> [b, r+n, n, pair_dim]
        zero_pad_top = torch.zeros(
            b, r, n, dim_pair, device=seqs.device
        )  # [b, r, n, dim_pair]
        pair = torch.cat([zero_pad_top, pair], dim=1)  # [b, r+n, n, dim_pair]
        # [b, r+n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        zero_pad_left = torch.zeros(
            b, r + n, r, dim_pair, device=seqs.device
        )  # [b, r+n, r, dim_pair]
        pair = torch.cat([zero_pad_left, pair], dim=2)  # [b, r+n, r+n, dim+pair]

        # Extend cond
        zero_tensor = torch.zeros(
            b, r, dim_cond, device=seqs.device
        )  # [b, r, dim_cond]
        cond_seq = torch.cat([zero_tensor, cond_seq], dim=1)  # [b, r+n, dim_cond]

        return seqs, pair, mask, cond_seq

    def _undo_registers(self, seqs, pair, mask):
        """
        Undoes register padding.

        Args:
            - seqs: sequence representation, shape [b, r+n, dim_token]
            - pair: pair representation, shape [b, r+n, r+n, dim_pair]
            - mask: binary mask, shape [b, r+n]

        Returns:
            All three elements with the register padding removed.
        """
        if self.num_registers == 0:
            return seqs, pair, mask
        r = self.num_registers
        return seqs[:, r:, :], pair[:, r:, r:, :], mask[:, r:]

    def forward(self, batch_nn: Dict[str, torch.Tensor], force_compile: bool = False):
        # TorchDynamo treats `requires_grad` / grad-mode as a compile guard. In Lightning we
        # call the model under both grad-enabled (training) and `torch.no_grad()` (validation,
        # validation sampling, self-conditioning helpers). If we always run the compiled graph,
        # Dynamo will keep recompiling on these mode switches until hitting `recompile_limit`
        # and then fall back to eager.
        #
        # To keep training fast/stable and avoid validation recompiles, only use `torch.compile`
        # in grad-enabled contexts and run eager under `no_grad()`.
        if self.use_torch_compile and (torch.is_grad_enabled() or force_compile):
            if getattr(self, "_forward_compiled", None) is None:
                self._forward_compiled = torch.compile(self._forward_impl)
            return self._forward_compiled(batch_nn)
        return self._forward_impl(batch_nn)

    def _forward_impl(self, batch_nn: Dict[str, torch.Tensor]):
        """
        Runs the network.

        Args:
            batch_nn: dictionary with keys
                - "x_t": tensor of shape [b, n, 3]
                - "t": tensor of shape [b]
                - "mask": binary tensor of shape [b, n]
                - "x_sc" (optional): tensor of shape [b, n, 3]
                - "cath_code" (optional): list of cath codes [b, ?]
                - And potentially others... All in the data batch.

        Returns:
            Predicted clean coordinates, shape [b, n, 3].
        """
        mask = batch_nn["mask"]
        pair_mask = mask[..., None] * mask[..., None, :]  # [b, n, n]

        # Conditioning variables
        c = self.cond_factory(batch_nn)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Prepare input - coordinates and initial sequence representation from features
        if self.contact_map_mode:
            contact_map = batch_nn["contact_map_t"].unsqueeze(-1)  # [b, n, n, 1]
            contact_map_embed = self.linear_contact_embed(contact_map) * pair_mask[..., None] # [b, n, n, pair_repr_dim]
            pair_f_repr = self.pair_repr_builder(batch_nn)  # [b, n, n, pair_repr_dim]
            pair_rep = (contact_map_embed + pair_f_repr) * pair_mask[..., None]  # [b, n, n, pair_repr_dim]
            if torch.isnan(contact_map_embed).any() or torch.isinf(contact_map_embed).any():
                print(
                    f"[pair_init_debug] contact_map_embed nan={torch.isnan(contact_map_embed).any().item()} "
                    f"inf={torch.isinf(contact_map_embed).any().item()} "
                    f"min={contact_map_embed.min().item()} max={contact_map_embed.max().item()}"
                )
            if torch.isnan(pair_f_repr).any() or torch.isinf(pair_f_repr).any():
                print(
                    f"[pair_init_debug] pair_f_repr nan={torch.isnan(pair_f_repr).any().item()} "
                    f"inf={torch.isinf(pair_f_repr).any().item()} "
                    f"min={pair_f_repr.min().item()} max={pair_f_repr.max().item()}"
                )
            if torch.isnan(pair_rep).any() or torch.isinf(pair_rep).any():
                print(
                    f"[pair_init_debug] pair_rep nan={torch.isnan(pair_rep).any().item()} "
                    f"inf={torch.isinf(pair_rep).any().item()} "
                    f"min={pair_rep.min().item()} max={pair_rep.max().item()}"
                )

            # Sequence representation
            seqs = self.init_repr_factory(batch_nn) * mask[..., None]  # [b, n, token_dim]
            if torch.isnan(seqs).any() or torch.isinf(seqs).any():
                print(
                    f"[seqs_init_debug] nan={torch.isnan(seqs).any().item()} "
                    f"inf={torch.isinf(seqs).any().item()} "
                    f"min={seqs.min().item()} max={seqs.max().item()}"
                )
        else:
            coors_3d = batch_nn["x_t"] * mask[..., None]  # [b, n, 3]
            coors_embed = (
                self.linear_3d_embed(coors_3d) * mask[..., None]
            )  # [b, n, token_dim]
            seq_f_repr = self.init_repr_factory(batch_nn)  # [b, n, token_dim]
            seqs = (coors_embed + seq_f_repr) * mask[..., None]  # [b, n, token_dim]

            # Pair representation
            pair_rep = None
            if self.use_attn_pair_bias:
                pair_rep = self.pair_repr_builder(batch_nn)  # [b, n, n, pair_dim]

        # Apply registers
        seqs, pair_rep, mask, c = self._extend_w_registers(seqs, pair_rep, mask, c)
        if torch.isnan(seqs).any() or torch.isinf(seqs).any() or torch.isnan(pair_rep).any() or torch.isinf(pair_rep).any():
            print(
                f"[after_extend_debug] seqs_nan={torch.isnan(seqs).any().item()} "
                f"seqs_inf={torch.isinf(seqs).any().item()} "
                f"seqs_min={seqs.min().item()} seqs_max={seqs.max().item()} "
                f"pair_nan={torch.isnan(pair_rep).any().item()} "
                f"pair_inf={torch.isinf(pair_rep).any().item()} "
                f"pair_min={pair_rep.min().item()} pair_max={pair_rep.max().item()}"
            )

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, pair_dim]
                if torch.isnan(seqs).any() or torch.isinf(seqs).any() or torch.isnan(pair_rep).any() or torch.isinf(pair_rep).any():
                    print(
                        f"[trunk_debug_layer{i}] seqs_nan={torch.isnan(seqs).any().item()} "
                        f"seqs_inf={torch.isinf(seqs).any().item()} "
                        f"seqs_min={seqs.min().item()} seqs_max={seqs.max().item()} "
                        f"pair_nan={torch.isnan(pair_rep).any().item()} "
                        f"pair_inf={torch.isinf(pair_rep).any().item()} "
                        f"pair_min={pair_rep.min().item()} pair_max={pair_rep.max().item()}"
                    )

        # Undo registers
        seqs, pair_rep, mask = self._undo_registers(seqs, pair_rep, mask)
        
        # Symmetrize pair representation
        pair_rep = (pair_rep + pair_rep.transpose(-2, -3)) / 2.0
        if torch.isnan(pair_rep).any() or torch.isinf(pair_rep).any():
            print(
                f"[pair_rep_sym_debug] nan={torch.isnan(pair_rep).any().item()} "
                f"inf={torch.isinf(pair_rep).any().item()} "
                f"min={pair_rep.min().item()} max={pair_rep.max().item()}"
            )

        nn_out = {}
        coords_pred = None
        if not self.contact_map_mode and self.coors_3d_decoder is not None:
            coords_pred = self.coors_3d_decoder(seqs) * mask[..., None]  # [b, n, 3]
            nn_out["coords_pred"] = coords_pred
        elif self.contact_map_mode and self.predict_coords == "ipa":
            # Use unmasked residue types for IPA geometry if provided (masked residue_type
            # is still used for sequence-conditioning embeddings).
            aatype = batch_nn.get("residue_type_unmasked", batch_nn.get("residue_type", None))
            if aatype is None:
                raise ValueError(
                    "residue_type is required in batch for IPA coordinate prediction."
                )
            if torch.isnan(seqs).any() or torch.isinf(seqs).any() or torch.isnan(pair_rep).any() or torch.isinf(pair_rep).any():
                print(
                    f"[ipa_input_debug] seqs_nan={torch.isnan(seqs).any().item()} "
                    f"seqs_inf={torch.isinf(seqs).any().item()} "
                    f"seqs_min={seqs.min().item()} seqs_max={seqs.max().item()} "
                    f"pair_nan={torch.isnan(pair_rep).any().item()} "
                    f"pair_inf={torch.isinf(pair_rep).any().item()} "
                    f"pair_min={pair_rep.min().item()} pair_max={pair_rep.max().item()}"
                )
            struct_out = self.coors_3d_decoder(
                {"single": seqs, "pair": pair_rep},
                aatype=aatype,
                mask=mask,
                inplace_safe=False,
                _offload_inference=False,
            )
            # Predicted backbone frames from the StructureModule (OpenFold-style).
            frames7 = (
                struct_out["frames"][-1]
                if struct_out["frames"].dim() == 4
                else struct_out["frames"]
            )  # [b, n, 7]
            atom14 = struct_out["positions"][-1] if struct_out["positions"].dim() == 5 else struct_out["positions"]
            if torch.isnan(atom14).any() or torch.isinf(atom14).any():
                print(
                    f"[ipa_pos_debug] nan={torch.isnan(atom14).any().item()} "
                    f"inf={torch.isinf(atom14).any().item()} "
                    f"min={atom14.min().item()} max={atom14.max().item()}"
                )
            atom14 = atom14 * mask[..., None, None]
            # Convert atom14 -> atom37 safely via gather (scatter is unsafe because
            # rc.RESTYPE_ATOM14_TO_ATOM37 uses 0 as a dummy index for missing atoms,
            # which would overwrite the true N atom at index 0).
            aatype_safe = torch.clamp(
                aatype,
                min=0,
                max=self.restype_atom37_to_atom14.shape[0] - 1,
            )
            residx_atom37_to_atom14 = self.restype_atom37_to_atom14[aatype_safe]  # [b, n, 37]
            atom37 = atom14.gather(
                2, residx_atom37_to_atom14[..., None].expand(-1, -1, -1, 3)
            )  # [b, n, 37, 3]
            atom37_mask = self.restype_atom37_mask[aatype_safe] * mask[..., None]  # [b, n, 37]
            atom37 = atom37 * atom37_mask[..., None]
            # Convert Å -> nm to match the rest of the training pipeline.
            coords_pred = atom37 / 10.0
            nn_out["coords_pred"] = coords_pred
            # Convert frame translations Å -> nm (quaternions unchanged).
            frame_scale = frames7.new_tensor([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
            nn_out["frames_pred"] = frames7 * frame_scale
        elif self.contact_map_mode and self.predict_coords == "linear":
            raw = self.coors_3d_decoder(seqs)  # [b, n, 9]
            coords_pred = raw.view(raw.shape[0], raw.shape[1], 3, 3)
            coords_pred = coords_pred * mask[..., None, None]
            nn_out["coords_pred"] = coords_pred

        if self.update_pair_repr and self.num_buckets_predict_pair is not None:
            pair_logits = self.pair_head_prediction(pair_rep)
            if torch.isnan(pair_logits).any() or torch.isinf(pair_logits).any():
                print(
                    f"[pair_logits_debug] nan={torch.isnan(pair_logits).any().item()} "
                    f"inf={torch.isinf(pair_logits).any().item()} "
                    f"pair_rep_nan={torch.isnan(pair_rep).any().item()} "
                    f"pair_rep_inf={torch.isinf(pair_rep).any().item()} "
                    f"pair_rep_min={pair_rep.min().item()} pair_rep_max={pair_rep.max().item()}"
                )
            if coords_pred is not None:
                coords_pred = coords_pred + torch.mean(pair_logits) * 0.0
                coords_pred = coords_pred * (mask[..., None] if coords_pred.dim() == 3 else mask[..., None, None])
            nn_out["pair_logits"] = pair_logits

        # Contact map prediction (for contact map diffusion mode)
        if self.contact_map_mode:
            contact_map_logits = self.contact_map_decoder(pair_rep)  # [b, n, n, 1]
            contact_map_logits = contact_map_logits.squeeze(-1)  # [b, n, n]
            contact_map_logits = contact_map_logits * pair_mask
            nn_out["contact_map_logits"] = contact_map_logits
            if self.non_contact_value == -1:
                nn_out["contact_map_pred"] = torch.tanh(contact_map_logits)
            else:
                nn_out["contact_map_pred"] = torch.sigmoid(contact_map_logits)

        return nn_out
