# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


VALID_ATTN_IMPLS = ("vanilla", "sdpa", "flash", "cuequivariance")


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py

    Supports multiple attention backends via the ``attn_impl`` parameter:

    - ``"vanilla"``: hand-coded einsum + masked_fill + softmax (default, original)
    - ``"sdpa"``: ``torch.nn.functional.scaled_dot_product_attention`` with pair bias
      fused into the additive ``attn_mask``; compile-friendly
    - ``"flash"``: ``flash_attn.flash_attn_func``; requires fp16/bf16, **no pair bias**
      (raises ``ValueError`` at construction if ``pair_dim`` is set); not compatible with
      ``torch.compile`` (falls back to vanilla when compiling)
    - ``"cuequivariance"``: ``cuequivariance_ops_torch.attention_pair_bias`` Triton fused
      kernel that fuses pair LayerNorm + pair bias projection + SDPA + gating + output
      projection; requires ``pair_dim`` set and ``dim_head % 32 == 0``

    Eager-path backend override (``attn_impl_eager``):

    When set to a non-None value, this overrides the backend used in *eager* (non-
    ``torch.compile``) forward passes — typically validation and self-conditioning,
    which both run under ``torch.no_grad()`` outside the compiled training graph.
    The compiled training path continues to use ``attn_impl`` because
    ``torch.compiler.is_compiling()`` is True during Dynamo tracing and the
    ``_get_impl()`` dispatcher short-circuits before reading ``attn_impl_eager``.
    Dynamo therefore never sees ``attn_impl_eager`` and creates no guard for it,
    so changing it post-construction does not trigger recompilation.

    Typical use: ``attn_impl="vanilla"`` + ``attn_impl_eager="cuequivariance"`` to
    get Inductor's best compile fusions for training while running the cuEq fused
    Triton kernel during all eager (val + SC) forward passes.
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        attn_impl: str = "vanilla",
        attn_impl_eager: Optional[str] = None,
        **kawrgs  # noqa
    ):
        super().__init__()
        if attn_impl not in VALID_ATTN_IMPLS:
            raise ValueError(
                f"attn_impl must be one of {VALID_ATTN_IMPLS}, got {attn_impl!r}"
            )
        if attn_impl_eager is not None and attn_impl_eager not in VALID_ATTN_IMPLS:
            raise ValueError(
                f"attn_impl_eager must be None or one of {VALID_ATTN_IMPLS}, "
                f"got {attn_impl_eager!r}"
            )

        # Constraint checks apply to both attn_impl and attn_impl_eager because
        # either may end up dispatching to the corresponding backend at runtime.
        impls_to_check = {attn_impl}
        if attn_impl_eager is not None:
            impls_to_check.add(attn_impl_eager)

        if "flash" in impls_to_check and exists(pair_dim):
            raise ValueError(
                "attn_impl='flash' (or attn_impl_eager='flash') cannot be used with "
                "pair_dim set — flash_attn 2.x has no pair bias support. "
                "Use 'sdpa' for pair-biased attention."
            )
        if "cuequivariance" in impls_to_check:
            try:
                import cuequivariance_ops_torch  # noqa
            except ImportError:
                raise ImportError(
                    "attn_impl='cuequivariance' (or attn_impl_eager='cuequivariance') "
                    "requires cuequivariance_ops_torch. Install it or choose a "
                    "different backend."
                )
            if not exists(pair_dim):
                raise ValueError(
                    "attn_impl='cuequivariance' (or attn_impl_eager='cuequivariance') "
                    "requires pair_dim to be set."
                )
            if dim_head % 32 != 0:
                raise ValueError(
                    f"attn_impl='cuequivariance' (or attn_impl_eager='cuequivariance') "
                    f"requires dim_head % 32 == 0, got dim_head={dim_head}. The Triton "
                    f"kernel falls back to PyTorch reference when this is violated."
                )

        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.attn_impl = attn_impl
        # Eager-path override. Stored as a plain Python attribute; intentionally
        # not registered as a buffer/parameter and never read inside a compiled
        # branch (see _get_impl below) so Dynamo creates no guard on it.
        self.attn_impl_eager = attn_impl_eager
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def _get_impl(self) -> str:
        """Resolve which attention backend to dispatch to this call.

        Under ``torch.compile``, ``torch.compiler.is_compiling()`` evaluates to
        ``True`` and Dynamo specializes on the truthy branch (``return
        self.attn_impl``). Dynamo therefore never observes ``self.attn_impl_eager``
        and creates no guard for it. Outside compile, when an override is set,
        the eager backend is returned.

        This lets a single module instance use one backend for the compiled
        training path (``attn_impl``) and a different backend for all eager paths
        — validation, self-conditioning, etc. — without invalidating the compile
        cache when the override is set or cleared.
        """
        if not torch.compiler.is_compiling() and self.attn_impl_eager is not None:
            return self.attn_impl_eager
        return self.attn_impl

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies, shape (b,n,n)
        :return: updated node features of shape (b,n,d_s)
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_normed = self.node_norm(node_feats)
        impl = self._get_impl()

        if impl == "cuequivariance":
            # The cuEq kernel fuses pair_norm internally via w_ln_z/b_ln_z weights.
            # Applying pair_norm here would double-normalize — skip it for this path.
            return self._forward_cueq(node_normed, pair_feats, mask)

        # For vanilla / sdpa / flash: apply pair_norm explicitly before projection.
        pair_feats_n = self.pair_norm(pair_feats) if exists(pair_feats) else None
        h = self.heads
        q, k, v = self.to_qkv(node_normed).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_normed)
        b = (
            self.to_bias(pair_feats_n).permute(0, 3, 1, 2)
            if exists(pair_feats_n)
            else 0
        )
        q, k, v, g = map(
            lambda t: t.unflatten(-1, (h, t.shape[-1] // h)).permute(0, 2, 1, 3),
            (q, k, v, g),
        )
        attn_feats = self._attn(q, k, v, b, mask, impl)
        attn_feats = (torch.sigmoid(g) * attn_feats).permute(0, 2, 1, 3).flatten(2)
        return self.to_out_node(attn_feats)

    # ------------------------------------------------------------------
    # Attention kernel dispatch
    # ------------------------------------------------------------------

    def _attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        b,
        mask: Optional[Tensor],
        impl: Optional[str] = None,
    ) -> Tensor:
        # ``impl`` is passed from forward() to avoid re-resolving the dispatcher;
        # default to a fresh resolve if a caller invokes _attn() directly.
        if impl is None:
            impl = self._get_impl()
        if impl == "sdpa":
            return self._attn_sdpa(q, k, v, b, mask)
        if impl == "flash":
            return self._attn_flash(q, k, v, b, mask)
        return self._attn_vanilla(q, k, v, b, mask)

    def _attn_vanilla(self, q: Tensor, k: Tensor, v: Tensor, b, mask: Optional[Tensor]) -> Tensor:
        """Original einsum + masked_fill + softmax implementation."""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            sim = sim.masked_fill(~mask.unsqueeze(1), max_neg_value(sim))
        attn = torch.softmax((sim + b).contiguous(), dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)

    def _attn_sdpa(self, q: Tensor, k: Tensor, v: Tensor, b, mask: Optional[Tensor]) -> Tensor:
        """SDPA backend: fuses pair bias and padding mask into a single float attn_mask.

        Float additive masks (rather than boolean) avoid shape guards under torch.compile
        and sidestep NaN in fully-masked rows (dtype.min is finite; -inf is not).
        """
        has_bias = isinstance(b, Tensor)
        if mask is not None:
            # Convert bool pair mask (B,N,N) to float additive (B,1,N,N).
            float_mask = torch.zeros(mask.shape, dtype=q.dtype, device=q.device)
            float_mask = float_mask.masked_fill_(~mask, torch.finfo(q.dtype).min)
            float_mask = float_mask.unsqueeze(1)  # broadcast over H
            attn_mask = (b + float_mask) if has_bias else float_mask
        else:
            attn_mask = b if has_bias else None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

    def _attn_flash(self, q: Tensor, k: Tensor, v: Tensor, b, mask: Optional[Tensor]) -> Tensor:
        """Flash Attention backend (flash_attn 2.x).

        Limitations:
        - No pair bias (construction-time check prevents b from being a Tensor here).
        - key-padding mask is not forwarded: flash_attn_func has no mask parameter.
          For benchmarking with all-valid tokens this is correct. For production use
          with variable-length sequences, switch to flash_attn_varlen_func.
        - Not compatible with torch.compile; falls back to vanilla when compiling.
        - Only fp16 and bf16 are supported by flash_attn 2.x. fp32 inputs are silently
          cast to bf16 and the result is cast back; expect ~1e-3 vs a pure fp32 baseline.
        """
        if torch.compiler.is_compiling():
            return self._attn_vanilla(q, k, v, b, mask)
        from flash_attn import flash_attn_func

        # flash_attn requires fp16/bf16; auto-cast fp32 inputs
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        # flash_attn expects (B, N, H, D); our tensors are (B, H, N, D)
        q_fa = q.permute(0, 2, 1, 3).contiguous()
        k_fa = k.permute(0, 2, 1, 3).contiguous()
        v_fa = v.permute(0, 2, 1, 3).contiguous()
        out = flash_attn_func(q_fa, k_fa, v_fa, softmax_scale=self.scale, causal=False)
        out = out.permute(0, 2, 1, 3)  # back to (B, H, N, D)

        if orig_dtype == torch.float32:
            out = out.to(torch.float32)
        return out

    # ------------------------------------------------------------------
    # cuEquivariance fused path (replaces forward entirely)
    # ------------------------------------------------------------------

    def _forward_cueq(
        self,
        node_normed: Tensor,          # (B, N, node_dim) — node_norm already applied
        pair_feats: Optional[Tensor], # (B, N, N, pair_dim) — RAW, pair_norm NOT applied
        mask: Optional[Tensor],       # (B, N, N) bool pair mask, or None
    ) -> Tensor:
        """cuEquivariance fused attention kernel.

        Fuses: pair LayerNorm + pair bias projection + SDPA + gating + output projection
        into a single Triton kernel (for N > 100 and dim_head % 32 == 0).

        The gating input ``s`` is ``node_normed`` (same as the input to ``to_g`` in the
        vanilla path). The kernel computes sigmoid(s @ w_proj_g.T) * attn_out internally,
        matching the vanilla gating exactly.

        Mask convention: cuEq expects a 1D key mask (B, N) with 1=valid, 0=masked.
        We recover it from the 2D pair mask via ``mask.any(dim=-2)``. This is exact when
        the pair mask is a separable outer product of a 1D key mask, which is always the
        case in the current codebase (MultiHeadBiasedAttentionADALN_MM builds it that way).
        """
        from cuequivariance_ops_torch import attention_pair_bias

        h = self.heads
        # Project Q/K/V and apply optional QK layer norms before the kernel.
        q, k, v = self.to_qkv(node_normed).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        q, k, v = map(
            lambda t: t.unflatten(-1, (h, t.shape[-1] // h)).permute(0, 2, 1, 3),
            (q, k, v),
        )

        # Recover 1D float key mask from 2D bool pair mask.
        if mask is not None:
            key_mask = mask.any(dim=-2).float()   # (B, N)
        else:
            B, _, N, _ = q.shape
            key_mask = torch.ones(B, N, dtype=q.dtype, device=q.device)

        out, _ = attention_pair_bias(
            s=node_normed,               # (B, N, node_dim)
            q=q, k=k, v=v,               # (B, H, N, DH)
            z=pair_feats,                # (B, N, N, pair_dim) — fused pair_norm via w_ln_z
            mask=key_mask,               # (B, N) float
            num_heads=h,
            w_proj_z=self.to_bias.weight,        # (heads, pair_dim)
            w_proj_g=self.to_g.weight,           # (inner_dim, node_dim)
            w_proj_o=self.to_out_node.weight,    # (dim_out, inner_dim)
            w_ln_z=self.pair_norm.weight,        # (pair_dim,)
            b_ln_z=self.pair_norm.bias,          # (pair_dim,)
            b_proj_z=None,                       # to_bias has bias=False
            b_proj_g=self.to_g.bias,             # (inner_dim,)
            b_proj_o=self.to_out_node.bias,      # (dim_out,)
            attn_scale=self.scale,
            return_z_proj=False,
        )
        return out   # (B, N, dim_out) — gating + output proj fused inside the kernel


# ----------------------------------------------------------------------
# Module-level utility
# ----------------------------------------------------------------------


def set_attn_impl_eager(module: nn.Module, attn_impl_eager: Optional[str]) -> int:
    """Set ``attn_impl_eager`` on every :class:`PairBiasAttention` submodule.

    Use this after loading a checkpoint (or anywhere the module tree was built
    without an explicit eager override) to swap the eager-path backend in place.
    Does not trigger Dynamo recompilation — the attribute is invisible to compiled
    branches by construction (see :meth:`PairBiasAttention._get_impl`).

    Args:
        module: Root module to walk. Any nested PairBiasAttention modules are mutated.
        attn_impl_eager: New eager backend (one of VALID_ATTN_IMPLS), or ``None``
            to clear the override and fall back to ``attn_impl``.

    Returns:
        Number of PairBiasAttention modules that were updated.
    """
    if attn_impl_eager is not None and attn_impl_eager not in VALID_ATTN_IMPLS:
        raise ValueError(
            f"attn_impl_eager must be None or one of {VALID_ATTN_IMPLS}, "
            f"got {attn_impl_eager!r}"
        )
    n = 0
    for m in module.modules():
        if isinstance(m, PairBiasAttention):
            m.attn_impl_eager = attn_impl_eager
            n += 1
    return n
