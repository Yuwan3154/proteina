# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Muon optimizer implementation for proteina.
#
# This implementation is based on:
#   - PyTorch 2.9 torch.optim.Muon API specification
#     https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html
#   - Keller Jordan's reference implementation
#     https://github.com/KellerJordan/Muon
#   - "Muon is Scalable for LLM Training" (Moonshot AI)
#     https://arxiv.org/pdf/2502.16982
#
# It provides a drop-in Muon optimizer that can be used standalone for >=2D
# hidden layer parameters.  For a complete training setup, pair it with a
# standard AdamW for embeddings, biases, and normalization parameters — see
# ``build_optimizer_param_groups`` in ``param_groups.py`` and the
# ``HybridMuonAdamW`` convenience wrapper in this module.


from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

def _newton_schulz_orthogonalize(
    G: Tensor,
    ns_steps: int = 5,
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> Tensor:
    """Approximate the zeroth power (orthogonalization) of matrix *G* using
    a quintic Newton-Schulz iteration.

    The iteration replaces the singular values of G with ~1 while
    preserving the singular vectors, effectively producing the nearest
    orthogonal matrix.  All computation is done in **bfloat16** for
    efficiency (Tensor-Core friendly), and the result is cast back to
    G's original dtype.

    Args:
        G: 2-D gradient / momentum matrix of shape ``(m, n)``.
        ns_steps: Number of Newton-Schulz iterations (default 5).
        ns_coefficients: ``(a, b, c)`` polynomial coefficients.
        eps: Small constant for numerical stability in the Frobenius norm.

    Returns:
        Orthogonalized matrix of the same shape and dtype as *G*.
    """
    assert G.ndim >= 2, f"Newton-Schulz requires >= 2D tensor, got {G.ndim}D"
    original_dtype = G.dtype
    a, b, c = ns_coefficients

    X = G.bfloat16()

    # Work with the "wider" orientation to keep NS stable
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Normalise so spectral norm <= 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    # Quintic NS iterations
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(original_dtype)


# ---------------------------------------------------------------------------
# Learning-rate adjustment helpers (following PyTorch 2.9 API)
# ---------------------------------------------------------------------------

def _adjust_lr_original(lr: float, shape: Tuple[int, ...]) -> float:
    """Keller Jordan's original scaling: ``lr * sqrt(max(1, m/n))``."""
    m, n = shape[-2], shape[-1]
    return lr * math.sqrt(max(1, m / n))


def _adjust_lr_match_rms_adamw(lr: float, shape: Tuple[int, ...]) -> float:
    """Moonshot scaling for RMS matching with AdamW:
    ``0.2 * lr * sqrt(max(m, n))``

    With this adjustment one can directly reuse the learning rate and weight
    decay tuned for AdamW.
    """
    m, n = shape[-2], shape[-1]
    return 0.2 * lr * math.sqrt(max(m, n))


_ADJUST_LR_FNS = {
    "original": _adjust_lr_original,
    "match_rms_adamw": _adjust_lr_match_rms_adamw,
}


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(Optimizer):
    r"""Muon — MomentUm Orthogonalized by Newton-schulz.

    Implements the Muon algorithm following the **PyTorch 2.9** API surface
    (``torch.optim.Muon``).  This is a *local* re-implementation so that
    the proteina codebase does not require PyTorch >= 2.9.

    Muon is designed for **≥ 2-D hidden-layer weight matrices**.  For
    embeddings, biases, normalization gains/shifts, and output heads, use a
    standard optimizer such as AdamW.

    Key features:
        * 5-step quintic Newton-Schulz orthogonalization in bfloat16.
        * Optional Nesterov momentum.
        * Two LR adjustment modes: ``"original"`` (raw Keller scaling) and
          ``"match_rms_adamw"`` (Moonshot scaling so you can share the same
          LR as AdamW).
        * Decoupled weight decay.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default ``1e-3``).
        momentum: Momentum factor (default ``0.95``).
        weight_decay: Decoupled weight decay (default ``0.0``).
        nesterov: If ``True``, use Nesterov momentum (default ``True``).
        ns_steps: Number of Newton-Schulz iterations (default ``5``).
        ns_coefficients: ``(a, b, c)`` polynomial coefficients for NS
            iteration (default ``(3.4445, -4.775, 2.0315)``).
        eps: Numerical stability term for NS norm (default ``1e-7``).
        adjust_lr_fn: Learning-rate adjustment strategy.
            * ``"original"`` — Keller's ``sqrt(max(1, m/n))`` scaling.
              Use a separate (typically larger) LR for Muon.
            * ``"match_rms_adamw"`` — Moonshot's ``0.2 * sqrt(max(m,n))``
              scaling.  Allows sharing the same LR as AdamW.
            Default is ``"original"``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        adjust_lr_fn: Optional[str] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        if adjust_lr_fn is None:
            adjust_lr_fn = "original"
        if adjust_lr_fn not in _ADJUST_LR_FNS:
            raise ValueError(
                f"Invalid adjust_lr_fn: {adjust_lr_fn!r}. "
                f"Must be one of {list(_ADJUST_LR_FNS.keys())}"
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Muon optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum_beta = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            ns_coefficients = tuple(group["ns_coefficients"])
            eps = group["eps"]
            adjust_fn = _ADJUST_LR_FNS[group["adjust_lr_fn"]]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                # Lazy state init
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)

                state["step"] += 1
                buf = state["momentum_buffer"]

                # Momentum update
                buf.lerp_(grad, 1 - momentum_beta)

                # Nesterov look-ahead
                if nesterov:
                    update = grad.lerp(buf, momentum_beta)
                else:
                    update = buf.clone()

                # Flatten ≥3-D tensors to 2-D for Newton-Schulz
                orig_shape = update.shape
                if update.ndim == 1:
                    # 1-D params shouldn't be in Muon, but handle gracefully
                    # by skipping orthogonalization
                    pass
                elif update.ndim >= 3:
                    update = update.reshape(update.shape[0], -1)
                    update = _newton_schulz_orthogonalize(
                        update, ns_steps, ns_coefficients, eps
                    )
                    update = update.reshape(orig_shape)
                else:
                    update = _newton_schulz_orthogonalize(
                        update, ns_steps, ns_coefficients, eps
                    )

                # LR adjustment based on matrix shape
                adjusted_lr = adjust_fn(lr, orig_shape)

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - adjusted_lr * weight_decay)

                # Parameter update
                p.add_(update, alpha=-adjusted_lr)

        return loss


# ---------------------------------------------------------------------------
# HybridMuonAdamW — single optimizer for the full model
# ---------------------------------------------------------------------------

class HybridMuonAdamW(Optimizer):
    """A single ``torch.optim.Optimizer`` that wraps Muon for ≥ 2-D hidden
    parameters and AdamW (or Adam) for everything else.

    This avoids the complexity of managing two separate optimizers and
    interacts cleanly with PyTorch Lightning's ``configure_optimizers()``
    and LR scheduler infrastructure.

    Parameter groups must contain a ``"use_muon"`` boolean key to indicate
    which sub-optimizer should handle them.

    Example usage (see also ``build_optimizer_param_groups``)::

        muon_group = dict(
            params=hidden_2d_params,
            lr=1e-4,
            use_muon=True,
            momentum=0.95,
            weight_decay=0.0,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        adam_group = dict(
            params=other_params,
            lr=1e-4,
            use_muon=False,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        optimizer = HybridMuonAdamW([muon_group, adam_group])
    """

    def __init__(
        self,
        param_groups: List[dict],
        *,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps_ns: float = 1e-7,
    ):
        # Validate and fill defaults per group
        processed_groups = []
        for group in param_groups:
            g = dict(group)  # shallow copy
            if "use_muon" not in g:
                raise ValueError(
                    "Each param group must contain a 'use_muon' boolean key."
                )
            if g["use_muon"]:
                g.setdefault("lr", 1e-3)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("nesterov", True)
                g.setdefault("ns_steps", 5)
                g.setdefault("adjust_lr_fn", "original")
                g["ns_coefficients"] = ns_coefficients
                g["eps_ns"] = eps_ns
                # Dummy values so Optimizer infra doesn't complain
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-8)
            else:
                g.setdefault("lr", 1e-4)
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.0)
                # Defaults unused by Adam but needed for uniform group schema
                g.setdefault("momentum", 0.95)
                g.setdefault("nesterov", True)
                g.setdefault("ns_steps", 5)
                g.setdefault("adjust_lr_fn", "original")
                g["ns_coefficients"] = ns_coefficients
                g["eps_ns"] = eps_ns
            processed_groups.append(g)

        # Use a blank defaults dict — each group carries its own config
        super().__init__(processed_groups, {})

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)

        return loss

    # -----------------------------------------------------------------
    # Muon sub-step
    # -----------------------------------------------------------------
    def _step_muon(self, group: dict):
        lr = group["lr"]
        momentum_beta = group["momentum"]
        weight_decay = group["weight_decay"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        ns_coefficients = tuple(group["ns_coefficients"])
        eps_ns = group["eps_ns"]
        adjust_fn = _ADJUST_LR_FNS[group["adjust_lr_fn"]]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p)

            state["step"] += 1
            buf = state["momentum_buffer"]

            # Momentum
            buf.lerp_(grad, 1 - momentum_beta)

            # Nesterov
            if nesterov:
                update = grad.lerp(buf, momentum_beta)
            else:
                update = buf.clone()

            # Newton-Schulz orthogonalization
            orig_shape = update.shape
            if update.ndim == 1:
                pass  # skip NS for 1-D (shouldn't happen)
            elif update.ndim >= 3:
                update = update.reshape(update.shape[0], -1)
                update = _newton_schulz_orthogonalize(
                    update, ns_steps, ns_coefficients, eps_ns
                )
                update = update.reshape(orig_shape)
            else:
                update = _newton_schulz_orthogonalize(
                    update, ns_steps, ns_coefficients, eps_ns
                )

            adjusted_lr = adjust_fn(lr, orig_shape)

            # Decoupled weight decay
            if weight_decay != 0:
                p.mul_(1 - adjusted_lr * weight_decay)

            p.add_(update, alpha=-adjusted_lr)

    # -----------------------------------------------------------------
    # AdamW sub-step
    # -----------------------------------------------------------------
    def _step_adamw(self, group: dict):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    "HybridMuonAdamW AdamW path does not support sparse grads"
                )

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Bias-corrected AdamW
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # Decoupled weight decay
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            p.addcdiv_(exp_avg, denom, value=-step_size)
