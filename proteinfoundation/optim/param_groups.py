# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Utility for partitioning model parameters into Muon vs Adam/AdamW groups.

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger


# ── 1. Name-based exclusion patterns ─────────────────────────────────────
# Parameters whose *name* matches any of these patterns are always routed
# to AdamW, regardless of their tensor dimensionality.
#
# Rationale (from the research document):
#   - Embeddings: sparse one-hot lookups, orthogonalization destroys token
#     localisation.
#   - Output heads: empirically better with scalar-adaptive optimizers.
#   - Registers: learnable register tokens (1-D or small).
#
# NOTE: LayerNorms, RMSNorms, and Bias vectors are NOT listed here because
# they are intrinsically handled by the `param.ndim < 2` check below.
# As the Muon migration guide specifies, 1D tensors cannot be subjected to
# SVD/orthogonalization and must be explicitly routed to AdamW.

_ADAM_NAME_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in (
        r"embed",          # any embedding layer
        r"residue_type",   # amino-acid residue type embeddings
        r"coors_3d_decoder",    # coordinate output head
        r"contact_map_decoder", # contact-map output head
        r"pair_head_prediction",  # distogram head
        r"dssp_head",      # secondary-structure output head
        r"registers",      # learnable register tokens
    )
)


def _is_adam_param_by_name(name: str) -> bool:
    """Return ``True`` if the parameter *name* matches an AdamW-only pattern."""
    return any(pat.search(name) for pat in _ADAM_NAME_PATTERNS)


# ── Public API ───────────────────────────────────────────────────────────

def build_optimizer_param_groups(
    model: torch.nn.Module,
    opt_cfg: Any,
    *,
    trainable_only: bool = True,
) -> List[Dict]:
    """Partition *model*'s parameters into Muon and AdamW groups.

    **Routing rules** (applied in order):

    1. ``requires_grad is False`` → skipped entirely.
    2. Name matches an exclusion pattern (embeddings, output heads) → **AdamW**.
    3. ``param.ndim < 2`` (biases, LayerNorm scale/shift) → **AdamW**.
    4. Everything else (≥ 2-D hidden weight matrices) → **Muon**.

    Args:
        model: The ``nn.Module`` (typically ``ModelTrainerBase`` or
            ``Proteina`` instance).
        opt_cfg: The ``cfg_exp.opt`` OmegaConf node.  Expected keys
            (all optional with sensible defaults):

            - ``lr``: base learning rate (used for AdamW group; also for
              Muon when ``adjust_lr_fn="match_rms_adamw"``).
            - ``muon_lr``: explicit Muon LR override (default: use ``lr``).
            - ``muon_momentum``: Muon momentum beta (default 0.95).
            - ``muon_weight_decay``: weight decay for Muon params (default 0.0).
            - ``muon_nesterov``: Nesterov momentum (default True).
            - ``muon_ns_steps``: Newton-Schulz iterations (default 5).
            - ``muon_adjust_lr_fn``: ``"original"`` or ``"match_rms_adamw"``
              (default ``"match_rms_adamw"``).
            - ``weight_decay``: AdamW weight decay (default 0.0).
            - ``adam_betas``: AdamW betas (default ``[0.9, 0.999]``).
            - ``adam_eps``: AdamW epsilon (default ``1e-8``).
        trainable_only: If ``True`` (default), skip params with
            ``requires_grad=False``.

    Returns:
        A list of two param-group dicts suitable for
        :class:`~proteinfoundation.optim.muon.HybridMuonAdamW`.
    """
    # --- Read config (robust to OmegaConf DictConfig) ---
    _get = getattr(opt_cfg, "get", lambda k, d=None: getattr(opt_cfg, k, d))

    base_lr = float(_get("lr", 1e-4))

    # Muon settings
    muon_lr = _get("muon_lr", None)
    if muon_lr is not None:
        muon_lr = float(muon_lr)
    else:
        muon_lr = base_lr  # shared LR (RMS matching handles scale)

    muon_momentum = float(_get("muon_momentum", 0.95))
    muon_weight_decay = float(_get("muon_weight_decay", 0.0))
    muon_nesterov = bool(_get("muon_nesterov", True))
    muon_ns_steps = int(_get("muon_ns_steps", 5))
    muon_adjust_lr_fn = str(_get("muon_adjust_lr_fn", "match_rms_adamw"))

    # AdamW settings
    adam_wd = float(_get("weight_decay", 0.0))
    adam_betas_raw = _get("adam_betas", [0.9, 0.999])
    adam_betas = tuple(float(b) for b in adam_betas_raw)
    adam_eps = float(_get("adam_eps", 1e-8))

    # --- Partition parameters ---
    muon_params: List[torch.nn.Parameter] = []
    adam_params: List[torch.nn.Parameter] = []

    muon_names: List[str] = []
    adam_names: List[str] = []

    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue

        if _is_adam_param_by_name(name):
            adam_params.append(param)
            adam_names.append(name)
        elif param.ndim < 2:
            adam_params.append(param)
            adam_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    # --- Logging ---
    total = len(muon_params) + len(adam_params)
    muon_numel = sum(p.numel() for p in muon_params)
    adam_numel = sum(p.numel() for p in adam_params)
    total_numel = muon_numel + adam_numel

    logger.info(
        f"Muon parameter partitioning: "
        f"{len(muon_params)}/{total} params → Muon "
        f"({muon_numel:,} elements, {muon_numel/total_numel*100:.1f}%), "
        f"{len(adam_params)}/{total} params → AdamW "
        f"({adam_numel:,} elements, {adam_numel/total_numel*100:.1f}%)"
    )
    logger.debug(f"Muon params: {muon_names}")
    logger.debug(f"AdamW params: {adam_names}")

    # --- Build groups ---
    groups = []

    if muon_params:
        groups.append(
            dict(
                params=muon_params,
                lr=muon_lr,
                use_muon=True,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr_fn,
            )
        )

    if adam_params:
        groups.append(
            dict(
                params=adam_params,
                lr=base_lr,
                use_muon=False,
                betas=adam_betas,
                eps=adam_eps,
                weight_decay=adam_wd,
            )
        )

    return groups
