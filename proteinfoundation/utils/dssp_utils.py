# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
DSSP secondary structure utilities for auxiliary loss.
Computes 3-state DSSP targets (loop, helix, strand) from backbone coordinates.
"""

from typing import Optional

import torch


def compute_dssp_target(
    coords: torch.Tensor,
    mask: torch.Tensor,
    coord_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Compute DSSP 3-state secondary structure targets from coordinates.

    Args:
        coords: [b, n, atoms, 3] in Angstrom. Expects atom37 or backbone format
                with N(0), CA(1), C(2), O(3).
        mask: [b, n] boolean mask; True = valid residue.
        coord_mask: Optional [b, n, atoms] per-atom mask. If provided, residues
                    with missing N/CA/C/O are excluded from DSSP.

    Returns:
        dssp_target: [b, n] long tensor with values 0=loop, 1=helix, 2=strand.
                     Invalid/masked positions are set to -1 (ignore_index for CE).
        None if coords has fewer than 4 atoms (e.g. CA-only data).
    """
    if coords.shape[2] < 4:
        return None

    import pydssp

    # Extract N, CA, C, O (indices 0, 1, 2, 3)
    ncao = coords[:, :, [0, 1, 2, 3], :]  # [b, n, 4, 3]

    # Refine mask: exclude residues with missing backbone atoms
    if coord_mask is not None and coord_mask.shape[2] >= 4:
        ncao_valid = coord_mask[:, :, 0] & coord_mask[:, :, 1] & coord_mask[:, :, 2] & coord_mask[:, :, 3]
        valid_mask = mask & ncao_valid
    else:
        valid_mask = mask

    # PyDSSP expects [batch, L, 4, 3]; coords in Angstrom
    dssp_out = pydssp.assign(ncao, out_type="index")  # [b, n]

    # Set invalid positions to -1 for cross_entropy ignore_index
    dssp_target = dssp_out.long().clone()
    dssp_target = torch.where(valid_mask, dssp_target, torch.full_like(dssp_target, -1))

    return dssp_target
