# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
DSSP secondary structure utilities for auxiliary loss.
Computes 3-state DSSP targets (loop, helix, strand) from backbone coordinates.
"""

from typing import Literal, Optional

import torch


def compute_dssp_target(
    coords: torch.Tensor,
    mask: torch.Tensor,
    coord_mask: Optional[torch.Tensor] = None,
    coord_layout: Literal["atom37", "pdb"] = "atom37",
) -> Optional[torch.Tensor]:
    """
    Compute DSSP 3-state secondary structure targets from coordinates.

    The atom indices used for N/CA/C/O depend on the coord layout:
        - "atom37"  (OpenFold ordering, default — matches the live data
          pipeline after pdb_data.py:695): N=0, CA=1, C=2, CB=3, O=4.
          Extracts atoms [0, 1, 2, 4].
        - "pdb"     (raw PDB ordering, before the OpenFold reorder):
          N=0, CA=1, C=2, O=3. Extracts atoms [0, 1, 2, 3].

    Picking the wrong layout silently feeds CB to pydssp in place of O and
    produces ~all-loop labels.

    Args:
        coords: [b, n, atoms, 3] in Angstrom.
        mask: [b, n] boolean mask; True = valid residue.
        coord_mask: Optional [b, n, atoms] per-atom mask. If provided, residues
                    with missing N/CA/C/O are excluded from DSSP.
        coord_layout: "atom37" (default) or "pdb". See above.

    Returns:
        dssp_target: [b, n] long tensor with values 0=loop, 1=helix, 2=strand.
                     Invalid/masked positions are set to -1 (ignore_index for CE).
        None if coords lack enough atoms for the requested layout (e.g. CA-only).
    """
    if coord_layout == "atom37":
        o_idx = 4
        min_atoms = 5
    elif coord_layout == "pdb":
        o_idx = 3
        min_atoms = 4
    else:
        raise ValueError(
            f"coord_layout must be 'atom37' or 'pdb', got {coord_layout!r}"
        )

    if coords.shape[2] < min_atoms:
        return None

    import pydssp

    # Extract N, CA, C, O
    ncao = coords[:, :, [0, 1, 2, o_idx], :]  # [b, n, 4, 3]

    # Refine mask: exclude residues with missing backbone atoms
    if coord_mask is not None and coord_mask.shape[2] >= min_atoms:
        ncao_valid = (
            coord_mask[:, :, 0]
            & coord_mask[:, :, 1]
            & coord_mask[:, :, 2]
            & coord_mask[:, :, o_idx]
        )
        valid_mask = mask & ncao_valid
    else:
        valid_mask = mask

    # PyDSSP expects [batch, L, 4, 3]; coords in Angstrom
    dssp_out = pydssp.assign(ncao, out_type="index")  # [b, n]

    # Set invalid positions to -1 for cross_entropy ignore_index
    dssp_target = dssp_out.long().clone()
    dssp_target = torch.where(valid_mask, dssp_target, torch.full_like(dssp_target, -1))

    return dssp_target
