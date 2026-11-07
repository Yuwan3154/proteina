# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
DSSP secondary structure utilities for auxiliary loss.
Computes 3-state DSSP targets (loop, helix, strand) from backbone coordinates.
"""

from typing import Literal, Optional

import pydssp
import torch

# Exact CA-distance thresholds from TMalign / USalign sec_str(d13, d14, d15, d24, d25, d35)
# Reference: Zhang & Skolnick (2005) Nucleic Acids Res 33:2302-2309; USalign TMalign.h make_sec / smooth.
TMALIGN_HELIX_D15 = 6.37
TMALIGN_HELIX_D14_D25 = 5.18
TMALIGN_HELIX_D13_D24_D35 = 5.45
TMALIGN_HELIX_TOL = 2.1

TMALIGN_STRAND_D15 = 13.0
TMALIGN_STRAND_D14_D25 = 10.4
TMALIGN_STRAND_D13_D24_D35 = 6.1
TMALIGN_STRAND_TOL = 1.42


def smooth_tmalign_dssp(sec: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply USalign's exact 3-pass `smooth(int *sec, int len)` on 3-state labels (0=loop, 1=helix, 2=strand).

    Pass 1: --x-- -> ----- (isolated single helix/strand residue surrounded by 2 non-matching on each side).
    Pass 2: --xx-- -> ------ (isolated pair of helix/strand residues surrounded by 2 non-matching on each side).
    Pass 3: x-x -> xxx (single-residue gap inside a helix or strand bridged to match its immediate neighbors).
    """
    out = sec.clone()
    B, L = out.shape
    for b in range(B):
        s = out[b]
        # Pass 1: --x-- -> -----
        if L >= 5:
            for i in range(2, L - 2):
                v = int(s[i].item())
                if v in (1, 2):
                    if (
                        int(s[i - 2].item()) != v
                        and int(s[i - 1].item()) != v
                        and int(s[i + 1].item()) != v
                        and int(s[i + 2].item()) != v
                    ):
                        s[i] = 0
        # Pass 2: --xx-- -> ------
        if L > 5:
            for i in range(2, L - 3):
                for j in (1, 2):
                    if (
                        int(s[i - 2].item()) != j
                        and int(s[i - 1].item()) != j
                        and int(s[i].item()) == j
                        and int(s[i + 1].item()) == j
                        and int(s[i + 2].item()) != j
                        and int(s[i + 3].item()) != j
                    ):
                        s[i] = 0
                        s[i + 1] = 0
        # Pass 3: x-x -> xxx
        if L >= 3:
            for i in range(1, L - 1):
                for j in (1, 2):
                    if (
                        int(s[i - 1].item()) == j
                        and int(s[i].item()) != j
                        and int(s[i + 1].item()) == j
                    ):
                        s[i] = j
    if valid_mask is not None:
        out = torch.where(valid_mask, out, torch.full_like(out, -1))
    return out


def compute_tmalign_dssp(
    ca_coords: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth: bool = False,
) -> torch.Tensor:
    """Compute 3-state SSE (0=loop, 1=helix, 2=strand, -1=invalid) using TMalign/USalign `make_sec`.

    Args:
        ca_coords: [b, n, 3] CA coordinates in Angstrom.
        valid_mask: [b, n] boolean mask; True = valid residue with resolved CA.
        smooth: If True, apply USalign's 3-pass `smooth()` post-filter.

    Returns:
        dssp_target: [b, n] long tensor with values 0=loop, 1=helix, 2=strand, -1=invalid.
    """
    b, n, _ = ca_coords.shape
    sec = torch.zeros((b, n), dtype=torch.long, device=ca_coords.device)
    if n >= 5:
        ca = ca_coords.double()
        j1 = ca[:, :-4, :]
        j2 = ca[:, 1:-3, :]
        j3 = ca[:, 2:-2, :]
        j4 = ca[:, 3:-1, :]
        j5 = ca[:, 4:, :]

        d13 = torch.linalg.norm(j1 - j3, dim=-1)
        d14 = torch.linalg.norm(j1 - j4, dim=-1)
        d15 = torch.linalg.norm(j1 - j5, dim=-1)
        d24 = torch.linalg.norm(j2 - j4, dim=-1)
        d25 = torch.linalg.norm(j2 - j5, dim=-1)
        d35 = torch.linalg.norm(j3 - j5, dim=-1)

        win_valid = (
            valid_mask[:, :-4]
            & valid_mask[:, 1:-3]
            & valid_mask[:, 2:-2]
            & valid_mask[:, 3:-1]
            & valid_mask[:, 4:]
        )

        is_helix = (
            win_valid
            & ((d15 - TMALIGN_HELIX_D15).abs() < TMALIGN_HELIX_TOL)
            & ((d14 - TMALIGN_HELIX_D14_D25).abs() < TMALIGN_HELIX_TOL)
            & ((d25 - TMALIGN_HELIX_D14_D25).abs() < TMALIGN_HELIX_TOL)
            & ((d13 - TMALIGN_HELIX_D13_D24_D35).abs() < TMALIGN_HELIX_TOL)
            & ((d24 - TMALIGN_HELIX_D13_D24_D35).abs() < TMALIGN_HELIX_TOL)
            & ((d35 - TMALIGN_HELIX_D13_D24_D35).abs() < TMALIGN_HELIX_TOL)
        )
        is_strand = (
            win_valid
            & (~is_helix)
            & ((d15 - TMALIGN_STRAND_D15).abs() < TMALIGN_STRAND_TOL)
            & ((d14 - TMALIGN_STRAND_D14_D25).abs() < TMALIGN_STRAND_TOL)
            & ((d25 - TMALIGN_STRAND_D14_D25).abs() < TMALIGN_STRAND_TOL)
            & ((d13 - TMALIGN_STRAND_D13_D24_D35).abs() < TMALIGN_STRAND_TOL)
            & ((d24 - TMALIGN_STRAND_D13_D24_D35).abs() < TMALIGN_STRAND_TOL)
            & ((d35 - TMALIGN_STRAND_D13_D24_D35).abs() < TMALIGN_STRAND_TOL)
        )
        sec[:, 2:-2] = torch.where(
            is_helix,
            torch.ones_like(sec[:, 2:-2]),
            torch.where(is_strand, torch.full_like(sec[:, 2:-2], 2), torch.zeros_like(sec[:, 2:-2])),
        )
    if smooth:
        sec = smooth_tmalign_dssp(sec, valid_mask=valid_mask)
    else:
        sec = torch.where(valid_mask, sec, torch.full_like(sec, -1))
    return sec


def compute_dssp_target(
    coords: torch.Tensor,
    mask: torch.Tensor,
    coord_mask: Optional[torch.Tensor] = None,
    coord_layout: Literal["atom37", "pdb"] = "atom37",
    donor_mask: Optional[torch.Tensor] = None,
    dssp_def: Literal["h_bond", "tmalign", "tmalign_smooth"] = "h_bond",
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
        donor_mask: Optional [1, n]; 0 marks residues that cannot donate a backbone H-bond (proline).
                    None keeps the pydssp call exactly as before.
        dssp_def: "h_bond" (default pydssp), "tmalign" (CA-only USalign make_sec),
                  or "tmalign_smooth" (CA-only USalign make_sec + 3-pass smooth).

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

    if dssp_def in ("tmalign", "tmalign_smooth"):
        if coords.shape[2] < 2:
            return None
        ca_coords = coords[:, :, 1, :]
        if coord_mask is not None and coord_mask.shape[2] >= 2:
            valid_mask = mask & coord_mask[:, :, 1]
        else:
            valid_mask = mask
        return compute_tmalign_dssp(
            ca_coords, valid_mask, smooth=(dssp_def == "tmalign_smooth")
        )

    if dssp_def != "h_bond":
        raise ValueError(
            f"dssp_def must be 'h_bond', 'tmalign', or 'tmalign_smooth', got {dssp_def!r}"
        )

    if coords.shape[2] < min_atoms:
        return None

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
    if donor_mask is None:
        dssp_out = pydssp.assign(ncao, out_type="index")  # [b, n]
    else:  # pydssp takes one [n] donor mask for the whole batch
        assert donor_mask.shape == (1, ncao.shape[1]), donor_mask.shape
        dssp_out = pydssp.assign(ncao, donor_mask=donor_mask[0].float(), out_type="index")

    # Set invalid positions to -1 for cross_entropy ignore_index
    dssp_target = dssp_out.long().clone()
    dssp_target = torch.where(valid_mask, dssp_target, torch.full_like(dssp_target, -1))

    return dssp_target
