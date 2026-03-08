# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities for computing per-residue ext_lig labels.

ext_lig encodes whether external (non-self-chain, non-water) atoms exist near
each residue's CA atom:

    0 = absent   -- no external atoms within cutoff
    1 = present  -- at least one external atom within cutoff
    2 = unknown  -- CA missing or AFDB predicted structure
"""

from typing import List, Optional, Union

import numpy as np
import torch

EXT_LIG_ABSENT = 0
EXT_LIG_PRESENT = 1
EXT_LIG_UNKNOWN = 2

WATER_RESIDUE_NAMES = {"HOH", "WAT", "TIP", "TIP3", "SOL", "H2O"}
_CA_CUTOFF = 8.0


def compute_ext_lig_from_df(
    full_df,
    self_chains: Union[str, List[str]],
    graph_residue_ids: List[str],
    cutoff: float = _CA_CUTOFF,
    fill_value_coords: float = 1e-5,
) -> torch.Tensor:
    """Compute per-residue ext_lig from the full deposit dataframe.

    Args:
        full_df: pandas DataFrame of the entire deposit (all chains, ATOM+HETATM).
        self_chains: Chain(s) constituting the "self" protein. Atoms in these
            chains do not count as external.  Pass ``"all"`` if the entire
            deposit is one chain (AFDB).
        graph_residue_ids: Ordered list of residue ID strings produced by
            ``protein_to_pyg`` (e.g. ``"A:ALA:10:"``).  The output tensor is
            aligned to this ordering.
        cutoff: Distance threshold in Angstrom (default 8.0).
        fill_value_coords: Fill value used for missing atoms in the coordinate
            tensor (used to detect missing CA).

    Returns:
        Long tensor [L] with values in {0=absent, 1=present, 2=unknown}.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    L = len(graph_residue_ids)
    ext_lig = torch.full((L,), EXT_LIG_ABSENT, dtype=torch.long)

    if self_chains == "all":
        chain_ids_in_df = full_df["chain_id"].unique()
        if len(chain_ids_in_df) <= 1:
            return ext_lig
        self_chain_set = set(chain_ids_in_df)
    else:
        if isinstance(self_chains, str):
            self_chain_set = {self_chains}
        else:
            self_chain_set = set(self_chains)

    external_mask = ~full_df["chain_id"].isin(self_chain_set)
    non_water_mask = ~full_df["residue_name"].isin(WATER_RESIDUE_NAMES)
    ext_atoms = full_df[external_mask & non_water_mask]

    if len(ext_atoms) == 0:
        return ext_lig

    ext_coords = ext_atoms[["x_coord", "y_coord", "z_coord"]].values.astype(np.float64)
    tree = cKDTree(ext_coords)

    self_atoms = full_df[full_df["chain_id"].isin(self_chain_set)]
    self_ca = self_atoms[self_atoms["atom_name"] == "CA"]

    ca_resid_to_coords = {}
    for _, row in self_ca.iterrows():
        chain = row["chain_id"]
        resname = row["residue_name"]
        resnum = str(int(row["residue_number"]))
        insertion = row.get("insertion", "")
        if pd.isna(insertion):
            insertion = ""
        resid = f"{chain}:{resname}:{resnum}:{insertion}"
        ca_resid_to_coords[resid] = np.array(
            [row["x_coord"], row["y_coord"], row["z_coord"]], dtype=np.float64
        )

    for i, resid in enumerate(graph_residue_ids):
        ca_coord = ca_resid_to_coords.get(resid)
        if ca_coord is None:
            ext_lig[i] = EXT_LIG_UNKNOWN
            continue
        if np.any(np.abs(ca_coord) < 1e-4) and np.allclose(ca_coord, 0.0, atol=1e-4):
            ext_lig[i] = EXT_LIG_UNKNOWN
            continue
        neighbors = tree.query_ball_point(ca_coord, r=cutoff)
        if len(neighbors) > 0:
            ext_lig[i] = EXT_LIG_PRESENT

    return ext_lig


def make_unknown_ext_lig(length: int) -> torch.Tensor:
    """Return ext_lig tensor of all unknown for AFDB structures."""
    return torch.full((length,), EXT_LIG_UNKNOWN, dtype=torch.long)
