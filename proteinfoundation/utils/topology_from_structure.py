# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Reads a topology reference straight off a PDB/CIF file, for conditioned inference.

Training references come out of a precomputed index keyed by chain id, which inference cannot
use: the structure a user wants to condition on is usually not in the training set at all. This
module produces the same description from a structure file instead.

It deliberately reuses the training path end to end -- the same parser, the same DSSP call, the
same ConFind contacts at the SAME threshold the index was built with, and the index's own
standardisation constants -- because a reference assembled even slightly differently from the
training ones is a distribution the model has never seen.

Kept out of ``datasets/topology_reference`` on purpose: this pulls in graphein and (optionally)
the ConFind binary, and the transform is imported by every dataloader worker and by the trainer.
"""

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_STRAND,
    dssp_to_runs,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform
from proteinfoundation.graphein_utils.graphein_utils import (
    protein_to_pyg,
    read_pdb_to_dataframe,
)
from proteinfoundation.utils.confind_utils import confind_raw_contact_map
from proteinfoundation.openfold_stub.np.residue_constants import resname_to_idx
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.dssp_utils import compute_dssp_target
from proteinfoundation.utils.frame2confind_utils import Frame2ConFindTransformPredictor

# protein_to_pyg writes this into every coordinate slot it could not fill, and pdb_data.py
# derives coord_mask by comparing against it. Same constant, same derivation.
FILL_VALUE_COORDS = 1e-5


def parse_structure(structure_path: str, chain: str = "all") -> "Data":
    """PDB/CIF -> a graph shaped exactly like the training pipeline's, BEFORE its atom37 reorder.

    PDB ordering is not an oversight: ``graph_to_f2s_coords`` and ``write_graph_pdb`` both apply
    the reorder themselves, so handing them atom37 coords would silently permute the atoms twice.
    """
    if not os.path.exists(structure_path):
        raise FileNotFoundError(structure_path)
    df = read_pdb_to_dataframe(path=structure_path)
    graph = protein_to_pyg(
        df=df.copy(),
        path=structure_path,
        chain_selection=chain,
        keep_insertions=True,
        fill_value_coords=FILL_VALUE_COORDS,
    )
    graph.coords = torch.as_tensor(np.asarray(graph.coords, dtype=np.float32))
    graph.coord_mask = (graph.coords != FILL_VALUE_COORDS)[..., 0]
    graph.residue_type = torch.tensor(
        [resname_to_idx[r] for r in graph.residues], dtype=torch.long
    )
    return graph


def load_structure(structure_path: str, chain: str = "all"):
    """PDB/CIF -> (coords [L, 37, 3] in OpenFold atom order, coord_mask [L, 37], dssp [L])."""
    return atom37_and_dssp(parse_structure(structure_path, chain=chain))


def atom37_and_dssp(graph):
    """Graph in PDB atom order -> the atom37 view the topology featurisation expects, plus DSSP.

    Mirrors ``pdb_data.PDBDataset`` exactly: same reorder, and DSSP read from the atom37 layout
    (where O is index 4, not 3 -- reading index 3 there feeds CB to pydssp and returns all-loop).
    """
    coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
    coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
    L = coords.shape[0]
    dssp = compute_dssp_target(
        coords[None],
        torch.ones(1, L, dtype=torch.bool),
        coord_mask[None],
        coord_layout="atom37",
    )
    if dssp is None:
        raise ValueError(
            f"DSSP needs N/CA/C/O and the structure resolves only {coords.shape[1]} atom slots -- "
            "a CA-only model cannot supply a topology."
        )
    return coords, coord_mask, dssp[0]


def structure_to_topology_source(
    structure_path: str,
    index_path: str,
    chain: str = "all",
    contact_method: str = "frame2confind",
    frame2confind_checkpoint: Optional[str] = None,
    rotlib_path: Optional[str] = None,
    confind_bin: str = "confind",
    raw_contact_map: Optional[torch.Tensor] = None,
) -> Tuple[Sequence[Tuple[int, int]], torch.Tensor, torch.Tensor]:
    """Structure file -> (runs, element contact map, structural pair features).

    Split out from the assembly because those three describe the structure while the assembly
    rescales them onto a target generation length -- one structure, many lengths.

    ``contact_method`` defaults to ``frame2confind`` because that is what actually produced the
    training data: ``contact_map_confind`` in the processed .pt files was backfilled with
    Frame2ConFind on GPU, so the CPU ``confind`` binary is the OUT-of-distribution choice here
    despite the field name. It is kept as an option. ``raw_contact_map`` accepts an [L, L] map
    computed elsewhere and skips both.

    Whichever produced it, the map is thresholded at the value stored IN THE INDEX, so a reference
    built here is binarised exactly like the training ones.
    """
    index = torch.load(index_path, map_location="cpu", weights_only=False, mmap=True)
    threshold = float(index["contact_threshold"])
    min_len = int(index["min_len"])

    graph = parse_structure(structure_path, chain=chain)
    coords, coord_mask, dssp = atom37_and_dssp(graph)
    L = coords.shape[0]

    if raw_contact_map is None:
        if contact_method == "frame2confind":
            predictor = Frame2ConFindTransformPredictor.get_or_create(
                **({"checkpoint": frame2confind_checkpoint} if frame2confind_checkpoint else {})
            )
            raw_contact_map = predictor.predict_graph(graph)
        elif contact_method == "confind":
            if rotlib_path is None:
                raise ValueError("contact_method='confind' needs rotlib_path")
            raw_contact_map = confind_raw_contact_map(
                graph, rotlib_path, confind_bin=confind_bin
            )
        else:
            raise ValueError(
                f"contact_method must be 'frame2confind' or 'confind', got {contact_method!r}"
            )
    raw_contact_map = torch.as_tensor(raw_contact_map, dtype=torch.float32)
    if raw_contact_map.shape != (L, L):
        raise ValueError(
            f"contact map is {tuple(raw_contact_map.shape)} but the structure has {L} residues"
        )

    cm = (raw_contact_map >= threshold).float()
    runs = dssp_to_runs(dssp, min_len=min_len)
    ref, keep = sse_contact_reference(cm, runs, keep_types=(DSSP_HELIX, DSSP_STRAND))
    structural = sse_structural_pair_features(cm, coords, coord_mask, runs, keep)
    return runs, ref, structural


def topology_reference_from_structure(
    structure_path: str,
    index_path: str,
    target_len: int,
    chain: str = "all",
    max_topology_len: int = 128,
    max_topology_he_len: int = 64,
    contact_method: str = "frame2confind",
    frame2confind_checkpoint: Optional[str] = None,
    rotlib_path: Optional[str] = None,
    confind_bin: str = "confind",
    raw_contact_map: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """The six ``topology_*`` tensors for a structure file, rescaled onto ``target_len``.

    ``max_topology_len`` / ``max_topology_he_len`` must match the model's, since they are the caps
    the reference was truncated to during training.
    """
    runs, ref, structural = structure_to_topology_source(
        structure_path,
        index_path,
        chain=chain,
        contact_method=contact_method,
        frame2confind_checkpoint=frame2confind_checkpoint,
        rotlib_path=rotlib_path,
        confind_bin=confind_bin,
        raw_contact_map=raw_contact_map,
    )
    transform = TopologyReferenceTransform(
        index_path=index_path,
        max_topology_len=max_topology_len,
        max_topology_he_len=max_topology_he_len,
        mutate_prob=0.0,
        sigma_frac=0.0,
        drop_prob=0.0,
    )
    return transform.assemble_reference(runs, ref, structural, target_len, augment=False)
