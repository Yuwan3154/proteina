# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gzip
import os
import pickle
import re
from collections import defaultdict

try:
    import msgpack
except ImportError:
    msgpack = None
from math import prod
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import wget
from loguru import logger
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from scipy.spatial.transform import Rotation as Scipy_Rotation
from torch_geometric import transforms as T
from torch_geometric.data import Data

from proteinfoundation.datasets.cath_utils import load_cath_mapping, parse_cath_codes_to_indices
from proteinfoundation.utils.dense_padding_data_loader import FLOAT_PADDING_VALUE


def sample_uniform_rotation(shape=(), dtype=None, device=None) -> torch.Tensor:
    """Samples rotation matrices uniformly from SO(3).

    Args:
        shape: Batch dimensions for sampling multiple rotations
        dtype: Data type for the output tensor
        device: Device to place the output tensor on

    Returns:
        Tensor of shape [*shape, 3, 3] containing uniformly sampled rotation matrices
    """
    return torch.tensor(
        Scipy_Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class CopyCoordinatesTransform(T.BaseTransform):
    """Creates a backup copy of coordinates before applying modifications.

    This transform copies the original coordinates to coords_unmodified before any
    other transformations (like noising or rotations) are applied.
    """

    def forward(self, graph: Data) -> Data:
        """Copies coordinates to coords_unmodified.

        Args:
            graph: PyG Data object containing protein structure data

        Returns:
            Modified graph with coords_unmodified added
        """
        graph.coords_unmodified = graph.coords.clone()
        return graph


class DSSPTargetTransform(T.BaseTransform):
    """Computes 3-state DSSP secondary structure targets from backbone coordinates.

    Adds graph.dssp_target [L] with values 0=loop, 1=helix, 2=strand. Invalid residues
    (missing N/CA/C/O or masked) are set to -1 (ignore_index for CE loss).
    Skips if coords have fewer than 4 atoms (e.g. CA-only data).
    """

    def forward(self, graph: Data) -> Data:
        """Computes DSSP targets and adds to graph.

        Args:
            graph: PyG Data with coords [L, atoms, 3], coord_mask [L, atoms]

        Returns:
            Graph with dssp_target [L] added (or unchanged if CA-only)
        """
        coords = getattr(graph, "coords", None)
        coord_mask = getattr(graph, "coord_mask", None)
        if coords is None or coords.shape[1] < 4:
            return graph

        import pydssp

        # Extract N, CA, C, O (indices 0, 1, 2, 3)
        ncao = coords[:, [0, 1, 2, 3], :]  # [L, 4, 3]
        ncao_batch = ncao.unsqueeze(0)  # [1, L, 4, 3] for PyDSSP

        dssp_out = pydssp.assign(ncao_batch, out_type="index")  # [1, L]
        dssp_target = dssp_out[0].long().clone()  # [L]

        # Mask invalid residues (missing backbone atoms)
        if coord_mask is not None and coord_mask.shape[1] >= 4:
            ncao_valid = (
                coord_mask[:, 0] & coord_mask[:, 1] & coord_mask[:, 2] & coord_mask[:, 3]
            )
            dssp_target = torch.where(
                ncao_valid, dssp_target, torch.full_like(dssp_target, -1)
            )

        graph.dssp_target = dssp_target
        return graph


class ChainBreakPerResidueTransform(T.BaseTransform):
    """Identifies chain breaks in protein structures.

    Creates a binary mask indicating whether each residue has a chain break,
    determined by CA-CA distances exceeding a threshold.
    """

    def __init__(self, chain_break_cutoff: float = 4.0):
        """Initializes the transform.

        Args:
            chain_break_cutoff: Maximum allowed distance between consecutive CA atoms
                before considering it a chain break
        """
        self.chain_break_cutoff = chain_break_cutoff

    def forward(self, graph: Data) -> Data:
        """Identifies chain breaks and adds mask to graph.

        Args:
            graph: PyG Data object containing protein structure

        Returns:
            Graph with added chain_breaks_per_residue mask
        """
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        chain_breaks_per_residue = ca_dists > self.chain_break_cutoff
        graph.chain_breaks_per_residue = torch.cat(
            (
                chain_breaks_per_residue,
                torch.tensor([False], dtype=torch.bool, device=chain_breaks_per_residue.device),
            )
        )
        return graph


class PaddingTransform(T.BaseTransform):
    """Pads tensors in graph to a specified maximum size.

    Ensures all tensors in the graph have consistent size by padding
    with a fill value up to max_size along the first dimension.
    """

    def __init__(self, max_size=256, fill_value=0):
        """Initializes the transform.

        Args:
            max_size: Target size for padding
            fill_value: Value to use for padding
        """
        self.max_size = max_size
        self.fill_value = fill_value

    def forward(self, graph: Data) -> Data:
        """Applies padding to all applicable tensors in graph.

        Args:
            graph: PyG Data object to pad

        Returns:
            Graph with padded tensors
        """
        for key, value in graph:
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1:
                    # For 2D tensors like contact_map [n, n], pad both dimensions
                    fill_value = FLOAT_PADDING_VALUE if torch.is_floating_point(value) else self.fill_value
                    if key == "dssp_target":
                        fill_value = -1  # ignore_index for CE loss
                    elif key == "ext_lig":
                        fill_value = 2  # unknown class for padded residues
                    if "contact_map" in key and value.dim() == 2:
                        # Pad dimension 0 first, then dimension 1
                        value = self.pad_tensor(value, self.max_size, dim=0, fill_value=fill_value)
                        value = self.pad_tensor(value, self.max_size, dim=1, fill_value=fill_value)
                    else:
                        value = self.pad_tensor(value, self.max_size, dim=0, fill_value=fill_value)
                    graph[key] = value
        return graph

    def pad_tensor(self, tensor, max_size, dim, fill_value=0):
        """Pads a single tensor to specified size. Truncates if larger than max_size
        to avoid OOM when .pt files from other configs exceed the current max_size."""
        if tensor.size(dim) > max_size:
            # Truncate to max_size to match dataset config (e.g. max_length 256)
            tensor = tensor.narrow(dim, 0, max_size)
        if tensor.size(dim) >= max_size:
            return tensor
        pad_size = max_size - tensor.size(dim)
        padding = [0] * (2 * tensor.dim())
        padding[2 * (tensor.dim() - 1 - dim) + 1] = pad_size
        return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)

    def __repr__(self) -> str:
        """Get a string representation of the class.

        Returns:
            str: String representation of the class
        """
        return f"{self.__class__.__name__}(max_size={self.max_size}, fill_value={self.fill_value})"


class GlobalRotationTransform(T.BaseTransform):
    """Applies random global rotation to protein coordinates.

    Should be used as the first transform that modifies coordinates to maintain
    consistency in subsequent transformations.
    """

    def __init__(self, rotation_strategy: Literal["uniform"] = "uniform"):
        """Initializes the transform.

        Args:
            rotation_strategy: Method for sampling rotations. Currently only "uniform" supported
        """
        self.rotation_strategy = rotation_strategy

    def forward(self, graph: Data) -> Data:
        """Applies random rotation to coordinates.

        Args:
            graph: PyG Data object containing protein structure

        Returns:
            Graph with rotated coordinates

        Raises:
            ValueError: If rotation_strategy is not supported
        """
        if self.rotation_strategy == "uniform":
            rot = sample_uniform_rotation(dtype=graph.coords.dtype, device=graph.coords.device)
        else:
            raise ValueError(f"Rotation strategy {self.rotation_strategy} not supported")
        graph.coords = torch.matmul(graph.coords, rot)
        return graph


class CATHLabelTransform(T.BaseTransform):
    """Adds CATH labels if available to the protein."""

    def __init__(self, root_dir: str):
        """Initialize the transform with the root directory of the CATH data.

        Args:
            root_dir (str): where the CATH data is/should be stored
        """
        self.root_dir = Path(root_dir)
        self.pdb_chain_cath_uniprot_url = (
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_cath_uniprot.tsv.gz"
        )
        self.cath_id_cath_code_url = (
            "http://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-all.gz"
        )
        self.cath_id_cath_code_filename = Path(self.cath_id_cath_code_url).name
        self.pdb_chain_cath_uniprot_filename = Path(self.pdb_chain_cath_uniprot_url).name

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        if not os.path.exists(self.root_dir / self.pdb_chain_cath_uniprot_filename):
            rank_zero_info("Downloading Uniprot/PDB CATH map...")
            wget.download(self.pdb_chain_cath_uniprot_url, out=str(self.root_dir))

        if not os.path.exists(self.root_dir / self.cath_id_cath_code_filename):
            rank_zero_info("Downloading CATH ID to CATH code map...")
            wget.download(self.cath_id_cath_code_url, out=str(self.root_dir))

        rank_zero_info("Processing Uniprot/PDB CATH map...")
        self.pdbchain_to_cathid_mapping = self._parse_cath_id()
        rank_zero_info("Processing CATH ID to CATH code map...")
        self.cathid_to_cathcode_mapping, self.cathid_to_segment_mapping = self._parse_cath_code()

    def forward(self, graph: Data) -> Data:
        """Map each PDB chain to its CATH ID and CATH code.

        Args:
           graph (Data): A Data object containing a PDB chain ID.

        Returns:
            Data: A Data object with the CATH ID and CATH code mapped to each PDB chain.
        """
        cath_ids = self.pdbchain_to_cathid_mapping.get(graph.id, None)
        if cath_ids:
            cath_code = [self.cathid_to_cathcode_mapping.get(cath_id, "x.x.x.x") for cath_id in cath_ids]
        else:
            cath_code = None
        if cath_code:  # check for list of Nones in cath code list
            graph.cath_code = cath_code
        else:
            graph.cath_code = []
        return graph

    def _parse_cath_id(self) -> Dict[str, str]:
        """Parse the CATH ID for all PDB chains.

        Args:
            None

        Returns:
            Dict[str, str]: Dictionary of PDB chain ID with their
            corresponding CATH ID.
        """
        pdbchain_to_cathid_mapping = defaultdict(list)
        with gzip.open(self.root_dir / self.pdb_chain_cath_uniprot_filename, "rt") as f:
            next(f)  # Skip header line
            for line in f:
                try:
                    pdb, chain, uniprot_id, cath_id = line.strip().split("\t")
                    key = f"{pdb}_{chain}"
                    pdbchain_to_cathid_mapping[key].append(cath_id)
                except ValueError as e:
                    rank_zero_warn(str(e))
                    continue
        return pdbchain_to_cathid_mapping

    def _parse_cath_code(self) -> Dict[str, str]:
        """Parse CATH codes and segment information from the CATH database file.

        Processes the CATH database file to extract CATH IDs, codes, and segment information.
        Handles both single and multiple segment cases, parsing the chain and position
        information for each segment.

        Args:
            None

        Returns:
            tuple:
                - Dict[str, str]: Mapping of CATH IDs to their CATH codes
                - Dict[str, list]: Mapping of CATH IDs to lists of segment information tuples
                                Each tuple contains (chain, segment_start, segment_end)

        Raises:
            ValueError: If the line format is invalid or cannot be parsed
        """
        cathid_to_cathcode_mapping = {}
        cathid_to_segment_mapping = {}

        with gzip.open(self.root_dir / self.cath_id_cath_code_filename, "rt") as f:
            for line in f:
                try:
                    # Split line into components
                    cath_id, cath_version, cath_code, cath_segment_and_chain = line.strip().split()

                    # Process segments
                    cath_segments_and_chains = []
                    if "," in cath_segment_and_chain:
                        segments = cath_segment_and_chain.split(",")
                        for segment in segments:
                            cath_segments_and_chains.append(segment)
                    else:
                        cath_segments_and_chains.append(cath_segment_and_chain)

                    # Separate segments and chains
                    cath_segments = []
                    cath_chains = []
                    for item in cath_segments_and_chains:
                        segment, chain = item.split(":")
                        cath_segments.append(segment)
                        cath_chains.append(chain)

                    # Process start and end positions
                    cath_segments_start = []
                    cath_segments_end = []
                    for segment in cath_segments:
                        start, end = self.split_segment(segment)
                        cath_segments_start.append(start)
                        cath_segments_end.append(end)

                    # Store mappings
                    cathid_to_cathcode_mapping[cath_id] = cath_code

                    # Create segment info list
                    segment_info = []
                    for i in range(len(cath_chains)):
                        segment_info.append((cath_chains[i], cath_segments_start[i], cath_segments_end[i]))
                    cathid_to_segment_mapping[cath_id] = segment_info

                except ValueError as e:
                    rank_zero_warn(str(e))
                    continue

        return cathid_to_cathcode_mapping, cathid_to_segment_mapping

    def split_segment(self, segment: str) -> Tuple[str, str]:
        """Split a segment into start position and end position.

        Handles cases where start or end position are negative numbers.

        Args:
            segment (str): segment description, for example `1-48` or `-2-36` or `1T-14M`.

        Returns:
            Tuple[str, str]: tuple containing start and end position, for example `(1, 48)` or `(-2, 36)` or `(1T, 14M)`.
        """
        # This regex pattern matches (potentially negative) numbers with potentially letters after them and separates segments by hyphen
        pattern = r"(-?\d+[A-Za-z]*)-(-?\d+[A-Za-z]*)"
        match = re.match(pattern, segment)
        if match:
            return match.groups()
        raise ValueError(f"Segment {segment} is not in the correct format")


class CATHToIndicesTransform(T.BaseTransform):
    """Converts CATH code strings to numerical indices for use in the model.

    Runs after CATHLabelTransform. Expects graph.cath_code (list of strings) and
    adds graph.cath_code_indices (list of [C_idx, A_idx, T_idx] per label).
    """

    def __init__(self, cath_code_dir: str):
        """Initialize with path to directory containing cath_label_mapping.pt.

        Args:
            cath_code_dir: Directory containing cath_label_mapping.pt
        """
        self.cath_code_dir = Path(cath_code_dir)
        mapping_C, mapping_A, mapping_T, nC, nA, nT = load_cath_mapping(str(self.cath_code_dir))
        self.class_mapping_C = mapping_C
        self.class_mapping_A = mapping_A
        self.class_mapping_T = mapping_T
        self.num_classes_C = nC
        self.num_classes_A = nA
        self.num_classes_T = nT

    def forward(self, graph: Data) -> Data:
        """Convert graph.cath_code to graph.cath_code_indices.

        Args:
            graph: Data with cath_code (list of strings)

        Returns:
            Graph with cath_code_indices added (list of [C,A,T] int tuples)
        """
        cath_code = getattr(graph, "cath_code", None)
        if cath_code is None or len(cath_code) == 0:
            null_idx = [self.num_classes_C, self.num_classes_A, self.num_classes_T]
            graph.cath_code_indices = [null_idx]
        else:
            result = parse_cath_codes_to_indices(
                [cath_code],
                self.class_mapping_C,
                self.class_mapping_A,
                self.class_mapping_T,
                self.num_classes_C,
                self.num_classes_A,
                self.num_classes_T,
            )
            graph.cath_code_indices = result[0]
        return graph


class PurgeConfindTransform(T.BaseTransform):
    """Removes contact_map_confind from the graph when not using confind contact maps.

    Use this transform in configs that do NOT include ContactMapTransform with
    contact_method='confind'. Preprocessed .pt files may contain contact_map_confind;
    this transform purges it so it does not reach the collator and cause KeyError
    when batching samples with mixed attributes.
    """

    def forward(self, graph: Data) -> Data:
        if hasattr(graph, "contact_map_confind"):
            del graph["contact_map_confind"]
        return graph


class ContactMapTransform(T.BaseTransform):
    """Extracts contact map from ground-truth protein coordinates.

    Creates an L×L binary contact map based on pairwise distances between
    specified atom types (CA or CB) with a configurable distance cutoff.
    """

    def __init__(
        self,
        contact_atom_type: Literal["CA", "CB"] = "CB",
        contact_distance_cutoff: float = 8.0,
        contact_method: Literal["distance", "confind"] = "distance",
        confind_bin: str = "confind",
        confind_rotamer_lib: Optional[str] = None,
        confind_contact_threshold: float = 0.0,
    ):
        """Initializes the transform.

        Args:
            contact_atom_type: Atom type to use for contact calculation.
                "CA" for alpha-carbon, "CB" for beta-carbon (pseudo-CB for Glycine).
            contact_distance_cutoff: Distance threshold in Angstroms for defining contacts.
                Residue pairs with distance <= cutoff are considered in contact.
            contact_method: Contact map generation mode ("distance" or "confind").
                When set to "confind", this transform expects precomputed raw maps
                in the graph under contact_map_confind.
            confind_bin: Path to the confind binary (kept for config compatibility).
            confind_rotamer_lib: Path to rotamer library directory containing EBL.out/BEBL.out.
            confind_contact_threshold: Minimum confind contact score to include (>=).
        """
        self.contact_atom_type = contact_atom_type
        self.contact_distance_cutoff = contact_distance_cutoff
        self.contact_method = contact_method
        self.confind_bin = confind_bin
        self.confind_rotamer_lib = confind_rotamer_lib
        self.confind_contact_threshold = confind_contact_threshold

    def _compute_pseudo_cb(self, coords: torch.Tensor) -> torch.Tensor:
        """Computes pseudo-CB position for residues (used for Glycine).

        The pseudo-CB is placed at a position that would be occupied by CB
        based on the backbone geometry (N, CA, C atoms).

        Args:
            coords: Coordinates tensor of shape [L, num_atoms, 3]
                    Atom order: N(0), CA(1), C(2), O(3), CB(4), ...

        Returns:
            Pseudo-CB coordinates of shape [L, 3]
        """
        # Extract backbone atoms
        n_coords = coords[:, 0, :]   # [L, 3]
        ca_coords = coords[:, 1, :]  # [L, 3]
        c_coords = coords[:, 2, :]   # [L, 3]

        # Compute pseudo-CB using standard geometry
        # CB is approximately at: CA + 1.52 * normalized(rotation of N-CA by ~120 degrees in N-CA-C plane)
        # Simplified approach: CB = CA + normalized(CA-N + CA-C) * 1.52
        ca_n = ca_coords - n_coords  # [L, 3]
        ca_c = ca_coords - c_coords  # [L, 3]
        
        # Normalize vectors
        ca_n_norm = ca_n / (torch.linalg.norm(ca_n, dim=-1, keepdim=True) + 1e-8)
        ca_c_norm = ca_c / (torch.linalg.norm(ca_c, dim=-1, keepdim=True) + 1e-8)
        
        # CB direction is roughly the sum of these normalized vectors
        cb_direction = ca_n_norm + ca_c_norm
        cb_direction = cb_direction / (torch.linalg.norm(cb_direction, dim=-1, keepdim=True) + 1e-8)
        
        # CB is approximately 1.52 Angstroms from CA
        pseudo_cb = ca_coords + cb_direction * 1.52
        
        return pseudo_cb

    def _contact_map_from_distance(self, graph: Data) -> torch.Tensor:
        coords = graph.coords  # [L, num_atoms, 3]

        if self.contact_atom_type == "CA":
            # CA is at index 1
            atom_coords = coords[:, 1, :]  # [L, 3]
        elif self.contact_atom_type == "CB":
            # CB is at index 4, but Glycine doesn't have CB
            cb_coords = coords[:, 4, :]  # [L, 3]

            # Check for missing CB (coordinates might be zero or a fill value)
            # Compute pseudo-CB for all residues and use it where CB is missing
            pseudo_cb = self._compute_pseudo_cb(coords)  # [L, 3]

            # Detect missing CB: if CB coords are very close to zero or a fill value
            # Use residue_type if available to detect Glycine (index 5 in OpenFold)
            if hasattr(graph, "residue_type"):
                is_glycine = (graph.residue_type == 5)  # Glycine index
                atom_coords = torch.where(
                    is_glycine.unsqueeze(-1).expand(-1, 3),
                    pseudo_cb,
                    cb_coords,
                )
            else:
                # Fallback: use pseudo-CB where CB coords seem invalid (near zero)
                cb_norm = torch.linalg.norm(cb_coords, dim=-1)
                is_missing = cb_norm < 0.1
                atom_coords = torch.where(
                    is_missing.unsqueeze(-1).expand(-1, 3),
                    pseudo_cb,
                    cb_coords,
                )
        else:
            raise ValueError(f"Unknown contact_atom_type: {self.contact_atom_type}")

        # Compute pairwise distances
        diff = atom_coords.unsqueeze(0) - atom_coords.unsqueeze(1)  # [L, L, 3]
        distances = torch.linalg.norm(diff, dim=-1)  # [L, L]
        return (distances <= self.contact_distance_cutoff).to(dtype=coords.dtype)

    def _contact_map_from_confind_precomputed(self, graph: Data) -> torch.Tensor:
        raw_map = getattr(graph, "contact_map_confind", None)
        if raw_map is None:
            raise ValueError(
                "contact_map_confind is missing; run the confind precompute step first."
            )
        if not isinstance(raw_map, torch.Tensor):
            raw_map = torch.as_tensor(raw_map)
        if raw_map.shape[0] != graph.coords.shape[0] or raw_map.shape[1] != graph.coords.shape[0]:
            raise ValueError(
                f"contact_map_confind has shape {raw_map.shape}, expected [{graph.coords.shape[0]}, {graph.coords.shape[0]}]."
            )
        return (raw_map.float() >= self.confind_contact_threshold).to(dtype=graph.coords.dtype)

    def forward(self, graph: Data) -> Data:
        """Extracts contact map and adds it to the graph.

        Args:
            graph: PyG Data object containing protein structure with coords [L, num_atoms, 3]

        Returns:
            Graph with added contact_map tensor of shape [L, L] with values 0.0 or 1.0
        """
        if self.contact_method == "distance":
            if hasattr(graph, "contact_map_confind"):
                del graph["contact_map_confind"]
            contact_map = self._contact_map_from_distance(graph)
        elif self.contact_method == "confind":
            contact_map = self._contact_map_from_confind_precomputed(graph)
            # Purge raw confind map after use so it is not passed to collation
            if hasattr(graph, "contact_map_confind"):
                del graph["contact_map_confind"]
        else:
            raise ValueError(f"Unknown contact_method: {self.contact_method}")

        graph.contact_map = contact_map

        return graph

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"contact_atom_type={self.contact_atom_type}, "
            f"contact_distance_cutoff={self.contact_distance_cutoff}, "
            f"contact_method={self.contact_method})"
        )


class TEDLabelTransform(T.BaseTransform):
    """Adds CATH labels if available to the AFDB protein.

    Requires the TED domain summary TSV with CATH labels (21 tab-separated columns,
    column 14 / index 13 contains the CATH code). Download
    ``ted_365m.domain_summary.cath.globularity.taxid.tsv.gz`` from
    https://zenodo.org/records/13908086 and point to it via the ``file_path`` attribute.

    The boundary-only file ``ted_365m_domain_boundaries_consensus_level.tsv`` has only
    3 columns and does **not** contain CATH labels; using it here will raise an error.

    Cache strategy: on first run the full TSV (or existing chunked pickles) is filtered
    down to only the sample IDs present in ``processed/`` next to ``pkl_path``, and saved
    as a single small ``<pkl_path>.subset`` file.  Subsequent startups load only that file,
    which is orders of magnitude faster and smaller than the original chunked pickles.
    The old chunked ``<pkl_path>.N`` files are removed after a successful subset build.
    """

    _MIN_COLUMNS_FOR_CATH = 14  # column index 13 must exist
    _CATH_CODE_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

    def __init__(
        self,
        file_path,
        pkl_path,
        chunk_size=50000000,
    ):
        """Initialize the TEDLabelTransform.

        Args:
            file_path (str): Path to the TED domain summary file containing CATH labels.
                Supports plain ``.tsv`` and gzip-compressed ``.tsv.gz``.
            pkl_path (str): Base path (legacy chunked-pickle path).  A ``<pkl_path>.subset``
                file is used as the fast single-file cache.
            chunk_size (int): Unused; kept for backward-compatible configs.
        """
        self.file_path = file_path
        self.pkl_path = str(pkl_path)
        self.chunk_size = chunk_size
        self.sample_to_cath = {}
        self._process_file()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def _subset_path(self):
        return f"{self.pkl_path}.subset"

    @property
    def _subset_msgpack_path(self):
        return f"{self.pkl_path}.subset.msgpack"

    def _chunked_pkl_exists(self):
        return os.path.exists(f"{self.pkl_path}.0")

    def _subset_exists(self):
        return os.path.exists(self._subset_msgpack_path) or os.path.exists(self._subset_path)

    # ------------------------------------------------------------------
    # Dataset ID discovery
    # ------------------------------------------------------------------

    def _get_dataset_ids(self):
        """Return the set of sample IDs present in processed/ next to pkl_path.

        Returns None when the directory does not exist (fall back to full cache).
        """
        processed_dir = Path(self.pkl_path).parent / "processed"
        if not processed_dir.exists():
            return None
        ids = {p.stem for p in processed_dir.glob("*.pt")}
        return ids if ids else None

    # ------------------------------------------------------------------
    # TSV helpers
    # ------------------------------------------------------------------

    def _open_tsv(self):
        """Return a line iterator for the TSV file, supporting .gz transparently."""
        if str(self.file_path).endswith(".gz"):
            return gzip.open(self.file_path, "rt")
        return open(self.file_path)

    def _validate_schema(self):
        """Read the first non-empty line and verify it has the expected CATH-label schema."""
        with self._open_tsv() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < self._MIN_COLUMNS_FOR_CATH:
                    raise ValueError(
                        f"TEDLabelTransform: the file '{self.file_path}' has only "
                        f"{len(parts)} columns (expected >= {self._MIN_COLUMNS_FOR_CATH}). "
                        f"This looks like the boundary-only TSV "
                        f"(ted_365m_domain_boundaries_consensus_level.tsv) which does NOT "
                        f"contain CATH labels. Please use the domain summary file "
                        f"'ted_365m.domain_summary.cath.globularity.taxid.tsv[.gz]' instead."
                    )
                cath_field = parts[13]
                if cath_field != "-" and not self._CATH_CODE_RE.match(cath_field.split(",")[0]):
                    rank_zero_warn(
                        f"TEDLabelTransform: column 14 of '{self.file_path}' contains "
                        f"'{cath_field}' which does not look like a CATH code (X.X.X.X). "
                        f"Proceeding, but double-check the file format."
                    )
                return

    # ------------------------------------------------------------------
    # Cache build / load
    # ------------------------------------------------------------------

    def _process_file(self):
        if self._subset_exists():
            if msgpack is not None and os.path.exists(self._subset_msgpack_path):
                logger.info(f"Loading AFDB CATH subset cache from {self._subset_msgpack_path} (msgpack)")
                with open(self._subset_msgpack_path, "rb") as f:
                    self.sample_to_cath = msgpack.unpackb(f.read(), strict_map_key=False)
            else:
                logger.info(f"Loading AFDB CATH subset cache from {self._subset_path}")
                with open(self._subset_path, "rb") as f:
                    self.sample_to_cath = pickle.load(f)
            logger.info(f"Loaded {len(self.sample_to_cath)} entries from subset cache.")

        elif self._chunked_pkl_exists():
            dataset_ids = self._get_dataset_ids()
            if dataset_ids is None:
                # processed/ is empty — data hasn't been processed yet.
                # Don't load 8+ GB of pickles against an empty whitelist; CATH labels
                # will simply be absent this run (same as missing label).  The subset
                # cache will be built on the next run once processed/ has .pt files.
                logger.info(
                    "AFDB CATH: no processed .pt files found yet — skipping pickle load. "
                    "CATH labels will be empty this run; subset cache will be built next run."
                )
                return
            logger.info(
                f"Migrating chunked AFDB CATH pickles → subset cache "
                f"(keeping {len(dataset_ids)} dataset entries)."
            )
            self._load_chunked_pickles(filter_ids=dataset_ids)
            self._save_subset()
            self._delete_chunked_pickles()

        else:
            dataset_ids = self._get_dataset_ids()
            if dataset_ids is None:
                logger.info(
                    "AFDB CATH: no processed .pt files and no cache — skipping. "
                    "CATH labels will be empty this run; cache will be built next run."
                )
                return
            logger.info(
                f"Building AFDB CATH subset cache from TSV "
                f"(keeping {len(dataset_ids)} dataset entries)."
            )
            self._validate_schema()
            self._build_from_tsv(filter_ids=dataset_ids)
            self._save_subset()

    def _load_chunked_pickles(self, filter_ids=None):
        chunk_counter = 0
        while os.path.exists(f"{self.pkl_path}.{chunk_counter}"):
            logger.info(f"  Loading chunk {chunk_counter}…")
            with open(f"{self.pkl_path}.{chunk_counter}", "rb") as f:
                chunk = pickle.load(f)
            if filter_ids is not None:
                chunk = {k: v for k, v in chunk.items() if k in filter_ids}
            self.sample_to_cath.update(chunk)
            chunk_counter += 1

    def _build_from_tsv(self, filter_ids=None):
        counter = 0
        with self._open_tsv() as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < self._MIN_COLUMNS_FOR_CATH:
                    continue
                full_sample_id = parts[0]
                cath_codes_str = parts[13]
                if cath_codes_str == "-":
                    continue
                sample_id = "_".join(full_sample_id.split("_")[:-1])
                if filter_ids is not None and sample_id not in filter_ids:
                    continue
                cath_codes = cath_codes_str.split(",")
                if sample_id not in self.sample_to_cath:
                    self.sample_to_cath[sample_id] = []
                    counter += 1
                    if counter % 100000 == 0:
                        logger.info(f"  Collected {counter} matching entries…")
                self.sample_to_cath[sample_id].extend(cath_codes)

    def _save_subset(self):
        if msgpack is not None:
            path = self._subset_msgpack_path
            with open(path, "wb") as f:
                f.write(msgpack.packb(self.sample_to_cath, use_bin_type=True))
            if os.path.exists(self._subset_path):
                os.remove(self._subset_path)
        else:
            path = self._subset_path
            with open(path, "wb") as f:
                pickle.dump(self.sample_to_cath, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(
            f"Saved AFDB CATH subset cache: {len(self.sample_to_cath)} entries, "
            f"{size_mb:.1f} MB → {path}"
        )

    def _delete_chunked_pickles(self):
        chunk_counter = 0
        deleted = []
        while os.path.exists(f"{self.pkl_path}.{chunk_counter}"):
            path = f"{self.pkl_path}.{chunk_counter}"
            os.remove(path)
            deleted.append(path)
            chunk_counter += 1
        if deleted:
            logger.info(f"Deleted {len(deleted)} legacy chunked pickle file(s).")

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def forward(self, graph: Data) -> Data:
        """Call transform on sample.

        Args:
            graph (Data): protein graph

        Returns:
            Data: modified protein graph with CATH label
        """
        graph_id = graph.id
        _cath_code = self.sample_to_cath.get(graph_id, [])
        # For those only have CAT labels, pad them to CATH labels
        cath_code = []
        for code in _cath_code:
            if code.count(".") == 2:
                code = code + ".x"
            cath_code.append(code)
        graph.cath_code = cath_code
        return graph
