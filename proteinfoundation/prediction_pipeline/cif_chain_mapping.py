"""Utilities for resolving ground-truth mmCIF chain IDs.

Generated templates and predictions in the Proteina pipelines are single-chain
PDBs and should not be passed through this module. These helpers are only for
ground-truth/reference structures where dataset IDs may use mmCIF label chain
IDs while USalign/Bio.PDB expect author chain IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_cif_path(structure_path: str) -> bool:
    return Path(str(structure_path)).suffix.lower() in {".cif", ".mmcif"}


def cif_label_to_auth_chain(cif_path: str, label_chain: Optional[str]) -> Optional[str]:
    """Map a ground-truth mmCIF label_asym_id chain to auth_asym_id.

    If ``label_chain`` is already an author chain ID and not present as a label
    ID, it is returned unchanged. Non-CIF paths are returned unchanged.
    """
    if label_chain is None:
        return None
    if not _is_cif_path(cif_path):
        return label_chain

    mmcif = MMCIF2Dict(str(cif_path))
    labels = _as_list(mmcif.get("_atom_site.label_asym_id"))
    auths = _as_list(mmcif.get("_atom_site.auth_asym_id"))
    if not labels or not auths or len(labels) != len(auths):
        return label_chain

    mapped_auths = {auth for label, auth in zip(labels, auths) if label == label_chain}
    if len(mapped_auths) == 1:
        return next(iter(mapped_auths))
    if label_chain in set(auths):
        return label_chain
    return label_chain


def resolve_ground_truth_usalign_chain(
    structure_path: str,
    chain: Optional[str],
) -> Optional[str]:
    """Resolve a chain ID for a known ground-truth/reference USalign input."""
    return cif_label_to_auth_chain(structure_path, chain)


def resolve_biopython_chain_for_structure(
    structure_path: str,
    chain: Optional[str],
) -> Optional[str]:
    """Resolve a chain ID for Bio.PDB structure parsing of reference inputs."""
    return cif_label_to_auth_chain(structure_path, chain)
