#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.utils.align_utils.align_utils import mean_w_mask


def write_graph_pdb(graph, pdb_path: Path) -> None:
    coords = graph.coords.detach().clone()
    coord_mask = getattr(graph, "coord_mask", None)
    residue_type = getattr(graph, "residue_type", None)
    chain_index = getattr(graph, "chains", None)

    if residue_type is None:
        raise ValueError("Graph missing residue_type; cannot write PDB for confind.")
    if coord_mask is None:
        raise ValueError("Graph missing coord_mask; cannot write PDB for confind.")

    idx = PDB_TO_OPENFOLD_INDEX_TENSOR.to(coords.device)
    coords = coords[:, idx, :]
    coord_mask = coord_mask[:, idx]

    # Mask: coord_mask > 0.5 indicates valid (non-fill) coordinates in .pt files
    mask_bool = coord_mask > 0.5
    coords_flat = coords.reshape(1, -1, 3)
    mask_flat = mask_bool.reshape(1, -1)
    mean = mean_w_mask(coords_flat, mask_flat, keepdim=True)
    coords = coords - mean

    write_prot_to_pdb(
        coords.detach().cpu().numpy(),
        str(pdb_path),
        aatype=residue_type.detach().cpu().numpy(),
        atom37_mask=coord_mask.detach().cpu().numpy(),
        chain_index=chain_index.detach().cpu().numpy() if chain_index is not None else None,
        overwrite=True,
        no_indexing=True,
    )


def run_confind(
    pdb_path: Path,
    output_path: Path,
    rotlib_path: str,
    confind_bin: str = "confind",
    omp_threads: int = 1,
    renumber: bool = True,
    timeout: Optional[int] = None,
) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = [
        confind_bin,
        "--p",
        str(pdb_path),
        "--o",
        str(output_path),
        "--rLib",
        rotlib_path,
    ]
    if renumber:
        cmd.append("--ren")
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout,
    )


def parse_confind_contacts(
    contact_path: Path,
    expected_len: Optional[int] = None,
    renumbered: bool = True,
) -> np.ndarray:
    residue_keys = set()
    contacts = []
    with open(contact_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "contact" and len(parts) >= 4:
                res1 = parts[1]
                res2 = parts[2]
                score = float(parts[3])
                chain1, idx1 = res1.split(",")
                chain2, idx2 = res2.split(",")
                idx1 = int(idx1)
                idx2 = int(idx2)
                contacts.append((chain1, idx1, chain2, idx2, score))
                residue_keys.add((chain1, idx1))
                residue_keys.add((chain2, idx2))
            elif parts[0] == "freedom" and len(parts) >= 2:
                chain, idx = parts[1].split(",")
                residue_keys.add((chain, int(idx)))

    if not residue_keys:
        raise ValueError(f"No residues found in confind output {contact_path}")

    if renumbered and expected_len is not None and len({k[0] for k in residue_keys}) == 1:
        nres = expected_len
        contact_map = np.zeros((nres, nres), dtype=np.float32)
        for chain1, idx1, chain2, idx2, score in contacts:
            i = idx1 - 1
            j = idx2 - 1
            if i < 0 or j < 0 or i >= nres or j >= nres:
                raise ValueError(
                    f"Confind residue index out of bounds ({idx1}, {idx2}) for length {nres}"
                )
            if score > contact_map[i, j]:
                contact_map[i, j] = score
                contact_map[j, i] = score
        return contact_map

    keys_sorted = sorted(residue_keys, key=lambda x: (x[0], x[1]))
    key_to_idx: Dict[Tuple[str, int], int] = {key: i for i, key in enumerate(keys_sorted)}
    nres = expected_len if expected_len is not None else len(keys_sorted)
    contact_map = np.zeros((nres, nres), dtype=np.float32)
    for chain1, idx1, chain2, idx2, score in contacts:
        i = key_to_idx[(chain1, idx1)]
        j = key_to_idx[(chain2, idx2)]
        if i < 0 or j < 0 or i >= nres or j >= nres:
            continue
        if score > contact_map[i, j]:
            contact_map[i, j] = score
            contact_map[j, i] = score
    return contact_map


def confind_raw_contact_map(
    graph,
    rotlib_path: str,
    confind_bin: str = "confind",
    omp_threads: int = 1,
    renumber: bool = True,
    timeout: Optional[int] = None,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_path = tmpdir / "input.pdb"
        contact_path = tmpdir / "contacts.txt"
        write_graph_pdb(graph, pdb_path)
        run_confind(
            pdb_path,
            contact_path,
            rotlib_path=rotlib_path,
            confind_bin=confind_bin,
            omp_threads=omp_threads,
            renumber=renumber,
            timeout=timeout,
        )
        return parse_confind_contacts(
            contact_path, expected_len=graph.coords.shape[0], renumbered=renumber
        )
