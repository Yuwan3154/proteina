# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION & AFFILIATES
# is strictly prohibited.

"""CATH code string-to-index conversion utilities for data pipeline."""

import os
from typing import Dict, List, Literal, Optional, Tuple

import torch

from proteinfoundation.utils.ff_utils.pdb_utils import extract_cath_code_by_level


def load_cath_mapping(cath_code_dir: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int, int, int]:
    """Load CATH label mapping and return class mappings and num_classes.

    Args:
        cath_code_dir: Directory containing cath_label_mapping.pt

    Returns:
        Tuple of (class_mapping_C, class_mapping_A, class_mapping_T,
                  num_classes_C, num_classes_A, num_classes_T)
    """
    mapping_file = os.path.join(cath_code_dir, "cath_label_mapping.pt")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"{mapping_file} does not exist")
    class_mapping = torch.load(mapping_file, weights_only=False)
    mapping_C = class_mapping["C"]
    mapping_A = class_mapping["A"]
    mapping_T = class_mapping["T"]
    return (
        mapping_C,
        mapping_A,
        mapping_T,
        len(mapping_C),
        len(mapping_A),
        len(mapping_T),
    )


def parse_cath_codes_to_indices(
    cath_code_list: List[List[str]],
    class_mapping_C: Dict[str, int],
    class_mapping_A: Dict[str, int],
    class_mapping_T: Dict[str, int],
    num_classes_C: int,
    num_classes_A: int,
    num_classes_T: int,
) -> List[List[List[int]]]:
    """Convert list of CATH code strings to indices at C, A, T levels.

    Args:
        cath_code_list: List of cath codes per sample. Each sample has a list of
            CATH code strings (e.g. [["1.10.10.10"], ["1.10.20.10", "1.10.30.10"]])
        class_mapping_C, class_mapping_A, class_mapping_T: Dict mapping code string to index
        num_classes_C, num_classes_A, num_classes_T: Used as null index when unknown

    Returns:
        List of list of [C_idx, A_idx, T_idx] per sample. Shape [b, num_labels_per_sample, 3].
        Empty samples get [[num_classes_C, num_classes_A, num_classes_T]].
    """
    null_idx = [num_classes_C, num_classes_A, num_classes_T]
    results = []
    for cath_codes in cath_code_list:
        result = []
        for cath_code in cath_codes:
            result.append(
                [
                    class_mapping_C.get(extract_cath_code_by_level(cath_code, "C"), num_classes_C),
                    class_mapping_A.get(extract_cath_code_by_level(cath_code, "A"), num_classes_A),
                    class_mapping_T.get(extract_cath_code_by_level(cath_code, "T"), num_classes_T),
                ]
            )
        if len(result) == 0:
            result = [null_idx]
        results.append(result)
    return results


def mask_cath_indices_by_level(
    indices: torch.Tensor,
    level: Literal["C", "A", "T", "H"],
    num_classes_C: int,
    num_classes_A: int,
    num_classes_T: int,
) -> torch.Tensor:
    """Set C/A/T columns to null index when masking that level.

    Matches string-level semantics: masking a level also masks more specific levels.
    H level does not affect our [C,A,T] indices (H is the 4th CATH component).

    Args:
        indices: Tensor of shape [..., 3] with (C_idx, A_idx, T_idx) in last dim
        level: Which level to mask (H=no-op for indices, T=topo, A=arch+T, C=all)
        num_classes_C, num_classes_A, num_classes_T: Null indices for each level

    Returns:
        indices with the masked column(s) set to null
    """
    indices = indices.clone()
    if level == "H":
        pass  # H is 4th component; we only store C,A,T
    elif level == "T":
        indices[..., 2] = num_classes_T
    elif level == "A":
        indices[..., 1] = num_classes_A
        indices[..., 2] = num_classes_T
    elif level == "C":
        indices[..., 0] = num_classes_C
        indices[..., 1] = num_classes_A
        indices[..., 2] = num_classes_T
    return indices


def cath_code_strings_to_indices_for_model(
    cath_code_list: List[List[str]],
    cath_code_dir: str,
    multilabel_mode: Literal["sample", "average", "sum", "transformer"] = "sample",
    device: Optional["torch.device"] = None,
) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
    """Convert CATH code strings to indices for model input (inference/manual batches).

    Args:
        cath_code_list: List of list of CATH code strings per sample
        cath_code_dir: Directory containing cath_label_mapping.pt
        multilabel_mode: How to aggregate multiple labels (sample/average/sum/transformer)
        device: Device for output tensors

    Returns:
        (indices, mask): indices tensor [b, 3] or [b, max_labels, 3];
            mask is None for sample mode, else [b, max_labels] (True=pad)
    """
    import random

    mapping_C, mapping_A, mapping_T, nC, nA, nT = load_cath_mapping(cath_code_dir)
    null_idx = [nC, nA, nT]
    indices_nested = parse_cath_codes_to_indices(
        cath_code_list,
        mapping_C, mapping_A, mapping_T,
        nC, nA, nT,
    )
    if multilabel_mode == "sample":
        result = []
        for sample_indices in indices_nested:
            if len(sample_indices) == 0:
                result.append(null_idx)
            else:
                idx = random.randint(0, len(sample_indices) - 1)
                result.append(sample_indices[idx])
        out = torch.tensor(result, dtype=torch.long, device=device)
        return out, None
    # average/sum/transformer: pad to [b, max_labels, 3]
    max_num_label = max(max(len(s), 1) for s in indices_nested)
    padded = []
    mask = []
    for sample_indices in indices_nested:
        n = len(sample_indices)
        if n == 0:
            padded.append([null_idx] * max_num_label)
            mask.append([True] * max_num_label)
        else:
            row = list(sample_indices) + [null_idx] * (max_num_label - n)
            padded.append(row)
            mask.append([False] * n + [True] * (max_num_label - n))
    out = torch.tensor(padded, dtype=torch.long, device=device)
    mask_out = torch.tensor(mask, dtype=torch.bool, device=device)
    return out, mask_out


def apply_fold_mask_to_indices(
    indices: torch.Tensor,
    mask_T: bool,
    mask_A: bool,
    mask_C: bool,
    num_classes_C: int,
    num_classes_A: int,
    num_classes_T: int,
) -> torch.Tensor:
    """Apply progressive fold masking to CATH indices (T, A, C by flags).

    Matches model_trainer_base logic: mask_C implies mask_A and mask_T;
    mask_A implies mask_T.

    Args:
        indices: Tensor [..., 3] with (C_idx, A_idx, T_idx)
        mask_T, mask_A, mask_C: Whether to mask each level
        num_classes_*: Null indices

    Returns:
        Masked indices tensor
    """
    out = indices.clone()
    if mask_C:
        out[..., 0] = num_classes_C
        out[..., 1] = num_classes_A
        out[..., 2] = num_classes_T
    elif mask_A:
        out[..., 1] = num_classes_A
        out[..., 2] = num_classes_T
    elif mask_T:
        out[..., 2] = num_classes_T
    return out
