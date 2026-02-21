# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ops for all atom representations."""

from typing import Tuple

import torch

from proteinfoundation.openfold_stub.np import residue_constants as rc
from proteinfoundation.openfold_stub.utils import tensor_utils


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]


def atom14_to_atom37(
    atom14_data: torch.Tensor,  # (*, N, 14, ...)
    aatype: torch.Tensor # (*, N)
) -> Tuple:    # (*, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    idx_atom37_to_atom14 = get_rc_tensor(rc.RESTYPE_ATOM37_TO_ATOM14, aatype).long()
    no_batch_dims = len(aatype.shape) - 1
    atom37_data = tensor_utils.batched_gather(
        atom14_data, 
        idx_atom37_to_atom14, 
        dim=no_batch_dims + 1, 
        no_batch_dims=no_batch_dims + 1
    )
    atom37_mask = get_rc_tensor(rc.RESTYPE_ATOM37_MASK, aatype) 
    if len(atom14_data.shape) == no_batch_dims + 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == no_batch_dims + 3:
        atom37_data *= atom37_mask[..., None].to(dtype=atom37_data.dtype)
    else:
        raise ValueError("Incorrectly shaped data")
    return atom37_data, atom37_mask
