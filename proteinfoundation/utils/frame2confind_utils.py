#!/usr/bin/env python3
"""Frame2ConFind GPU-accelerated contact map utilities for proteina.

Provides a wrapper that converts proteina graph coordinates (PDB atom ordering)
to the Frame2seq [N, CA, C, CB, O] format and runs the Frame2ConFind neural
predictor to generate continuous ConFind-style contact probability maps.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Proteina PDB ordering indices for the five backbone atoms Frame2seq needs.
# PDB ordering: N=0, CA=1, C=2, O=3, CB=4 (see proteinfoundation/utils/constants.py)
# Frame2seq ordering: N, CA, C, CB, O → PDB indices [0, 1, 2, 4, 3]
_PDB_TO_F2S = [0, 1, 2, 4, 3]

# Default checkpoint path
_DEFAULT_CHECKPOINT = (
    "~/Frame2ConFind/runs/f2s_ft_max384_pair_ebs16_no-sin-pos-emb/best.pt"
)


def _place_cb(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Place ideal CB given N, CA, C using tetrahedral geometry.

    Matches Frame2ConFind/data/dataset.py _place_cb exactly.
    """
    b = ca - n
    c_ = c - ca
    a = torch.cross(b, c_, dim=-1)
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    c_n = c_ / (c_.norm(dim=-1, keepdim=True) + 1e-8)
    cb = -0.58273431 * a + 0.56802827 * b_n - 0.54067466 * c_n
    return ca + 1.522 * cb


def graph_to_f2s_coords(graph) -> tuple:
    """Convert a proteina graph to Frame2seq-format coordinates and mask.

    Args:
        graph: PyG Data object with ``coords`` [L, num_atoms, 3] (PDB ordering)
               and ``coord_mask`` [L, num_atoms].

    Returns:
        x_f2s: [L, 5, 3] tensor with N, CA, C, CB, O coordinates.
        mask: [L] boolean mask (True where CA coordinate is valid).
    """
    from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

    coords = graph.coords.clone()
    coord_mask = getattr(graph, "coord_mask", None)

    # Reorder from PDB ordering to OpenFold ordering (matches Frame2ConFind dataset code)
    idx = PDB_TO_OPENFOLD_INDEX_TENSOR.to(coords.device)
    coords = coords[:, idx, :]
    if coord_mask is not None:
        coord_mask = coord_mask[:, idx]

    # OpenFold ordering: N=0, CA=1, C=2, CB=3, O=4
    # Frame2seq wants: N=0, CA=1, C=2, CB=3, O=4 — same as first 5 OpenFold atoms
    x_f2s = coords[:, [0, 1, 2, 3, 4], :].clone()  # [L, 5, 3]

    # Build mask from CA coordinate validity
    if coord_mask is not None:
        mask = coord_mask[:, 1].bool()  # CA index in OpenFold ordering
        # Handle missing CB: place pseudo-CB for residues with valid backbone but missing CB
        cb_valid = coord_mask[:, 3].bool()  # CB index in OpenFold ordering
        missing_cb = ~cb_valid & mask
    else:
        mask = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
        missing_cb = torch.zeros_like(mask)

    if missing_cb.any():
        # N=0, CA=1, C=2 in OpenFold ordering
        cb_placed = _place_cb(coords[missing_cb, 0], coords[missing_cb, 1], coords[missing_cb, 2])
        x_f2s[missing_cb, 3, :] = cb_placed

    return x_f2s, mask


class Frame2ConFindTransformPredictor:
    """Singleton wrapper for using Frame2ConFind in proteina data transforms.

    Lazily loads the model on first invocation and caches it for subsequent calls.
    When running inside DataLoader workers, uses CPU to avoid GPU contention
    with the training process.
    """

    _instance: Optional["Frame2ConFindTransformPredictor"] = None
    _instance_key: Optional[tuple] = None

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        amp_dtype: str = "bf16",
        device: Optional[str] = None,
        compile_model: bool = False,
    ):
        self.checkpoint = str(Path(checkpoint).expanduser())
        self.amp_dtype = amp_dtype
        self.compile_model = compile_model
        self._predictor = None

        # Resolve device: use CPU in DataLoader workers, GPU in main process
        if device is not None:
            self.device = device
        elif torch.utils.data.get_worker_info() is not None:
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"
        else:
            self.device = "cpu"

    @classmethod
    def get_or_create(
        cls,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        amp_dtype: str = "bf16",
        device: Optional[str] = None,
        compile_model: bool = False,
    ) -> "Frame2ConFindTransformPredictor":
        """Return a cached predictor, creating one if needed."""
        key = (str(Path(checkpoint).expanduser()), amp_dtype, device, compile_model)
        if cls._instance is None or cls._instance_key != key:
            cls._instance = cls(
                checkpoint=checkpoint,
                amp_dtype=amp_dtype,
                device=device,
                compile_model=compile_model,
            )
            cls._instance_key = key
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        # Lazy import to avoid importing Frame2ConFind at module level
        sys.path.insert(0, "/home/ubuntu")
        from Frame2ConFind.inference.api import Frame2ConFindPredictor

        self._predictor = Frame2ConFindPredictor(
            checkpoint=self.checkpoint,
            device=self.device,
            amp_dtype=self.amp_dtype if self.device != "cpu" else "fp32",
            compile_model=self.compile_model,
        )

    @torch.no_grad()
    def predict_graph(self, graph) -> torch.Tensor:
        """Predict contact probability map from a proteina graph.

        Args:
            graph: PyG Data object with coords [L, num_atoms, 3] and coord_mask.

        Returns:
            [L, L] float32 tensor of contact probabilities in [0, 1].
        """
        self._ensure_loaded()
        x_f2s, mask = graph_to_f2s_coords(graph)
        L = x_f2s.shape[0]
        # Add batch dimension: [1, L, 5, 3]
        x_f2s = x_f2s.unsqueeze(0)
        mask = mask.unsqueeze(0)
        probs = self._predictor.predict_batch(x_f2s, mask)  # [1, L, L]
        return probs[0, :L, :L].cpu().float()
