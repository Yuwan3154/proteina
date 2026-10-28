"""Unit tests for Task 2: TMalign/USalign CA-based SSE classifier (`dssp_def="tmalign"` and `"tmalign_smooth"`)."""

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_LOOP,
    DSSP_STRAND,
    dssp_to_runs,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.utils.dssp_utils import (
    compute_dssp_target,
    compute_tmalign_dssp,
    smooth_tmalign_dssp,
)


def usalign_reference_make_sec(ca_np: np.ndarray) -> np.ndarray:
    """Direct scalar transcription of USalign C++ `make_sec(double **x, int len, char *sec)` -> {0, 1, 2}."""
    L = ca_np.shape[0]
    sec = np.zeros(L, dtype=np.int64)
    for i in range(L):
        if i < 2 or i + 2 >= L:
            sec[i] = 0
            continue
        j1, j2, j3, j4, j5 = i - 2, i - 1, i, i + 1, i + 2
        d13 = float(np.linalg.norm(ca_np[j1] - ca_np[j3]))
        d14 = float(np.linalg.norm(ca_np[j1] - ca_np[j4]))
        d15 = float(np.linalg.norm(ca_np[j1] - ca_np[j5]))
        d24 = float(np.linalg.norm(ca_np[j2] - ca_np[j4]))
        d25 = float(np.linalg.norm(ca_np[j2] - ca_np[j5]))
        d35 = float(np.linalg.norm(ca_np[j3] - ca_np[j5]))
        if (
            abs(d15 - 6.37) < 2.1
            and abs(d14 - 5.18) < 2.1
            and abs(d25 - 5.18) < 2.1
            and abs(d13 - 5.45) < 2.1
            and abs(d24 - 5.45) < 2.1
            and abs(d35 - 5.45) < 2.1
        ):
            sec[i] = 1
        elif (
            abs(d15 - 13.0) < 1.42
            and abs(d14 - 10.4) < 1.42
            and abs(d25 - 10.4) < 1.42
            and abs(d13 - 6.1) < 1.42
            and abs(d24 - 6.1) < 1.42
            and abs(d35 - 6.1) < 1.42
        ):
            sec[i] = 2
        else:
            sec[i] = 0
    return sec


def build_ideal_helix_atom37(n_res: int) -> torch.Tensor:
    """Build ideal alpha-helix backbone (N, CA, C, O in atom37 slots 0, 1, 2, 4)."""
    coords = torch.zeros((1, n_res, 37, 3), dtype=torch.float32)
    twist = math.radians(100.0)
    for i in range(n_res):
        theta = i * twist
        z = i * 1.5
        ca = torch.tensor([2.3 * math.cos(theta), 2.3 * math.sin(theta), z])
        n = torch.tensor([2.0 * math.cos(theta - 0.4), 2.0 * math.sin(theta - 0.4), z - 0.5])
        c = torch.tensor([2.0 * math.cos(theta + 0.4), 2.0 * math.sin(theta + 0.4), z + 0.5])
        o = torch.tensor([2.1 * math.cos(theta + 0.45), 2.1 * math.sin(theta + 0.45), z + 1.7])
        coords[0, i, 0] = n
        coords[0, i, 1] = ca
        coords[0, i, 2] = c
        coords[0, i, 4] = o
    return coords


def build_ideal_strand_ca(n_res: int) -> torch.Tensor:
    """Build ideal beta-strand CA trace (rise ~3.3 A along x, alternating 0.95 A along y)."""
    ca = torch.zeros((1, n_res, 3), dtype=torch.float32)
    for i in range(n_res):
        ca[0, i, 0] = i * 3.25
        ca[0, i, 1] = 0.95 if (i % 2 == 0) else -0.95
        ca[0, i, 2] = 0.0
    return ca


def test_helix_and_strand_classification():
    helix_coords = build_ideal_helix_atom37(20)
    mask = torch.ones((1, 20), dtype=torch.bool)
    dssp_tm = compute_dssp_target(helix_coords, mask, coord_layout="atom37", dssp_def="tmalign")
    assert dssp_tm is not None
    assert torch.all(dssp_tm[0, 2:-2] == DSSP_HELIX), f"Expected all interior helix, got {dssp_tm[0]}"
    assert torch.all(dssp_tm[0, :2] == DSSP_LOOP) and torch.all(dssp_tm[0, -2:] == DSSP_LOOP)

    strand_ca = build_ideal_strand_ca(16)
    dssp_e = compute_tmalign_dssp(strand_ca, torch.ones((1, 16), dtype=torch.bool), smooth=False)
    assert torch.all(dssp_e[0, 2:-2] == DSSP_STRAND), f"Expected all interior strand, got {dssp_e[0]}"


def test_usalign_scalar_equivalence_and_mask():
    rng = np.random.default_rng(42)
    for trial in range(30):
        L = int(rng.integers(8, 80))
        helix = build_ideal_helix_atom37(L)[0, :, 1, :].numpy()
        strand = build_ideal_strand_ca(L)[0].numpy()
        alpha = rng.uniform(0.0, 1.0)
        ca_np = alpha * helix + (1.0 - alpha) * strand + rng.normal(0.0, 0.8, size=(L, 3))
        ref = usalign_reference_make_sec(ca_np)
        got = compute_tmalign_dssp(
            torch.from_numpy(ca_np).float()[None],
            torch.ones((1, L), dtype=torch.bool),
            smooth=False,
        )[0].numpy()
        assert np.array_equal(ref, got), f"Trial {trial}: mismatch ref={ref} vs got={got}"

    ca_t = build_ideal_helix_atom37(15)[:, :, 1, :]
    vmask = torch.ones((1, 15), dtype=torch.bool)
    vmask[0, 7] = False
    out = compute_tmalign_dssp(ca_t, vmask, smooth=False)[0]
    assert int(out[7].item()) == -1
    for idx in (5, 6, 8, 9):
        assert int(out[idx].item()) == 0, f"Expected loop at window neighbor {idx}, got {out[idx]}"


def test_smooth_passes():
    raw = torch.tensor([[0, 0, 1, 0, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.long)
    sm = smooth_tmalign_dssp(raw)
    expected = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    assert torch.equal(sm, expected), f"Smooth mismatch: got {sm[0].tolist()} vs {expected[0].tolist()}"


def test_robustness_under_backbone_distortion():
    torch.manual_seed(7)
    coords = build_ideal_helix_atom37(24)
    perturbed = coords.clone()
    perturbed[:, 8:14, 4, :] += torch.randn_like(perturbed[:, 8:14, 4, :]) * 1.5
    perturbed[:, :, 1, :] += torch.randn_like(perturbed[:, :, 1, :]) * 0.20
    mask = torch.ones((1, 24), dtype=torch.bool)

    dssp_hb = compute_dssp_target(perturbed, mask, coord_layout="atom37", dssp_def="h_bond")[0]
    dssp_tm = compute_dssp_target(perturbed, mask, coord_layout="atom37", dssp_def="tmalign_smooth")[0]

    runs_hb = [r for r in dssp_to_runs(dssp_hb, min_len=1) if r[0] == DSSP_HELIX]
    runs_tm = [r for r in dssp_to_runs(dssp_tm, min_len=1) if r[0] == DSSP_HELIX]
    n_h_hb = int((dssp_hb == DSSP_HELIX).sum().item())
    n_h_tm = int((dssp_tm == DSSP_HELIX).sum().item())
    assert n_h_tm > n_h_hb, f"Expected tmalign to retain more helix residues than pydssp ({n_h_tm} vs {n_h_hb})"
    assert len(runs_tm) == 1, f"Expected 1 contiguous helix run from tmalign, got {runs_tm}"

    cm = (torch.cdist(perturbed[0, :, 1, :], perturbed[0, :, 1, :]) < 8.0).float()
    all_runs = dssp_to_runs(dssp_tm, min_len=1)
    ref, keep = sse_contact_reference(cm, all_runs, keep_types=(DSSP_HELIX, DSSP_STRAND))
    struct_feat = sse_structural_pair_features(cm, perturbed[0], torch.ones((24, 37), dtype=torch.bool), all_runs, keep)
    assert ref.shape == (len(keep), len(keep))
    assert struct_feat.shape[:2] == (len(keep), len(keep))


if __name__ == "__main__":
    test_helix_and_strand_classification()
    test_usalign_scalar_equivalence_and_mask()
    test_smooth_passes()
    test_robustness_under_backbone_distortion()
    print("ALL TMALIGN DSSP TESTS PASSED")
