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
    # Alpha-helix parameters: radius 2.3 A, rise 1.5 A/res, twist 100 deg/res
    twist = math.radians(100.0)
    for i in range(n_res):
        theta = i * twist
        z = i * 1.5
        # CA at (2.3 cos theta, 2.3 sin theta, z)
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
    # Interior residues 2..17 must all be classified as helix (1)
    assert torch.all(dssp_tm[0, 2:-2] == DSSP_HELIX), f"Expected all interior helix, got {dssp_tm[0]}"
    assert torch.all(dssp_tm[0, :2] == DSSP_LOOP) and torch.all(dssp_tm[0, -2:] == DSSP_LOOP)

    strand_ca = build_ideal_strand_ca(16)
    dssp_e = compute_tmalign_dssp(strand_ca, torch.ones((1, 16), dtype=torch.bool), smooth=False)
    assert torch.all(dssp_e[0, 2:-2] == DSSP_STRAND), f"Expected all interior strand, got {dssp_e[0]}"


def test_usalign_scalar_equivalence_and_mask():
    rng = np.random.default_rng(42)
    for trial in range(30):
        L = int(rng.integers(8, 80))
        # Mix of helix, strand, and noisy segments
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

    # Verify invalid mask propagation (-1 on invalid positions and window neighbors)
    ca_t = build_ideal_helix_atom37(15)[:, :, 1, :]
    vmask = torch.ones((1, 15), dtype=torch.bool)
    vmask[0, 7] = False
    out = compute_tmalign_dssp(ca_t, vmask, smooth=False)[0]
    assert int(out[7].item()) == -1
    # Window [i-2..i+2] overlaps index 7 for i in {5, 6, 8, 9} -> must fall back to loop (0)
    for idx in (5, 6, 8, 9):
        assert int(out[idx].item()) == 0, f"Expected loop at window neighbor {idx}, got {out[idx]}"


def test_smooth_passes():
    # Pass 1 (--x-- -> -----), Pass 2 (--xx-- -> ------), Pass 3 (x-x -> xxx)
    raw = torch.tensor([[0, 0, 1, 0, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.long)
    sm = smooth_tmalign_dssp(raw)
    expected = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    assert torch.equal(sm, expected), f"Smooth mismatch: got {sm[0].tolist()} vs {expected[0].tolist()}"


def test_robustness_under_backbone_distortion():
    """Verify that under unrelaxed backbone O-atom / CA distortion, tmalign retains the helix while pydssp fragments."""
    torch.manual_seed(7)
    coords = build_ideal_helix_atom37(24)
    # Perturb carbonyl oxygen O (slot 4) by 1.2 A and CA by 0.25 A (typical unrelaxed c2c / partial-diffusion noise)
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

    # Verify downstream sse_contact_reference + sse_structural_pair_features execute cleanly
    cm = (torch.cdist(perturbed[0, :, 1, :], perturbed[0, :, 1, :]) < 8.0).float()
    all_runs = dssp_to_runs(dssp_tm, min_len=1)
    ref, keep = sse_contact_reference(cm, all_runs, keep_types=(DSSP_HELIX, DSSP_STRAND))
    struct_feat = sse_structural_pair_features(cm, perturbed[0], torch.ones((24, 37), dtype=torch.bool), all_runs, keep)
    assert ref.shape == (len(keep), len(keep))
    assert struct_feat.shape[:2] == (len(keep), len(keep))


def test_topology_reference_transform_tmalign_smooth():
    """Verify TopologyReferenceTransform(dssp_def='tmalign_smooth') projects runs, pair features, and align targets."""
    import tempfile
    from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES, STRUCTURAL_PAIR_FEATURES
    from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform

    # Create a fragmented synthetic index row: loop(2), H(5), loop(1), H(6), loop(2), H(1), loop(2), E(4), loop(2) -> L=25
    # Old elements (H/E): 0=H(5) [2:7], 1=H(6) [8:14], 2=H(1) [16:17], 3=E(4) [19:23] (4 elements)
    # After smooth_tmalign_dssp: H(5)+loop(1)+H(6) merges into H(12) [2:14]; H(1) [16:17] is erased to loop; E(4) [19:23] stays!
    # So new elements: 0=H(12) [2:14], 1=E(4) [19:23] (2 elements!)
    runs = [(0, 2), (1, 5), (0, 1), (1, 6), (0, 2), (1, 1), (0, 2), (2, 4), (0, 2)]
    old_he = torch.tensor(
        [[1, 1, 0, 1],
         [1, 1, 0, 1],
         [0, 0, 1, 0],
         [1, 1, 0, 1]], dtype=torch.uint8
    )
    n_s = len(STRUCTURAL_PAIR_FEATURES)
    old_struct = torch.ones((4, 4, n_s), dtype=torch.float16) * 0.5
    # Align target of length 25 pointing to old elements {0, 1, 2, 3, -1}
    align_raw = torch.full((25,), -1, dtype=torch.int16)
    align_raw[2:7] = 0
    align_raw[8:14] = 1
    align_raw[16:17] = 2   # junk 1-res helix -> should remap to -1!
    align_raw[19:23] = 3   # strand -> should remap to new element 1!

    with tempfile.TemporaryDirectory() as td:
        idx_path = os.path.join(td, "synth_idx.pt")
        torch.save({
            "ids": ["1abc_A", "1abc_A@rw100#0"],
            "cluster_of": torch.tensor([0, 0], dtype=torch.int32),
            "members_flat": torch.tensor([1], dtype=torch.int32),
            "members_offset": torch.tensor([0, 1], dtype=torch.int64),
            "seq_hash": torch.tensor([123, 123], dtype=torch.int64),
            "runs_flat": torch.tensor(runs + runs, dtype=torch.int16),
            "runs_offset": torch.tensor([0, len(runs), 2 * len(runs)], dtype=torch.int64),
            "he_offset": torch.tensor([0, 16, 32], dtype=torch.int64),
            "he_size": torch.tensor([4, 4], dtype=torch.int16),
            "he_flat": torch.cat([old_he.flatten(), old_he.flatten()]),
            "feat_offset": torch.tensor([0, 4 * 4 * n_s, 2 * 4 * 4 * n_s], dtype=torch.int64),
            "feat_flat": torch.cat([old_struct.flatten(), old_struct.flatten()]),
            "align_offset": torch.tensor([0, 0, 25], dtype=torch.int64),
            "align_flat": align_raw,
            "row_tm": torch.tensor([1.0, 0.7], dtype=torch.float16),
            "row_is_native": torch.tensor([True, False], dtype=torch.bool),
            "row_rewind": torch.tensor([0, 100], dtype=torch.int16),
            "row_usalign_tm": torch.tensor([1.0, 0.7], dtype=torch.float16),
            "pair_feature_names": ["contact"] + list(STRUCTURAL_PAIR_FEATURES) + ["rel_seq_pos"],
            "structural_feature_names": list(STRUCTURAL_PAIR_FEATURES),
            "pair_feature_mean": torch.zeros(N_PAIR_FEATURES, dtype=torch.float32),
            "pair_feature_std": torch.ones(N_PAIR_FEATURES, dtype=torch.float32),
            "elem_feature_names": ["length"],
            "elem_feature_mean": torch.tensor(0.0, dtype=torch.float32),
            "elem_feature_std": torch.tensor(1.0, dtype=torch.float32),
            "min_len": 1,
            "contact_threshold": 0.5,
            "contact_def": "ConFind via Frame2ConFind, prob >= 0.01 (native: stored contact_map_confind)",
            "dssp_def": "native: stored dssp_target; templates: pydssp; no donor mask",
            "reference_source": "synthetic",
            "element_types": [1, 2],
            "loop_axis": None,
            "template_gate": "none",
        }, idx_path)

        tf_smooth = TopologyReferenceTransform(
            index_path=idx_path,
            reference_source="synthetic",
            tm_range=(0.5, 0.9),
            mutate_prob=0.0,
            sigma_frac=0.0,
            drop_prob=0.0,
            dssp_def="tmalign_smooth",
        )
        tf_smooth._ensure_loaded()
        new_runs, new_he, new_struct, old_to_new, _ = tf_smooth._resolve_row_reference_inputs(1)
        assert int(new_he.shape[0]) == 2, f"Expected 2 merged elements after tmalign_smooth, got {new_he.shape}"
        assert new_struct.shape == (2, 2, n_s)
        assert old_to_new.tolist() == [0, 0, -1, 1], f"Unexpected old_to_new map: {old_to_new.tolist()}"
        align_mapped = tf_smooth._align_for(1, length=25, n_he=2, old_to_new_elem=old_to_new)
        assert torch.all(align_mapped[2:7] == 0) and torch.all(align_mapped[8:14] == 0)
        assert int(align_mapped[16].item()) == -1
        assert torch.all(align_mapped[19:23] == 1)


if __name__ == "__main__":
    test_helix_and_strand_classification()
    test_usalign_scalar_equivalence_and_mask()
    test_smooth_passes()
    test_robustness_under_backbone_distortion()
    test_topology_reference_transform_tmalign_smooth()
    print("ALL TMALIGN DSSP TESTS PASSED")
