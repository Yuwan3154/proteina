# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Tests for ext_lig per-residue feature computation.

Covers:
- compute_ext_lig_from_df: all semantic cases + edge cases
- make_unknown_ext_lig: shape and value
- Oracle agreement: cKDTree vs brute-force
- ExtLigEmbeddingSeqFeat: with / without ext_lig in batch
- PaddingTransform: correct fill value (2) for ext_lig
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from proteinfoundation.datasets.ext_lig_utils import (
    EXT_LIG_ABSENT,
    EXT_LIG_PRESENT,
    EXT_LIG_UNKNOWN,
    WATER_RESIDUE_NAMES,
    _CA_CUTOFF,
    compute_ext_lig_from_df,
    make_unknown_ext_lig,
)
from proteinfoundation.datasets.transforms import PaddingTransform
from proteinfoundation.nn.feature_factory import ExtLigEmbeddingSeqFeat

# ---------------------------------------------------------------------------
# Helpers for constructing synthetic deposit DataFrames
# ---------------------------------------------------------------------------

_ATOM_COUNTER = 0


def _atom(
    chain_id: str,
    residue_number: int,
    residue_name: str,
    atom_name: str,
    x: float,
    y: float,
    z: float,
    insertion: str = "",
) -> dict:
    global _ATOM_COUNTER
    _ATOM_COUNTER += 1
    return {
        "chain_id": chain_id,
        "residue_number": residue_number,
        "residue_name": residue_name,
        "atom_name": atom_name,
        "x_coord": x,
        "y_coord": y,
        "z_coord": z,
        "insertion": insertion,
        "atom_number": _ATOM_COUNTER,
    }


def _make_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _resid(chain: str, resname: str, resnum: int, insertion: str = "") -> str:
    return f"{chain}:{resname}:{resnum}:{insertion}"


# ===========================================================================
# 1. Tests for make_unknown_ext_lig
# ===========================================================================


class TestMakeUnknownExtLig:
    def test_shape(self):
        t = make_unknown_ext_lig(10)
        assert t.shape == (10,)

    def test_all_unknown(self):
        t = make_unknown_ext_lig(7)
        assert (t == EXT_LIG_UNKNOWN).all()

    def test_dtype_long(self):
        t = make_unknown_ext_lig(5)
        assert t.dtype == torch.long

    def test_zero_length(self):
        t = make_unknown_ext_lig(0)
        assert t.shape == (0,)


# ===========================================================================
# 2. Tests for compute_ext_lig_from_df – semantic cases
# ===========================================================================


class TestComputeExtLigSemantics:
    """Each test uses a minimal synthetic deposit and checks ext_lig values."""

    # -----------------------------------------------------------------------
    # 2a. Other-chain protein atom inside cutoff -> present
    # -----------------------------------------------------------------------

    def test_other_chain_protein_inside_cutoff(self):
        """Chain B CA at distance 5 Å from chain A residue 1 CA -> present."""
        rows = [
            # Self chain A  (non-zero CA to avoid fill-value detection)
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            _atom("A", 1, "ALA", "N", 10.0, 1.0, 0.0),
            # Other chain B: 5 Å away
            _atom("B", 1, "GLY", "CA", 15.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        graph_resids = [_resid("A", "ALA", 1)]
        result = compute_ext_lig_from_df(df, self_chains="A", graph_residue_ids=graph_resids)
        assert result.tolist() == [EXT_LIG_PRESENT]

    # -----------------------------------------------------------------------
    # 2b. Other-chain small molecule inside cutoff -> present
    # -----------------------------------------------------------------------

    def test_other_chain_small_molecule_inside_cutoff(self):
        """Ligand in chain D at 3 Å from chain A CA -> present."""
        rows = [
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            _atom("D", 1, "ATP", "C1", 13.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_PRESENT]

    # -----------------------------------------------------------------------
    # 2c. Same-chain atom only -> absent
    # -----------------------------------------------------------------------

    def test_same_chain_atom_only(self):
        """Only atoms from self-chain A -> absent for all residues."""
        rows = [
            _atom("A", 1, "ALA", "CA", 0.0, 0.0, 0.0),
            _atom("A", 1, "ALA", "CB", 1.5, 0.0, 0.0),
            _atom("A", 2, "GLY", "CA", 3.8, 0.0, 0.0),
        ]
        df = _make_df(rows)
        graph_resids = [_resid("A", "ALA", 1), _resid("A", "GLY", 2)]
        result = compute_ext_lig_from_df(df, "A", graph_resids)
        assert result.tolist() == [EXT_LIG_ABSENT, EXT_LIG_ABSENT]

    # -----------------------------------------------------------------------
    # 2d. Water-only neighbors -> absent
    # -----------------------------------------------------------------------

    def test_water_only_neighbors(self):
        """All external atoms are water residues -> absent."""
        water_rows = [
            _atom("W", i, wname, "O", float(i), 0.0, 0.0)
            for i, wname in enumerate(sorted(WATER_RESIDUE_NAMES), 1)
        ]
        self_row = [_atom("A", 1, "ALA", "CA", 0.0, 0.0, 0.0)]
        df = _make_df(self_row + water_rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_ABSENT]

    # -----------------------------------------------------------------------
    # 2e. Boundary cases at exactly 8 Å and 8.01 Å
    # -----------------------------------------------------------------------

    def test_external_atom_at_cutoff_inside(self):
        """External atom exactly at 8.0 Å should be within cutoff -> present."""
        ca_x = 10.0
        rows = [
            _atom("A", 1, "ALA", "CA", ca_x, 0.0, 0.0),
            _atom("B", 1, "GLY", "CA", ca_x + _CA_CUTOFF, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_PRESENT]

    def test_external_atom_just_outside_cutoff(self):
        """External atom at 8.01 Å should be outside cutoff -> absent."""
        ca_x = 10.0
        rows = [
            _atom("A", 1, "ALA", "CA", ca_x, 0.0, 0.0),
            _atom("B", 1, "GLY", "CA", ca_x + _CA_CUTOFF + 0.01, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_ABSENT]

    # -----------------------------------------------------------------------
    # 2f. Multiple selected chains, asymmetric external context
    # -----------------------------------------------------------------------

    def test_multiple_self_chains_asymmetric(self):
        """
        Self chains A and B.  External chain C is close only to residue in A.
        Residue in B should be absent.
        """
        rows = [
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            _atom("B", 1, "GLY", "CA", 100.0, 0.0, 0.0),
            _atom("C", 1, "ATP", "C1", 15.0, 0.0, 0.0),  # 5 Å from A1, >79 Å from B1
        ]
        df = _make_df(rows)
        graph_resids = [_resid("A", "ALA", 1), _resid("B", "GLY", 1)]
        result = compute_ext_lig_from_df(df, ["A", "B"], graph_resids)
        assert result[0].item() == EXT_LIG_PRESENT
        assert result[1].item() == EXT_LIG_ABSENT

    # -----------------------------------------------------------------------
    # 2g. Missing CA (fill_value_coords sentinel) -> unknown
    # -----------------------------------------------------------------------

    def test_missing_ca_is_unknown(self):
        """Residue without a CA row should get EXT_LIG_UNKNOWN."""
        rows = [
            # Only N atom; no CA
            _atom("A", 1, "ALA", "N", 0.0, 0.0, 0.0),
            _atom("B", 1, "GLY", "CA", 3.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_UNKNOWN]

    # -----------------------------------------------------------------------
    # 2h. CA at the fill_value_coords sentinel -> unknown
    # -----------------------------------------------------------------------

    def test_fill_value_ca_is_unknown(self):
        """CA at near-zero sentinel coordinates produced by graphein -> unknown."""
        rows = [
            _atom("A", 1, "ALA", "CA", 1e-6, 0.0, 0.0),  # near-zero sentinel
            _atom("B", 1, "GLY", "CA", 3.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_UNKNOWN]

    # -----------------------------------------------------------------------
    # 2i. Insertion-code residue alignment correctness
    # -----------------------------------------------------------------------

    def test_insertion_code_alignment(self):
        """Residues with insertion codes must match the graph residue_id exactly."""
        rows = [
            _atom("A", 10, "ALA", "CA", 10.0, 0.0, 0.0, insertion=""),   # A:ALA:10:
            _atom("A", 10, "GLY", "CA", 10.5, 0.0, 0.0, insertion="A"),  # A:GLY:10:A
            _atom("B", 1, "LIG", "C1", 14.0, 0.0, 0.0),                   # 3.5 Å from GLY:10:A
        ]
        df = _make_df(rows)
        # Only include the insertion-code residue
        graph_resids = [_resid("A", "GLY", 10, "A")]
        result = compute_ext_lig_from_df(df, "A", graph_resids)
        assert result.tolist() == [EXT_LIG_PRESENT]

    # -----------------------------------------------------------------------
    # 2j. Empty external atom pool -> all absent
    # -----------------------------------------------------------------------

    def test_empty_external_pool(self):
        """No external atoms in deposit -> all absent."""
        rows = [
            _atom("A", 1, "ALA", "CA", 0.0, 0.0, 0.0),
            _atom("A", 2, "GLY", "CA", 3.8, 0.0, 0.0),
        ]
        df = _make_df(rows)
        graph_resids = [_resid("A", "ALA", 1), _resid("A", "GLY", 2)]
        result = compute_ext_lig_from_df(df, "A", graph_resids)
        assert result.tolist() == [EXT_LIG_ABSENT, EXT_LIG_ABSENT]

    # -----------------------------------------------------------------------
    # 2k. Residue ID not in graph but in df (extra residues filtered by graph list)
    # -----------------------------------------------------------------------

    def test_graph_resid_filtering(self):
        """Only the residues in graph_residue_ids are returned, ordered correctly."""
        rows = [
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            _atom("A", 2, "GLY", "CA", 50.0, 0.0, 0.0),
            _atom("B", 1, "LIG", "C1", 14.0, 0.0, 0.0),  # 4 Å from residue 1
        ]
        df = _make_df(rows)
        # Only request residue 1; residue 2 is far and not in list
        graph_resids = [_resid("A", "ALA", 1)]
        result = compute_ext_lig_from_df(df, "A", graph_resids)
        assert result.shape == (1,)
        assert result.tolist() == [EXT_LIG_PRESENT]

    # -----------------------------------------------------------------------
    # 2l. Chain B atom outside cutoff -> absent
    # -----------------------------------------------------------------------

    def test_other_chain_outside_cutoff(self):
        rows = [
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            _atom("B", 1, "GLY", "CA", 60.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_ABSENT]

    # -----------------------------------------------------------------------
    # 2m. Nucleic acid chain counts as external
    # -----------------------------------------------------------------------

    def test_nucleic_acid_chain_is_external(self):
        rows = [
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            _atom("C", 1, "DA", "C1'", 15.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_PRESENT]

    # -----------------------------------------------------------------------
    # 2n. Ion chain counts as external
    # -----------------------------------------------------------------------

    def test_ion_chain_is_external(self):
        rows = [
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            _atom("E", 1, "ZN", "ZN", 13.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        result = compute_ext_lig_from_df(df, "A", [_resid("A", "ALA", 1)])
        assert result.tolist() == [EXT_LIG_PRESENT]

    # -----------------------------------------------------------------------
    # 2o. Longer chain, mixed present/absent/unknown
    # -----------------------------------------------------------------------

    def test_mixed_labels(self):
        """Three residues: close to external, far from external, no CA."""
        rows = [
            # Residue 1: CA close to external
            _atom("A", 1, "ALA", "CA", 10.0, 0.0, 0.0),
            # Residue 2: CA far from external
            _atom("A", 2, "GLY", "CA", 100.0, 0.0, 0.0),
            # Residue 3: no CA row
            _atom("A", 3, "SER", "N", 200.0, 0.0, 0.0),
            # External ligand 5 Å from residue 1 only
            _atom("B", 1, "LIG", "C1", 15.0, 0.0, 0.0),
        ]
        df = _make_df(rows)
        graph_resids = [
            _resid("A", "ALA", 1),
            _resid("A", "GLY", 2),
            _resid("A", "SER", 3),
        ]
        result = compute_ext_lig_from_df(df, "A", graph_resids)
        assert result[0].item() == EXT_LIG_PRESENT
        assert result[1].item() == EXT_LIG_ABSENT
        assert result[2].item() == EXT_LIG_UNKNOWN


# ===========================================================================
# 3. Oracle agreement: cKDTree vs brute-force
# ===========================================================================


def _brute_force_ext_lig(
    full_df: pd.DataFrame,
    self_chains,
    graph_residue_ids,
    cutoff: float = _CA_CUTOFF,
) -> torch.Tensor:
    """Reference brute-force implementation using pairwise numpy distances."""
    L = len(graph_residue_ids)
    ext_lig = torch.full((L,), EXT_LIG_ABSENT, dtype=torch.long)

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

    self_atoms = full_df[full_df["chain_id"].isin(self_chain_set)]
    self_ca = self_atoms[self_atoms["atom_name"] == "CA"]

    ca_map = {}
    for _, row in self_ca.iterrows():
        chain = row["chain_id"]
        resname = row["residue_name"]
        resnum = str(int(row["residue_number"]))
        insertion = row.get("insertion", "")
        if pd.isna(insertion):
            insertion = ""
        key = f"{chain}:{resname}:{resnum}:{insertion}"
        ca_map[key] = np.array(
            [row["x_coord"], row["y_coord"], row["z_coord"]], dtype=np.float64
        )

    for i, resid in enumerate(graph_residue_ids):
        ca_coord = ca_map.get(resid)
        if ca_coord is None:
            ext_lig[i] = EXT_LIG_UNKNOWN
            continue
        if np.any(np.abs(ca_coord) < 1e-4) and np.allclose(ca_coord, 0.0, atol=1e-4):
            ext_lig[i] = EXT_LIG_UNKNOWN
            continue
        dists = np.linalg.norm(ext_coords - ca_coord, axis=1)
        if np.any(dists <= cutoff):
            ext_lig[i] = EXT_LIG_PRESENT

    return ext_lig


def _make_random_deposit(
    n_self: int = 30,
    n_ext: int = 50,
    seed: int = 42,
    n_extra_chains: int = 2,
) -> tuple:
    """Build a random synthetic deposit with self chain A and external chains."""
    rng = np.random.default_rng(seed)
    rows = []
    graph_resids = []

    for i in range(n_self):
        ca_xyz = rng.uniform(-30, 30, 3)
        resid = _resid("A", "ALA", i + 1)
        rows.append(_atom("A", i + 1, "ALA", "CA", *ca_xyz))
        graph_resids.append(resid)

    for chain_idx in range(n_extra_chains):
        chain = chr(ord("B") + chain_idx)
        for j in range(n_ext // n_extra_chains):
            xyz = rng.uniform(-30, 30, 3)
            rows.append(_atom(chain, j + 1, "LIG", "C1", *xyz))

    return _make_df(rows), graph_resids


class TestOracleAgreement:
    """Compare cKDTree implementation to brute-force on a range of synthetic deposits."""

    @pytest.mark.parametrize(
        "n_self, n_ext, seed",
        [
            (10, 20, 0),
            (30, 50, 1),
            (100, 200, 2),
            (5, 0, 3),    # no external atoms
            (1, 500, 4),  # single residue, many external
            (50, 50, 5),
        ],
    )
    def test_exact_agreement(self, n_self, n_ext, seed):
        df, graph_resids = _make_random_deposit(n_self=n_self, n_ext=n_ext, seed=seed)
        production = compute_ext_lig_from_df(df, "A", graph_resids)
        oracle = _brute_force_ext_lig(df, "A", graph_resids)
        assert torch.equal(production, oracle), (
            f"Mismatch (n_self={n_self}, n_ext={n_ext}, seed={seed}):\n"
            f"  production={production.tolist()}\n"
            f"  oracle    ={oracle.tolist()}"
        )

    def test_agreement_with_water_mixed_in(self):
        """Oracle and production must agree when water atoms are mixed in."""
        rng = np.random.default_rng(99)
        rows = []
        graph_resids = []
        for i in range(20):
            ca = rng.uniform(-20, 20, 3)
            rows.append(_atom("A", i + 1, "ALA", "CA", *ca))
            graph_resids.append(_resid("A", "ALA", i + 1))
        # Add non-water external
        for j in range(15):
            xyz = rng.uniform(-20, 20, 3)
            rows.append(_atom("B", j + 1, "LIG", "C1", *xyz))
        # Add water external (should be ignored)
        for j in range(30):
            xyz = rng.uniform(-5, 5, 3)
            rows.append(_atom("W", j + 1, "HOH", "O", *xyz))

        df = _make_df(rows)
        production = compute_ext_lig_from_df(df, "A", graph_resids)
        oracle = _brute_force_ext_lig(df, "A", graph_resids)
        assert torch.equal(production, oracle)

    def test_agreement_multiple_self_chains(self):
        """Agreement when self_chains is a list."""
        rng = np.random.default_rng(7)
        rows = []
        graph_resids = []
        for chain in ("A", "B"):
            for i in range(15):
                ca = rng.uniform(-20, 20, 3)
                rows.append(_atom(chain, i + 1, "ALA", "CA", *ca))
                graph_resids.append(_resid(chain, "ALA", i + 1))
        # External chain C
        for j in range(20):
            xyz = rng.uniform(-20, 20, 3)
            rows.append(_atom("C", j + 1, "LIG", "C1", *xyz))

        df = _make_df(rows)
        production = compute_ext_lig_from_df(df, ["A", "B"], graph_resids)
        oracle = _brute_force_ext_lig(df, ["A", "B"], graph_resids)
        assert torch.equal(production, oracle)


# ===========================================================================
# 4. ExtLigEmbeddingSeqFeat tests
# ===========================================================================


class TestExtLigEmbeddingSeqFeat:
    def _make_feat(self, dim: int = 16) -> ExtLigEmbeddingSeqFeat:
        return ExtLigEmbeddingSeqFeat(ext_lig_emb_dim=dim)

    def test_output_shape_with_ext_lig(self):
        feat = self._make_feat(16)
        b, n = 2, 12
        batch = {
            "ext_lig": torch.randint(0, 3, (b, n)),
            "x_t": torch.randn(b, n, 3),
        }
        out = feat(batch)
        assert out.shape == (b, n, 16)

    def test_output_shape_without_ext_lig(self):
        """When ext_lig absent, defaults to all-unknown and returns correct shape."""
        feat = self._make_feat(8)
        b, n = 3, 20
        batch = {"x_t": torch.randn(b, n, 3)}
        out = feat(batch)
        assert out.shape == (b, n, 8)

    def test_absent_values_use_embedding_row_0(self):
        feat = self._make_feat(4)
        # Force absent (0) for all
        batch = {
            "ext_lig": torch.zeros(2, 5, dtype=torch.long),
            "x_t": torch.randn(2, 5, 3),
        }
        out = feat(batch)
        expected = feat.embedding(torch.zeros(2, 5, dtype=torch.long))
        assert torch.allclose(out, expected)

    def test_present_values_use_embedding_row_1(self):
        feat = self._make_feat(4)
        batch = {
            "ext_lig": torch.ones(2, 5, dtype=torch.long),
            "x_t": torch.randn(2, 5, 3),
        }
        out = feat(batch)
        expected = feat.embedding(torch.ones(2, 5, dtype=torch.long))
        assert torch.allclose(out, expected)

    def test_unknown_values_use_embedding_row_2(self):
        feat = self._make_feat(4)
        batch = {
            "ext_lig": torch.full((2, 5), 2, dtype=torch.long),
            "x_t": torch.randn(2, 5, 3),
        }
        out = feat(batch)
        expected = feat.embedding(torch.full((2, 5), 2, dtype=torch.long))
        assert torch.allclose(out, expected)

    def test_default_unknown_matches_explicit_unknown(self):
        """When ext_lig is absent from batch, output should equal all-unknown embedding."""
        feat = self._make_feat(8)
        b, n = 2, 10
        x_t = torch.randn(b, n, 3)
        batch_no_ext_lig = {"x_t": x_t}
        batch_all_unknown = {
            "ext_lig": torch.full((b, n), 2, dtype=torch.long),
            "x_t": x_t,
        }
        out_default = feat(batch_no_ext_lig)
        out_explicit = feat(batch_all_unknown)
        assert torch.allclose(out_default, out_explicit)

    def test_embedding_dimension_matches_init(self):
        for dim in [4, 16, 64]:
            feat = self._make_feat(dim)
            assert feat.embedding.embedding_dim == dim
            assert feat.embedding.num_embeddings == 3


# ===========================================================================
# 5. PaddingTransform: ext_lig fill value tests
# ===========================================================================


def _make_graph_with_ext_lig(n: int, extra_fields: dict | None = None) -> Data:
    """Build a minimal PyG Data object with an ext_lig tensor of length n."""
    d = Data(
        coords=torch.randn(n, 3),
        mask=torch.ones(n, dtype=torch.bool),
        ext_lig=torch.randint(0, 2, (n,)),
    )
    if extra_fields:
        for k, v in extra_fields.items():
            d[k] = v
    return d


class TestPaddingTransformExtLig:
    def test_ext_lig_padded_with_unknown(self):
        """PaddingTransform should pad ext_lig with value 2 (unknown)."""
        n, max_size = 5, 10
        graph = _make_graph_with_ext_lig(n)
        transform = PaddingTransform(max_size=max_size)
        result = transform(graph)
        assert result.ext_lig.shape[0] == max_size
        assert (result.ext_lig[n:] == EXT_LIG_UNKNOWN).all(), (
            "Padded positions should be filled with EXT_LIG_UNKNOWN (2)"
        )

    def test_ext_lig_original_values_preserved(self):
        """Original ext_lig values must not be modified by padding."""
        n, max_size = 6, 12
        graph = _make_graph_with_ext_lig(n)
        original = graph.ext_lig.clone()
        transform = PaddingTransform(max_size=max_size)
        result = transform(graph)
        assert torch.equal(result.ext_lig[:n], original)

    def test_ext_lig_not_padded_when_full(self):
        """No padding needed when graph length == max_size."""
        n = 8
        graph = _make_graph_with_ext_lig(n)
        original = graph.ext_lig.clone()
        transform = PaddingTransform(max_size=n)
        result = transform(graph)
        assert result.ext_lig.shape[0] == n
        assert torch.equal(result.ext_lig, original)

    def test_ext_lig_truncated_when_exceeds_max(self):
        """ext_lig longer than max_size should be truncated, not error out."""
        n, max_size = 20, 10
        graph = _make_graph_with_ext_lig(n)
        transform = PaddingTransform(max_size=max_size)
        result = transform(graph)
        assert result.ext_lig.shape[0] == max_size

    def test_dssp_target_still_uses_minus_one(self):
        """PaddingTransform should use fill=-1 for dssp_target (regression check)."""
        n, max_size = 5, 10
        graph = Data(
            coords=torch.randn(n, 3),
            mask=torch.ones(n, dtype=torch.bool),
            ext_lig=torch.zeros(n, dtype=torch.long),
            dssp_target=torch.randint(0, 3, (n,)),
        )
        transform = PaddingTransform(max_size=max_size)
        result = transform(graph)
        assert (result.dssp_target[n:] == -1).all()
        assert (result.ext_lig[n:] == EXT_LIG_UNKNOWN).all()

    def test_ext_lig_fill_is_not_zero(self):
        """Make sure ext_lig is NOT filled with 0 (absent) to avoid silent bugs."""
        n, max_size = 4, 8
        # Set all original ext_lig to 0 (absent) so we can distinguish pad vs data
        graph = Data(
            coords=torch.randn(n, 3),
            mask=torch.ones(n, dtype=torch.bool),
            ext_lig=torch.zeros(n, dtype=torch.long),
        )
        transform = PaddingTransform(max_size=max_size)
        result = transform(graph)
        # Original positions: 0 (absent); padded positions: 2 (unknown), not 0
        assert (result.ext_lig[:n] == 0).all()
        assert (result.ext_lig[n:] == EXT_LIG_UNKNOWN).all()
        assert not (result.ext_lig[n:] == 0).all()
