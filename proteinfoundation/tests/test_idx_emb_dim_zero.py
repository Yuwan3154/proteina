# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Test that ProteinTransformerAF3 works when idx_emb_dim is 0 or not specified.
"""

import torch

from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3


def _make_nn_config(idx_emb_dim_value=None):
    """Build minimal nn config, optionally with idx_emb_dim."""
    cfg = {
        "token_dim": 512,
        "nlayers": 2,
        "nheads": 8,
        "residual_mha": True,
        "residual_transition": True,
        "parallel_mha_transition": False,
        "use_attn_pair_bias": True,
        "strict_feats": False,
        "feats_init_seq": ["res_seq_pdb_idx", "chain_break_per_res", "x_sc"],
        "feats_cond_seq": ["time_emb"],  # omit fold_emb to avoid CATH mapping file dependency
        "t_emb_dim": 256,
        "dim_cond": 256,  # match time_emb only
        "fold_emb_dim": 256,
        "cath_code_dir": "/tmp",
        "multilabel_mode": "sample",
        "feats_pair_repr": ["rel_seq_sep", "x_sc_pair_dists", "xt_pair_dists"],
        "feats_pair_cond": ["time_emb"],
        "xt_pair_dist_dim": 64,
        "xt_pair_dist_min": 0.1,
        "xt_pair_dist_max": 3.0,
        "x_sc_pair_dist_dim": 128,
        "x_sc_pair_dist_min": 0.1,
        "x_sc_pair_dist_max": 3.0,
        "seq_sep_dim": 127,
        "pair_repr_dim": 256,
        "update_pair_repr": False,
        "use_tri_mult": False,
        "num_registers": 0,
        "use_qkln": True,
        "num_buckets_predict_pair": None,
        "predict_coords": True,
        "contact_map_mode": False,
        "use_torch_compile": False,
    }
    if idx_emb_dim_value is not None:
        cfg["idx_emb_dim"] = idx_emb_dim_value
    return cfg


def _make_batch(b=2, n=16, device="cpu"):
    """Minimal batch for forward pass."""
    mask = torch.ones(b, n, dtype=torch.bool, device=device)
    x_t = torch.randn(b, n, 3, device=device) * 0.1
    t = torch.rand(b, device=device)
    x_sc = torch.randn(b, n, 3, device=device) * 0.1
    cath_code = [["1.10.10.10", "1", "2", "3"] for _ in range(b)]
    return {
        "x_t": x_t,
        "t": t,
        "mask": mask,
        "x_sc": x_sc,
        "cath_code": cath_code,
    }


def test_feature_factory_skips_res_seq_pdb_idx_when_idx_emb_dim_zero():
    """FeatureFactory correctly skips res_seq_pdb_idx when idx_emb_dim=0 or omitted."""
    feats = ["res_seq_pdb_idx", "chain_break_per_res", "x_sc"]
    # chain_break_per_res dim=1, x_sc dim=3. res_seq_pdb_idx dim=128 when enabled.
    dim_chain_break = 1
    dim_x_sc = 3
    dim_res_seq_pdb_idx = 128

    # idx_emb_dim=0: res_seq_pdb_idx should be filtered out
    factory_zero = FeatureFactory(
        feats=feats.copy(),
        dim_feats_out=64,
        use_ln_out=False,
        mode="seq",
        idx_emb_dim=0,
    )
    expected_dim_zero = dim_chain_break + dim_x_sc  # 4
    assert factory_zero.linear_out.in_features == expected_dim_zero

    # idx_emb_dim omitted: should default to 0, same behavior
    factory_omitted = FeatureFactory(
        feats=feats.copy(),
        dim_feats_out=64,
        use_ln_out=False,
        mode="seq",
    )
    assert factory_omitted.linear_out.in_features == expected_dim_zero

    # idx_emb_dim=128: res_seq_pdb_idx should be included
    factory_128 = FeatureFactory(
        feats=feats.copy(),
        dim_feats_out=64,
        use_ln_out=False,
        mode="seq",
        idx_emb_dim=128,
    )
    expected_dim_128 = dim_res_seq_pdb_idx + dim_chain_break + dim_x_sc  # 132
    assert factory_128.linear_out.in_features == expected_dim_128


def test_feature_factory_forward_with_idx_emb_dim_zero():
    """FeatureFactory forward produces correct output when res_seq_pdb_idx is skipped."""
    feats = ["res_seq_pdb_idx", "chain_break_per_res", "x_sc"]
    factory = FeatureFactory(
        feats=feats,
        dim_feats_out=64,
        use_ln_out=False,
        mode="seq",
        idx_emb_dim=0,
    )
    batch = {
        "x_t": torch.randn(2, 16, 3) * 0.1,
        "mask": torch.ones(2, 16, dtype=torch.bool),
        "x_sc": torch.randn(2, 16, 3) * 0.1,
    }
    out = factory(batch)
    assert out.shape == (2, 16, 64)


def test_idx_emb_dim_zero():
    """Model instantiates and runs forward when idx_emb_dim=0."""
    cfg = _make_nn_config(idx_emb_dim_value=0)
    model = ProteinTransformerAF3(**cfg)
    batch = _make_batch()
    out = model(batch)
    assert out["coords_pred"].shape == (2, 16, 3)


def test_idx_emb_dim_omitted():
    """Model instantiates and runs forward when idx_emb_dim is not specified."""
    cfg = _make_nn_config(idx_emb_dim_value=None)  # omit
    model = ProteinTransformerAF3(**cfg)
    batch = _make_batch()
    out = model(batch)
    assert out["coords_pred"].shape == (2, 16, 3)


def test_idx_emb_dim_128_still_works():
    """Model still works with idx_emb_dim=128 (existing behavior)."""
    cfg = _make_nn_config(idx_emb_dim_value=128)
    model = ProteinTransformerAF3(**cfg)
    batch = _make_batch()
    out = model(batch)
    assert out["coords_pred"].shape == (2, 16, 3)
