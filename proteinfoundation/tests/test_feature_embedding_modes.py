# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Tests for IndividualFeatureFactory and feature_embedding_mode.
"""

import torch

from proteinfoundation.nn.feature_factory import FeatureFactory, IndividualFeatureFactory
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3


def _make_seq_batch(b=2, n=16, device="cpu"):
    """Minimal batch for seq feature forward."""
    return {
        "x_t": torch.randn(b, n, 3, device=device) * 0.1,
        "mask": torch.ones(b, n, dtype=torch.bool, device=device),
        "t": torch.rand(b, device=device),
        "x_sc": torch.randn(b, n, 3, device=device) * 0.1,
    }


def _make_pair_batch(b=2, n=16, device="cpu"):
    """Minimal batch for pair feature forward."""
    return {
        "x_t": torch.randn(b, n, 3, device=device) * 0.1,
        "mask": torch.ones(b, n, dtype=torch.bool, device=device),
        "t": torch.rand(b, device=device),
        "x_sc": torch.randn(b, n, 3, device=device) * 0.1,
    }


def _make_nn_config(feature_embedding_mode="concat"):
    """Build minimal nn config for ProteinTransformerAF3."""
    cfg = {
        "token_dim": 512,
        "nlayers": 2,
        "nheads": 8,
        "residual_mha": True,
        "residual_transition": True,
        "parallel_mha_transition": False,
        "use_attn_pair_bias": True,
        "strict_feats": False,
        "feats_init_seq": ["chain_break_per_res", "x_sc"],
        "feats_cond_seq": ["time_emb"],
        "t_emb_dim": 256,
        "dim_cond": 256,
        "idx_emb_dim": 0,
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
    if feature_embedding_mode != "concat":
        cfg["feature_embedding_mode"] = feature_embedding_mode
    return cfg


def _make_model_batch(b=2, n=16, device="cpu"):
    """Minimal batch for model forward."""
    return {
        "x_t": torch.randn(b, n, 3, device=device) * 0.1,
        "mask": torch.ones(b, n, dtype=torch.bool, device=device),
        "t": torch.rand(b, device=device),
        "x_sc": torch.randn(b, n, 3, device=device) * 0.1,
        "cath_code_indices": torch.zeros(b, 3, dtype=torch.long, device=device),
    }


def test_individual_feature_factory_same_output_shape_as_feature_factory_seq():
    """IndividualFeatureFactory produces same output shape as FeatureFactory for seq mode."""
    feats = ["chain_break_per_res", "x_sc"]
    dim_out = 64
    batch = _make_seq_batch()

    factory_concat = FeatureFactory(
        feats=feats,
        dim_feats_out=dim_out,
        use_ln_out=False,
        mode="seq",
        feature_embedding_mode="concat",
    )
    factory_individual = FeatureFactory(
        feats=feats,
        dim_feats_out=dim_out,
        use_ln_out=False,
        mode="seq",
        feature_embedding_mode="individual",
    )

    out_concat = factory_concat(batch)
    out_individual = factory_individual(batch)

    assert out_concat.shape == out_individual.shape == (2, 16, 64)


def test_individual_feature_factory_same_output_shape_as_feature_factory_pair():
    """IndividualFeatureFactory produces same output shape as FeatureFactory for pair mode."""
    feats = ["rel_seq_sep", "xt_pair_dists"]
    dim_out = 128
    batch = _make_pair_batch()

    factory_concat = FeatureFactory(
        feats=feats,
        dim_feats_out=dim_out,
        use_ln_out=True,
        mode="pair",
        feature_embedding_mode="concat",
        xt_pair_dist_dim=64,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3.0,
        seq_sep_dim=127,
    )
    factory_individual = FeatureFactory(
        feats=feats,
        dim_feats_out=dim_out,
        use_ln_out=True,
        mode="pair",
        feature_embedding_mode="individual",
        xt_pair_dist_dim=64,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3.0,
        seq_sep_dim=127,
    )

    out_concat = factory_concat(batch)
    out_individual = factory_individual(batch)

    assert out_concat.shape == out_individual.shape == (2, 16, 16, 128)


def test_model_forward_with_feature_embedding_mode_individual():
    """Model forward runs without error when feature_embedding_mode is individual."""
    cfg = _make_nn_config(feature_embedding_mode="individual")
    model = ProteinTransformerAF3(**cfg)
    batch = _make_model_batch()
    out = model(batch)
    assert out["coords_pred"].shape == (2, 16, 3)


def test_model_forward_concat_vs_individual_same_output_shape():
    """Output shapes match between concat and individual modes for same config."""
    batch = _make_model_batch()

    cfg_concat = _make_nn_config(feature_embedding_mode="concat")
    cfg_individual = _make_nn_config(feature_embedding_mode="individual")

    model_concat = ProteinTransformerAF3(**cfg_concat)
    model_individual = ProteinTransformerAF3(**cfg_individual)

    out_concat = model_concat(batch)
    out_individual = model_individual(batch)

    assert out_concat["coords_pred"].shape == out_individual["coords_pred"].shape == (2, 16, 3)


def test_individual_feature_factory_direct():
    """IndividualFeatureFactory can be used directly and produces correct shape."""
    feats = ["time_emb"]
    dim_out = 64
    batch = _make_seq_batch()

    factory = IndividualFeatureFactory(
        feats=feats,
        dim_feats_out=dim_out,
        use_ln_out=False,
        mode="seq",
        t_emb_dim=256,
    )
    out = factory(batch)
    assert out.shape == (2, 16, 64)
