"""
Unit tests for dual-compile logic in ProteinTransformerAF3.

Verifies that separate compiled artifacts are created for grad-enabled (training)
and no-grad (validation) contexts, and that outputs are consistent with eager.
"""
import torch
import pytest

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3


# Minimal model config for fast tests
MINIMAL_CONFIG = {
    "token_dim": 64,
    "nlayers": 2,
    "nheads": 2,
    "residual_mha": True,
    "residual_transition": True,
    "parallel_mha_transition": False,
    "use_attn_pair_bias": True,
    "strict_feats": False,
    "feats_init_seq": ["res_seq_pdb_idx"],
    "feats_cond_seq": ["time_emb"],
    "feats_pair_repr": ["rel_seq_sep"],
    "feats_pair_cond": [],
    "t_emb_dim": 32,
    "dim_cond": 64,
    "idx_emb_dim": 16,
    "seq_sep_dim": 15,
    "pair_repr_dim": 32,
    "update_pair_repr": False,
    "use_tri_mult": False,
    "num_registers": 0,
    "use_qkln": False,
    "predict_coords": True,
    "contact_map_mode": False,
    "use_torch_compile": False,  # overridden per test
}


def _make_batch(batch_size=2, seq_len=16, device="cpu"):
    """Create a minimal synthetic batch for ProteinTransformerAF3."""
    return {
        "x_t": torch.randn(batch_size, seq_len, 3, device=device),
        "t": torch.rand(batch_size, device=device),
        "mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
    }


class TestDualCompileArtifacts:
    """Tests for the dual train/eval compiled artifact logic."""

    def test_eager_when_compile_disabled(self):
        """When use_torch_compile=False, no compiled artifacts are created."""
        cfg = {**MINIMAL_CONFIG, "use_torch_compile": False}
        model = ProteinTransformerAF3(**cfg)
        batch = _make_batch()

        # Train mode (grad enabled)
        out = model(batch)
        assert "coords_pred" in out
        assert getattr(model, "_forward_compiled_train", None) is None
        assert getattr(model, "_forward_compiled_eval", None) is None

        # Eval mode (no grad)
        with torch.no_grad():
            out = model(batch)
        assert "coords_pred" in out
        assert getattr(model, "_forward_compiled_train", None) is None
        assert getattr(model, "_forward_compiled_eval", None) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
    def test_train_artifact_created_under_grad(self):
        """Compiled train artifact is created when grad is enabled."""
        cfg = {**MINIMAL_CONFIG, "use_torch_compile": True}
        model = ProteinTransformerAF3(**cfg).cuda()
        batch = _make_batch(device="cuda")

        # Before any forward
        assert getattr(model, "_forward_compiled_train", None) is None

        # Forward with grad enabled
        out = model(batch)
        assert "coords_pred" in out
        assert getattr(model, "_forward_compiled_train", None) is not None
        # Eval artifact should NOT be created yet
        assert getattr(model, "_forward_compiled_eval", None) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
    def test_eval_artifact_created_under_no_grad(self):
        """Compiled eval artifact is created under torch.no_grad()."""
        cfg = {**MINIMAL_CONFIG, "use_torch_compile": True}
        model = ProteinTransformerAF3(**cfg).cuda()
        batch = _make_batch(device="cuda")

        with torch.no_grad():
            out = model(batch)
        assert "coords_pred" in out
        assert getattr(model, "_forward_compiled_eval", None) is not None
        # Train artifact should NOT be created yet
        assert getattr(model, "_forward_compiled_train", None) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
    def test_both_artifacts_independent(self):
        """Both artifacts are created independently and are different objects."""
        cfg = {**MINIMAL_CONFIG, "use_torch_compile": True}
        model = ProteinTransformerAF3(**cfg).cuda()
        batch = _make_batch(device="cuda")

        # Create train artifact
        model(batch)
        train_artifact = model._forward_compiled_train

        # Create eval artifact
        with torch.no_grad():
            model(batch)
        eval_artifact = model._forward_compiled_eval

        assert train_artifact is not None
        assert eval_artifact is not None
        assert train_artifact is not eval_artifact

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
    def test_output_consistency_across_modes(self):
        """Compiled train and eval outputs are consistent with eager."""
        cfg_eager = {**MINIMAL_CONFIG, "use_torch_compile": False}
        cfg_compiled = {**MINIMAL_CONFIG, "use_torch_compile": True}

        model_eager = ProteinTransformerAF3(**cfg_eager).cuda().eval()
        model_compiled = ProteinTransformerAF3(**cfg_compiled).cuda().eval()

        # Copy weights from eager to compiled
        model_compiled.load_state_dict(model_eager.state_dict())

        batch = _make_batch(device="cuda")

        # Eager baseline
        with torch.no_grad():
            out_eager = model_eager(batch)

        # Compiled eval
        with torch.no_grad():
            out_compiled = model_compiled(batch)

        # Outputs should be very close (allow small float differences from compilation)
        diff = (out_eager["coords_pred"] - out_compiled["coords_pred"]).abs().max().item()
        assert diff < 1e-3, f"Output divergence too large: {diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
    def test_no_recompilation_on_mode_toggle(self):
        """Toggling between train/eval should NOT trigger recompilation."""
        torch._dynamo.reset()

        cfg = {**MINIMAL_CONFIG, "use_torch_compile": True}
        model = ProteinTransformerAF3(**cfg).cuda()
        batch = _make_batch(device="cuda")

        # Warm up both artifacts
        model(batch)
        with torch.no_grad():
            model(batch)

        # Record recompile counter
        counters_before = dict(torch._dynamo.utils.counters.get("frames", {}))

        # Toggle several times
        for _ in range(4):
            model(batch)
            with torch.no_grad():
                model(batch)

        counters_after = dict(torch._dynamo.utils.counters.get("frames", {}))

        # Frame counts should not increase (no new compilations after warmup)
        for key in counters_after:
            before = counters_before.get(key, 0)
            after = counters_after[key]
            # Allow at most 0 new frames (the warmup frames are already counted)
            assert after <= before, (
                f"Recompilation detected on mode toggle: {key}: {before} -> {after}"
            )
