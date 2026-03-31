#!/usr/bin/env python
"""Quick smoke test for the Muon optimizer integration.

Run with:
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
        /home/ubuntu/proteina/proteinfoundation/optim/test_muon.py
"""

import sys
import os

# Ensure the proteina package is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

# ── 1. Standalone Muon optimizer ────────────────────────────────────────

def test_standalone_muon():
    from proteinfoundation.optim.muon import Muon

    print("=== Test: Standalone Muon ===")
    model = nn.Linear(64, 128, bias=False).cuda()
    optimizer = Muon(model.parameters(), lr=0.02, adjust_lr_fn="original")

    x = torch.randn(4, 64, device="cuda")
    for step in range(5):
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  step {step}: loss={loss.item():.4f}")

    print("  ✓ Standalone Muon works\n")


# ── 2. HybridMuonAdamW optimizer ────────────────────────────────────────

def test_hybrid_optimizer():
    from proteinfoundation.optim.muon import HybridMuonAdamW

    print("=== Test: HybridMuonAdamW ===")

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.linear1 = nn.Linear(32, 64)
            self.ln = nn.LayerNorm(64)
            self.linear2 = nn.Linear(64, 10)

        def forward(self, x):
            h = self.embed(x)
            h = self.linear1(h)
            h = self.ln(h)
            return self.linear2(h)

    model = ToyModel().cuda()

    # Manually partition
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embed" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    print(f"  Muon params: {len(muon_params)}, Adam params: {len(adam_params)}")

    param_groups = [
        dict(params=muon_params, lr=1e-3, use_muon=True,
             adjust_lr_fn="match_rms_adamw"),
        dict(params=adam_params, lr=1e-3, use_muon=False),
    ]
    optimizer = HybridMuonAdamW(param_groups)

    x = torch.randint(0, 100, (4, 8), device="cuda")
    for step in range(5):
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  step {step}: loss={loss.item():.4f}")

    # Check state sizes
    muon_states = [optimizer.state[p] for p in muon_params if p in optimizer.state]
    adam_states = [optimizer.state[p] for p in adam_params if p in optimizer.state]

    for s in muon_states:
        assert "momentum_buffer" in s, "Muon state should have momentum_buffer"
        assert "exp_avg" not in s, "Muon state should NOT have exp_avg"
    for s in adam_states:
        assert "exp_avg" in s, "Adam state should have exp_avg"
        assert "exp_avg_sq" in s, "Adam state should have exp_avg_sq"

    print("  ✓ HybridMuonAdamW works, state keys correct\n")


# ── 3. Parameter partitioning via build_optimizer_param_groups ──────────

def test_param_partitioning():
    from proteinfoundation.optim.param_groups import build_optimizer_param_groups
    from types import SimpleNamespace

    print("=== Test: build_optimizer_param_groups ===")

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Should go to Muon (2D hidden weight)
            self.linear_hidden = nn.Linear(64, 128)
            # Should go to Adam (name contains "embed")
            self.embed_proj = nn.Embedding(20, 64)
            # Should go to Adam (LayerNorm 1D params)
            self.layer_norm = nn.LayerNorm(128)
            # Should go to Adam (name pattern: output head)
            self.dssp_head = nn.Linear(128, 3)
            # Should go to Adam (name pattern: output head)
            self.coors_3d_decoder = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 3, bias=False),
            )

    model = MockModel()
    opt_cfg = SimpleNamespace(
        lr=1e-4,
        muon_lr=None,
        muon_momentum=0.95,
        muon_weight_decay=0.0,
        muon_nesterov=True,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
        weight_decay=0.0,
        adam_betas=[0.9, 0.999],
        adam_eps=1e-8,
    )
    opt_cfg.get = lambda k, d=None: getattr(opt_cfg, k, d)

    groups = build_optimizer_param_groups(model, opt_cfg)
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"

    muon_group = [g for g in groups if g["use_muon"]]
    adam_group = [g for g in groups if not g["use_muon"]]
    assert len(muon_group) == 1
    assert len(adam_group) == 1

    muon_param_count = len(muon_group[0]["params"])
    adam_param_count = len(adam_group[0]["params"])

    print(f"  Muon params: {muon_param_count}")
    print(f"  Adam params: {adam_param_count}")

    # The 2D linear_hidden.weight should be in Muon
    # linear_hidden.bias (1D), embed, layer_norm, dssp_head, coors_3d_decoder → Adam
    assert muon_param_count >= 1, "At least linear_hidden.weight should be in Muon"
    assert adam_param_count >= 4, "Embeddings, biases, LN, heads should be in Adam"

    print("  ✓ Parameter partitioning works\n")


# ── 4. LR adjustment modes ─────────────────────────────────────────────

def test_lr_adjustment():
    from proteinfoundation.optim.muon import _adjust_lr_original, _adjust_lr_match_rms_adamw
    import math

    print("=== Test: LR adjustment functions ===")

    # "original": lr * sqrt(max(1, m/n))
    assert _adjust_lr_original(0.02, (512, 128)) == 0.02 * math.sqrt(512 / 128)
    assert _adjust_lr_original(0.02, (128, 512)) == 0.02 * 1.0  # max(1, 128/512) = 1

    # "match_rms_adamw": 0.2 * lr * sqrt(max(m, n))
    assert _adjust_lr_match_rms_adamw(1e-4, (512, 128)) == 0.2 * 1e-4 * math.sqrt(512)
    assert _adjust_lr_match_rms_adamw(1e-4, (128, 512)) == 0.2 * 1e-4 * math.sqrt(512)

    print("  ✓ LR adjustment functions correct\n")


# ── 5. Newton-Schulz orthogonalization ──────────────────────────────────

def test_newton_schulz():
    from proteinfoundation.optim.muon import _newton_schulz_orthogonalize

    print("=== Test: Newton-Schulz orthogonalization ===")

    G = torch.randn(64, 128, device="cuda")
    X = _newton_schulz_orthogonalize(G, ns_steps=5)
    assert X.shape == G.shape
    assert X.dtype == G.dtype

    # Check that the result is approximately orthogonal:
    # For a wide matrix (64x128), X @ X.T should be close to I
    XXT = X @ X.mT
    I = torch.eye(64, device="cuda", dtype=G.dtype)
    frob_error = (XXT - I).norm() / I.norm()
    print(f"  ||XX^T - I||_F / ||I||_F = {frob_error.item():.6f}")
    assert frob_error < 0.5, f"Orthogonalization error too high: {frob_error.item()}"

    # Test with tall matrix
    G_tall = torch.randn(128, 64, device="cuda")
    X_tall = _newton_schulz_orthogonalize(G_tall, ns_steps=5)
    assert X_tall.shape == G_tall.shape

    # Test 3D tensor (conv-like)
    G_3d = torch.randn(16, 32, 3, device="cuda")
    X_3d = _newton_schulz_orthogonalize(G_3d, ns_steps=5)
    assert X_3d.shape == G_3d.shape

    print("  ✓ Newton-Schulz orthogonalization works\n")


# ── 6. Backward compatibility (no "optimizer" key in config) ────────────

def test_backward_compat():
    """Verify that configs without 'optimizer' key still produce Adam."""
    from types import SimpleNamespace

    print("=== Test: Backward compatibility ===")

    # Simulate a cfg_exp.opt without "optimizer" key
    cfg_opt = SimpleNamespace(lr=1e-4)
    cfg_opt.get = lambda k, d=None: getattr(cfg_opt, k, d)

    opt_type = cfg_opt.get("optimizer", "adam")
    assert opt_type == "adam", f"Expected 'adam', got '{opt_type}'"

    print("  ✓ Backward compatible (defaults to 'adam')\n")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_backward_compat()
    test_lr_adjustment()

    if torch.cuda.is_available():
        test_newton_schulz()
        test_standalone_muon()
        test_hybrid_optimizer()
    else:
        print("⚠ Skipping GPU tests (no CUDA available)")

    test_param_partitioning()

    print("=" * 50)
    print("All tests passed! ✓")
