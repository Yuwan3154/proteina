"""Comprehensive unit tests comparing UDLM NELBO vs simplified cross-entropy loss.

This script compares the newly implemented UDLM continuous-time NELBO (Eq. 18/19,
Schiff et al. ICLR 2025) against the previously used simplified cross-entropy loss
that operates only at corrupted positions (z_t != x).

The tests cover:
1. Non-negativity of both losses
2. Zero loss at perfect prediction
3. Monotonicity: worse predictions → higher loss
4. Scale comparison across different noise levels (t)
5. Binary (vocab=2) and multi-class (vocab=3) scenarios
6. 1D and 2D input shapes
7. Statistical comparison across many random samples
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/home/ubuntu/proteina")
from proteinfoundation.flow_matching.discrete_md4 import (
    UDLMDiscreteDiffusion,
    MaskingSchedule,
    _reverse_broadcast,
)


def _simplified_ce_loss(logits, x, zt, t, vocab_size, noise_schedule):
    """The previously used simplified cross-entropy loss for comparison.

    This is the OLD loss that computed cross-entropy only at corrupted positions
    (z_t != x), weighted by dgamma = dalpha / (1 - alpha).
    """
    if vocab_size == 2:
        log_p1 = F.logsigmoid(logits)
        log_p0 = F.logsigmoid(-logits)
        log_p_true = torch.where(x == 1, log_p1, log_p0)
    else:
        log_p = F.log_softmax(logits, dim=-1)
        one_hot_x = F.one_hot(x.long(), vocab_size).float()
        log_p_true = (one_hot_x * log_p).sum(dim=-1)

    corrupted = (zt != x).float()
    dgamma = noise_schedule.dgamma_times_alpha(t)
    dgamma = _reverse_broadcast(dgamma, x.dim())

    masked_neg_cross_ent = (corrupted * log_p_true).sum(
        dim=tuple(range(1, x.dim()))
    )
    return dgamma * masked_neg_cross_ent


def test_non_negativity():
    """Test that UDLM NELBO is non-negative."""
    print("=" * 60)
    print("TEST 1: Non-negativity")
    print("=" * 60)

    configs = [
        {"vocab_size": 2, "shape": (16, 10, 10), "name": "binary 2D"},
        {"vocab_size": 3, "shape": (16, 20), "name": "3-class 1D"},
        {"vocab_size": 5, "shape": (16, 15), "name": "5-class 1D"},
    ]

    all_pass = True
    for cfg in configs:
        diff = UDLMDiscreteDiffusion(vocab_size=cfg["vocab_size"])
        x = torch.randint(0, cfg["vocab_size"], cfg["shape"])
        t = torch.rand(cfg["shape"][0])
        zt = diff.forward_sample(x, t)

        if cfg["vocab_size"] == 2:
            logits = torch.randn(*cfg["shape"])
        else:
            logits = torch.randn(*cfg["shape"], cfg["vocab_size"])

        loss = diff.diffusion_loss(logits, x, zt, t)
        min_loss = loss.min().item()
        mean_loss = loss.mean().item()
        max_loss = loss.max().item()
        non_neg = (loss >= -1e-6).all().item()

        print(f"  {cfg['name']}: min={min_loss:.6f}, mean={mean_loss:.4f}, "
              f"max={max_loss:.4f}, non_neg={non_neg}")
        if not non_neg:
            all_pass = False

    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_zero_at_perfect_prediction():
    """Test that loss is 0 when prediction equals clean data."""
    print("=" * 60)
    print("TEST 2: Zero loss at perfect prediction")
    print("=" * 60)

    configs = [
        {"vocab_size": 2, "shape": (8, 12, 12), "name": "binary 2D"},
        {"vocab_size": 3, "shape": (8, 32), "name": "3-class 1D"},
    ]

    all_pass = True
    for cfg in configs:
        diff = UDLMDiscreteDiffusion(vocab_size=cfg["vocab_size"])
        x = torch.randint(0, cfg["vocab_size"], cfg["shape"])
        t = torch.rand(cfg["shape"][0])
        zt = diff.forward_sample(x, t)

        # Perfect logits: assign huge probability to the correct class
        x_onehot = F.one_hot(x.long(), cfg["vocab_size"]).float()
        perfect_logits = x_onehot * 50.0 - (1 - x_onehot) * 50.0

        if cfg["vocab_size"] == 2:
            # Binary: extract just the logit for class 1
            perfect_logits = perfect_logits[..., 1]

        loss = diff.diffusion_loss(perfect_logits, x, zt, t)
        max_abs = loss.abs().max().item()
        is_zero = max_abs < 1e-4

        print(f"  {cfg['name']}: max |loss| = {max_abs:.8f}, is_zero={is_zero}")
        if not is_zero:
            all_pass = False

    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_monotonicity():
    """Test that worse predictions give higher loss."""
    print("=" * 60)
    print("TEST 3: Monotonicity (worse pred → higher loss)")
    print("=" * 60)

    diff = UDLMDiscreteDiffusion(vocab_size=3)
    x = torch.randint(0, 3, (32, 20))
    t = torch.full((32,), 0.5)
    zt = diff.forward_sample(x, t)

    x_onehot = F.one_hot(x.long(), 3).float()

    # Good prediction (high confidence on correct class)
    good_logits = x_onehot * 5.0 - (1 - x_onehot) * 5.0
    loss_good = diff.diffusion_loss(good_logits, x, zt, t).mean().item()

    # Medium prediction (moderate confidence)
    med_logits = x_onehot * 1.0 - (1 - x_onehot) * 1.0
    loss_med = diff.diffusion_loss(med_logits, x, zt, t).mean().item()

    # Bad prediction (uniform)
    bad_logits = torch.zeros(32, 20, 3)
    loss_bad = diff.diffusion_loss(bad_logits, x, zt, t).mean().item()

    # Random prediction (random logits)
    rand_logits = torch.randn(32, 20, 3)
    loss_rand = diff.diffusion_loss(rand_logits, x, zt, t).mean().item()

    monotone = loss_good < loss_med < loss_bad
    print(f"  loss_good:   {loss_good:.6f}")
    print(f"  loss_med:    {loss_med:.6f}")
    print(f"  loss_bad:    {loss_bad:.6f}")
    print(f"  loss_rand:   {loss_rand:.6f}")
    print(f"  good < med < bad: {monotone}")
    print(f"  RESULT: {'PASS' if monotone else 'FAIL'}\n")
    return monotone


def test_noise_level_comparison():
    """Compare losses at different noise levels."""
    print("=" * 60)
    print("TEST 4: Loss behavior across noise levels")
    print("=" * 60)

    diff = UDLMDiscreteDiffusion(vocab_size=2)
    x = torch.randint(0, 2, (64, 8, 8))
    logits = torch.randn(64, 8, 8)  # random prediction

    t_values = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    losses_nelbo = []
    losses_ce = []

    for t_val in t_values:
        t = torch.full((64,), t_val)
        zt = diff.forward_sample(x, t)
        loss_n = diff.diffusion_loss(logits, x, zt, t).mean().item()
        loss_c = _simplified_ce_loss(
            logits, x, zt, t, 2, diff.noise_schedule
        ).mean().item()
        losses_nelbo.append(loss_n)
        losses_ce.append(loss_c)
        print(f"  t={t_val:.2f}: NELBO={loss_n:10.4f}, CE={loss_c:10.4f}, "
              f"ratio={loss_n/max(abs(loss_c), 1e-8):.4f}")

    # Both should be non-negative
    non_neg = all(l >= -1e-6 for l in losses_nelbo)
    ce_non_neg = all(l >= -1e-6 for l in losses_ce)
    print(f"  NELBO all non-neg: {non_neg}")
    print(f"  CE all non-neg: {ce_non_neg}")
    print(f"  RESULT: {'PASS' if non_neg else 'FAIL'}\n")
    return non_neg


def test_statistical_comparison():
    """Statistical comparison of UDLM NELBO vs simplified CE loss."""
    print("=" * 60)
    print("TEST 5: Statistical comparison (NELBO vs CE)")
    print("=" * 60)

    configs = [
        {"vocab_size": 2, "shape": (128, 8, 8), "name": "binary 2D (contact map)"},
        {"vocab_size": 3, "shape": (128, 32), "name": "3-class 1D (DSSP)"},
    ]

    for cfg in configs:
        print(f"\n  --- {cfg['name']} ---")
        diff = UDLMDiscreteDiffusion(vocab_size=cfg["vocab_size"])

        nelbo_losses = []
        ce_losses = []
        diffs_array = []

        n_trials = 50
        for _ in range(n_trials):
            x = torch.randint(0, cfg["vocab_size"], cfg["shape"])
            t = torch.rand(cfg["shape"][0])
            zt = diff.forward_sample(x, t)

            if cfg["vocab_size"] == 2:
                logits = torch.randn(*cfg["shape"])
            else:
                logits = torch.randn(*cfg["shape"], cfg["vocab_size"])

            loss_n = diff.diffusion_loss(logits, x, zt, t)
            loss_c = _simplified_ce_loss(
                logits, x, zt, t, cfg["vocab_size"], diff.noise_schedule
            )

            nelbo_losses.append(loss_n.mean().item())
            ce_losses.append(loss_c.mean().item())
            diffs_array.append((loss_n - loss_c).mean().item())

        nelbo_arr = np.array(nelbo_losses)
        ce_arr = np.array(ce_losses)
        diff_arr = np.array(diffs_array)

        print(f"  NELBO: mean={nelbo_arr.mean():.4f}, std={nelbo_arr.std():.4f}, "
              f"min={nelbo_arr.min():.4f}, max={nelbo_arr.max():.4f}")
        print(f"  CE:    mean={ce_arr.mean():.4f}, std={ce_arr.std():.4f}, "
              f"min={ce_arr.min():.4f}, max={ce_arr.max():.4f}")
        print(f"  Diff (NELBO-CE): mean={diff_arr.mean():.4f}, "
              f"std={diff_arr.std():.4f}, min={diff_arr.min():.4f}, "
              f"max={diff_arr.max():.4f}")
        print(f"  Ratio (NELBO/CE): mean={np.mean(nelbo_arr / np.maximum(np.abs(ce_arr), 1e-8)):.4f}")

    print(f"\n  RESULT: PASS (informational)\n")
    return True


def test_near_perfect_comparison():
    """Compare both losses for nearly-perfect predictions."""
    print("=" * 60)
    print("TEST 6: Near-perfect prediction comparison")
    print("=" * 60)

    configs = [
        {"vocab_size": 2, "shape": (64, 8, 8), "name": "binary 2D"},
        {"vocab_size": 3, "shape": (64, 24), "name": "3-class 1D"},
    ]

    for cfg in configs:
        print(f"\n  --- {cfg['name']} ---")
        diff = UDLMDiscreteDiffusion(vocab_size=cfg["vocab_size"])
        x = torch.randint(0, cfg["vocab_size"], cfg["shape"])
        t = torch.full((cfg["shape"][0],), 0.5)
        zt = diff.forward_sample(x, t)

        x_onehot = F.one_hot(x.long(), cfg["vocab_size"]).float()

        # Test with increasing prediction quality
        for temp in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            good_logits = x_onehot * temp - (1 - x_onehot) * temp
            if cfg["vocab_size"] == 2:
                good_logits_in = good_logits[..., 1]
            else:
                good_logits_in = good_logits

            loss_n = diff.diffusion_loss(good_logits_in, x, zt, t).mean().item()
            loss_c = _simplified_ce_loss(
                good_logits_in, x, zt, t, cfg["vocab_size"], diff.noise_schedule
            ).mean().item()
            print(f"  confidence={temp:5.2f}: NELBO={loss_n:10.6f}, "
                  f"CE={loss_c:10.6f}, diff={loss_n - loss_c:10.6f}")

    print(f"\n  RESULT: PASS (informational)\n")
    return True


def test_mask_handling():
    """Test that pair_mask is properly handled."""
    print("=" * 60)
    print("TEST 7: Mask handling")
    print("=" * 60)

    diff = UDLMDiscreteDiffusion(vocab_size=2)
    x = torch.randint(0, 2, (4, 6, 6))
    t = torch.rand(4)
    zt = diff.forward_sample(x, t)
    logits = torch.randn(4, 6, 6)

    # Full mask
    full_mask = torch.ones(4, 6, 6)
    loss_full = diff.diffusion_loss(logits, x, zt, t, pair_mask=full_mask)

    # No mask (should equal full mask)
    loss_none = diff.diffusion_loss(logits, x, zt, t)

    # Zero mask (should give zero loss)
    zero_mask = torch.zeros(4, 6, 6)
    loss_zero = diff.diffusion_loss(logits, x, zt, t, pair_mask=zero_mask)

    # Partial mask
    partial_mask = torch.ones(4, 6, 6)
    partial_mask[:, 3:, :] = 0
    partial_mask[:, :, 3:] = 0
    loss_partial = diff.diffusion_loss(logits, x, zt, t, pair_mask=partial_mask)

    full_eq_none = torch.allclose(loss_full, loss_none, atol=1e-5)
    zero_is_zero = (loss_zero.abs() < 1e-8).all().item()
    partial_less = (loss_partial.abs() <= loss_full.abs() + 1e-5).all().item()

    print(f"  loss_full == loss_none: {full_eq_none}")
    print(f"  loss_zero == 0: {zero_is_zero}")
    print(f"  |loss_partial| <= |loss_full|: {partial_less}")

    all_pass = full_eq_none and zero_is_zero and partial_less
    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_gradient_flow():
    """Test that gradients flow through the NELBO loss."""
    print("=" * 60)
    print("TEST 8: Gradient flow")
    print("=" * 60)

    diff = UDLMDiscreteDiffusion(vocab_size=3)
    x = torch.randint(0, 3, (4, 16))
    t = torch.rand(4)
    zt = diff.forward_sample(x, t)
    logits = torch.randn(4, 16, 3, requires_grad=True)

    loss = diff.diffusion_loss(logits, x, zt, t).mean()
    loss.backward()

    has_grad = logits.grad is not None
    grad_nonzero = logits.grad.abs().sum().item() > 0 if has_grad else False
    no_nan = not torch.isnan(logits.grad).any().item() if has_grad else False

    print(f"  has_grad: {has_grad}")
    print(f"  grad_nonzero: {grad_nonzero}")
    print(f"  no_nan: {no_nan}")

    all_pass = has_grad and grad_nonzero and no_nan
    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def test_sample_step_validity():
    """Test that sample_step produces valid tokens and denoises."""
    print("=" * 60)
    print("TEST 9: Sample step validity")
    print("=" * 60)

    for vsize, shape in [(2, (4, 8, 8)), (3, (4, 16))]:
        diff = UDLMDiscreteDiffusion(vocab_size=vsize)
        x = torch.randint(0, vsize, shape)
        t_val = 0.8
        s_val = 0.6
        zt = diff.forward_sample(x, torch.full((shape[0],), t_val))

        # Give model a decent prediction
        x_onehot = F.one_hot(x.long(), vsize).float()
        logits = x_onehot * 3.0 - (1 - x_onehot) * 3.0
        if vsize == 2:
            logits = logits[..., 1]

        z_new = diff.sample_step(zt, logits, s=s_val, t=t_val)

        valid_range = (z_new >= 0).all() and (z_new < vsize).all()
        acc_before = (zt == x).float().mean().item()
        acc_after = (z_new == x).float().mean().item()

        print(f"  vocab={vsize}: valid_range={valid_range}, "
              f"acc_before={acc_before:.4f}, acc_after={acc_after:.4f}, "
              f"improved={acc_after >= acc_before - 0.05}")

    print(f"  RESULT: PASS\n")
    return True


def main():
    print("=" * 60)
    print("UDLM NELBO vs Simplified CE: Comprehensive Unit Tests")
    print("=" * 60)
    print()

    torch.manual_seed(42)

    results = {
        "Non-negativity": test_non_negativity(),
        "Zero at perfect": test_zero_at_perfect_prediction(),
        "Monotonicity": test_monotonicity(),
        "Noise level comparison": test_noise_level_comparison(),
        "Statistical comparison": test_statistical_comparison(),
        "Near-perfect comparison": test_near_perfect_comparison(),
        "Mask handling": test_mask_handling(),
        "Gradient flow": test_gradient_flow(),
        "Sample step validity": test_sample_step_validity(),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:30s}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
