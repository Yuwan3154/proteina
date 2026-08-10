# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Unit tests for partial diffusion (T2): seeding the coordinate simulation from a real
structure noised to t_start, the flow-matching analog of RFdiffusion's diffuser.partial_T.

These tests use an IDENTITY predictor rather than the real network, so they run on CPU in
milliseconds and verify the seeding/scheduling math itself -- which is the part that would
silently produce wrong-but-plausible structures if it were subtly off.
"""

import pytest
import torch

from proteinfoundation.flow_matching.r3n_fm import FlowMatcher


def _fake_predictor(x_1_fixed):
    """Predictor that always 'predicts' a fixed clean structure, with the vector field that
    is exactly consistent with the interpolation path toward it."""

    def predict(nn_in):
        x_t, t = nn_in["x_t"], nn_in["t"]
        t_ = t[..., None, None]
        v = (x_1_fixed - x_t) / (1.0 - t_).clamp(min=1e-6)
        return {"coords": x_1_fixed.clone(), "v": v}

    return predict


def _setup(n=16, nsamples=2, seed=0):
    torch.manual_seed(seed)
    fm = FlowMatcher(modality="coordinates", zero_com=True, scale_ref=1.0)
    mask = torch.ones(nsamples, n, dtype=torch.bool)
    x_1 = torch.randn(nsamples, n, 3)
    x_1 = x_1 - x_1.mean(dim=-2, keepdim=True)  # zero-COM, as the matcher requires
    return fm, mask, x_1, n, nsamples


class TestPartialDiffusionSeeding:
    def test_requires_both_args(self):
        """Passing only one of the pair must fail loudly, not silently no-op."""
        fm, mask, x_1, n, nsamples = _setup()
        for kwargs in ({"x_1_partial": x_1}, {"t_start": 0.5}):
            with pytest.raises(AssertionError):
                fm.full_simulation(
                    _fake_predictor(x_1), dt=0.1, nsamples=nsamples, n=n,
                    self_cond=False, mask=mask, modality="coordinates", **kwargs,
                )

    @pytest.mark.parametrize("t_start", [-0.1, 0.0, 1.0, 1.5])
    def test_rejects_out_of_range_t_start(self, t_start):
        fm, mask, x_1, n, nsamples = _setup()
        with pytest.raises(AssertionError):
            fm.full_simulation(
                _fake_predictor(x_1), dt=0.1, nsamples=nsamples, n=n, self_cond=False,
                mask=mask, modality="coordinates", x_1_partial=x_1, t_start=t_start,
            )

    def test_higher_t_start_retains_more_of_the_input(self):
        """The defining property: larger t_start = less noise = state retains MORE of the
        input structure. This is the INVERTED convention vs RFdiffusion's partial_T and is
        the single easiest thing to get backwards, so it is asserted directly.

        Uses a stationary predictor (predicts x_t is already clean => v=0), so the output is
        exactly the seeded state and the measurement isolates seeding alone. A predictor with
        a nonzero vector field would confound this: with the exact field toward a fixed
        target, dx/dt = (x_tgt - x)/(1-t) converges to x_tgt from ANY start, analytically
        erasing the initial condition -- so such a test would show no t_start dependence even
        though the seeding is correct. This tests the seeding, NOT the denoising trajectory
        (the latter needs the real network + weights, so it is out of scope for a unit test).
        """
        fm, mask, x_1, n, nsamples = _setup()

        def stationary_predict(nn_in):
            return {
                "coords": nn_in["x_t"].clone(),
                "v": torch.zeros_like(nn_in["x_t"]),
            }

        dists = {}
        for t_start in (0.1, 0.5, 0.9):
            torch.manual_seed(7)  # identical reference-noise draw across settings
            out = fm.full_simulation(
                stationary_predict, dt=0.02, nsamples=nsamples, n=n,
                self_cond=False, mask=mask, modality="coordinates",
                x_1_partial=x_1, t_start=t_start,
            )["coords"]
            dists[t_start] = (out - x_1).pow(2).sum(-1).sqrt().mean().item()

        assert dists[0.9] < dists[0.5] < dists[0.1], (
            f"expected state to retain more of the input as t_start rises, got {dists}"
        )

    def test_seeded_state_is_on_the_training_interpolation_path(self):
        """The seeded x_t must equal interpolate(x_0, x_1, t_snap) exactly -- if it drifts off
        the path the model was trained on, the model is being asked to denoise a state it has
        never seen. Verified by running zero integration steps (t_start on the last grid point)
        and comparing against interpolate() computed independently."""
        fm, mask, x_1, n, nsamples = _setup()
        dt = 0.1
        nsteps = 10
        ts = fm.get_schedule(mode="uniform", nsteps=nsteps, modality="coordinates")

        captured = {}

        def capture_predict(nn_in):
            captured.setdefault("x_t", nn_in["x_t"].clone())
            captured.setdefault("t", nn_in["t"].clone())
            return {"coords": nn_in["x_t"].clone(), "v": torch.zeros_like(nn_in["x_t"])}

        t_start = float(ts[7])
        torch.manual_seed(11)
        fm.full_simulation(
            capture_predict, dt=dt, nsamples=nsamples, n=n, self_cond=False,
            mask=mask, modality="coordinates", x_1_partial=x_1, t_start=t_start,
        )

        # Re-derive the expected seed with the same reference draw.
        torch.manual_seed(11)
        x_0 = fm.sample_reference(n, shape=(nsamples,), mask=mask, modality="coordinates")
        expected = fm.interpolate(
            x_0=x_0, x_1=x_1, t=t_start * torch.ones(nsamples), mask=mask,
            modality="coordinates",
        )

        assert torch.allclose(captured["t"], torch.full((nsamples,), t_start), atol=1e-6), (
            "loop must first evaluate at exactly the snapped t used to build the state"
        )
        assert torch.allclose(captured["x_t"], expected, atol=1e-5), (
            "seeded state does not lie on the training interpolation path"
        )

    def test_t_start_snaps_onto_schedule_grid(self):
        """An off-grid t_start must snap to a grid point, so the first step's dt stays
        consistent with the state it is applied to."""
        fm, mask, x_1, n, nsamples = _setup()
        captured = {}

        def capture_predict(nn_in):
            captured.setdefault("t", float(nn_in["t"][0]))
            return {"coords": nn_in["x_t"].clone(), "v": torch.zeros_like(nn_in["x_t"])}

        ts = fm.get_schedule(mode="uniform", nsteps=10, modality="coordinates")
        fm.full_simulation(
            capture_predict, dt=0.1, nsamples=nsamples, n=n, self_cond=False,
            mask=mask, modality="coordinates", x_1_partial=x_1, t_start=0.4321,
        )
        assert captured["t"] in [pytest.approx(float(v)) for v in ts], (
            f"first evaluated t={captured['t']} is not a schedule grid point"
        )

    def test_default_path_unchanged_when_unused(self):
        """Regression guard: with the new args absent, behavior must be bit-identical to
        before (the feature is purely additive)."""
        fm, mask, x_1, n, nsamples = _setup()
        pred = _fake_predictor(x_1)

        torch.manual_seed(5)
        a = fm.full_simulation(
            pred, dt=0.05, nsamples=nsamples, n=n, self_cond=False, mask=mask,
            modality="coordinates",
        )["coords"]
        torch.manual_seed(5)
        b = fm.full_simulation(
            pred, dt=0.05, nsamples=nsamples, n=n, self_cond=False, mask=mask,
            modality="coordinates", x_1_partial=None, t_start=None,
        )["coords"]
        assert torch.equal(a, b)


class TestPartialDiffusionModalityGuard:
    def test_contact_map_modality_rejects_partial_args(self):
        """Contact-map mode cannot honor these args; it must raise rather than ignore them."""
        fm = FlowMatcher(modality="contact_map", scale_ref=1.0)
        n, nsamples = 8, 2
        mask = torch.ones(nsamples, n, dtype=torch.bool)
        x_1 = torch.randn(nsamples, n, 3)
        with pytest.raises(AssertionError, match="only supported in 'coordinates'"):
            fm.full_simulation(
                lambda nn_in: {"coords": None, "v": None},
                dt=0.1, nsamples=nsamples, n=n, self_cond=False, mask=mask,
                modality="contact_map", x_1_partial=x_1, t_start=0.5,
            )
