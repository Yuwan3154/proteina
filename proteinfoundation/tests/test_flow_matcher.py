# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Unit tests for FlowMatcher modalities.
"""

import torch

from proteinfoundation.flow_matching.r3n_fm import FlowMatcher


class TestFlowMatcherCoordinates:
    """Tests for coordinate modality."""

    def test_sample_reference_shape_and_center(self):
        fm = FlowMatcher(modality="coordinates", zero_com=True, scale_ref=1.0)
        n = 32
        nsamples = 3
        mask = torch.ones(nsamples, n, dtype=torch.bool)

        x = fm.sample_reference(n, shape=(nsamples,), mask=mask, modality="coordinates")
        assert x.shape == (nsamples, n, 3)
        com = torch.mean(x, dim=1)
        assert torch.allclose(com, torch.zeros_like(com), atol=1e-5)

    def test_interpolate_limits(self):
        fm = FlowMatcher(modality="coordinates", zero_com=True, scale_ref=1.0)
        n = 16
        nsamples = 2
        mask = torch.ones(nsamples, n, dtype=torch.bool)
        x_0 = fm.sample_reference(n, shape=(nsamples,), mask=mask, modality="coordinates")
        x_1 = torch.randn(nsamples, n, 3)
        x_1 = fm._mask_and_zero_com(x_1, mask)

        x_t0 = fm.interpolate(x_0, x_1, torch.zeros(nsamples), modality="coordinates")
        x_t1 = fm.interpolate(x_0, x_1, torch.ones(nsamples), modality="coordinates")
        assert torch.allclose(x_t0, x_0, atol=1e-5)
        assert torch.allclose(x_t1, x_1, atol=1e-5)


class TestFlowMatcherContactMap:
    """Tests for contact map modality."""

    def test_sample_reference_shape_and_symmetry(self):
        fm = FlowMatcher(modality="contact_map", scale_ref=1.0)
        n = 24
        nsamples = 2
        mask = torch.ones(nsamples, n, dtype=torch.bool)

        c = fm.sample_reference(n, shape=(nsamples,), mask=mask, modality="contact_map")
        assert c.shape == (nsamples, n, n)
        assert torch.allclose(c, c.transpose(-1, -2), atol=1e-5)

    def test_interpolate_limits(self):
        fm = FlowMatcher(modality="contact_map", scale_ref=1.0)
        n = 12
        nsamples = 2
        mask = torch.ones(nsamples, n, dtype=torch.bool)

        c_0 = fm.sample_reference(n, shape=(nsamples,), mask=mask, modality="contact_map")
        c_1 = torch.randint(0, 2, (nsamples, n, n)).float()
        c_1 = (c_1 + c_1.transpose(-1, -2)) / 2.0

        c_t0 = fm.interpolate(c_0, c_1, torch.zeros(nsamples), mask=mask, modality="contact_map")
        c_t1 = fm.interpolate(c_0, c_1, torch.ones(nsamples), mask=mask, modality="contact_map")
        assert torch.allclose(c_t0, c_0, atol=1e-5)
        assert torch.allclose(c_t1, c_1, atol=1e-5)

    def test_xt_dot_shape(self):
        fm = FlowMatcher(modality="contact_map", scale_ref=1.0)
        n = 10
        nsamples = 2
        mask = torch.ones(nsamples, n, dtype=torch.bool)
        c_1 = torch.randn(nsamples, n, n)
        c_1 = (c_1 + c_1.transpose(-1, -2)) / 2.0
        c_t = torch.randn(nsamples, n, n)
        c_t = (c_t + c_t.transpose(-1, -2)) / 2.0
        t = torch.full((nsamples,), 0.5)

        v = fm.xt_dot(c_1, c_t, t, mask=mask, modality="contact_map")
        assert v.shape == (nsamples, n, n)


class TestFlowMatcherModalitySwitch:
    """Ensure modality switching works without re-instantiation."""

    def test_switching_modalities(self):
        fm = FlowMatcher(modality="coordinates", zero_com=True, scale_ref=1.0)
        mask = torch.ones(1, 8, dtype=torch.bool)

        coord_sample = fm.sample_reference(8, shape=(1,), mask=mask)
        assert coord_sample.shape[-1] == 3

        fm.set_modality("contact_map")
        cm_sample = fm.sample_reference(8, shape=(1,), mask=mask)
        assert cm_sample.shape[-1] == 8
        assert torch.allclose(cm_sample, cm_sample.transpose(-1, -2), atol=1e-5)

