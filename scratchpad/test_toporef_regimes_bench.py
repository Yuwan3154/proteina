"""CPU unit tests for scratchpad/toporef_regimes_bench.py."""

import math
import os
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def _install_stubs():
    """Install lightweight CPU stubs for Frame2ConFind and contact_ebm before importing toporef_regimes_bench."""
    f2c_pkg = types.ModuleType("Frame2ConFind")
    f2c_inf = types.ModuleType("Frame2ConFind.inference")
    f2c_api = types.ModuleType("Frame2ConFind.inference.api")

    class _DummyPredictor:
        def __init__(self, checkpoint=None, amp_dtype="fp32", compile_model=False):
            self.device = torch.device("cpu")

        def predict_batch(self, x, mask):
            B, L = mask.shape
            ca = x[:, :, 1, :]
            d = torch.cdist(ca, ca)
            return torch.sigmoid((8.0 - d) * 2.0) * (1.0 - torch.eye(L)[None])

    f2c_api.Frame2ConFindPredictor = _DummyPredictor
    sys.modules["Frame2ConFind"] = f2c_pkg
    sys.modules["Frame2ConFind.inference"] = f2c_inf
    sys.modules["Frame2ConFind.inference.api"] = f2c_api

    cebm_pkg = types.ModuleType("contact_ebm")
    cebm_diff = types.ModuleType("contact_ebm.diffusion")
    cebm_sched = types.ModuleType("contact_ebm.diffusion.schedules")
    cebm_mod = types.ModuleType("contact_ebm.model")
    cebm_cebm = types.ModuleType("contact_ebm.model.contact_ebm")
    cebm_scripts = types.ModuleType("contact_ebm.scripts")
    cebm_train = types.ModuleType("contact_ebm.scripts.train")

    def symmetrize_matrix(x):
        iu = torch.triu(x, diagonal=1)
        return iu + iu.transpose(-1, -2)

    class _DummySchedule:
        def alpha(self, t):
            return 1.0 - t

    class _DummyDiffusion:
        def __init__(self, timesteps=20):
            self.timesteps = timesteps
            self.mask_id = 2
            self.schedule = _DummySchedule()

        def to(self, dev):
            return self

        def get_sampling_grid(self, i, timesteps):
            t = 1.0 - float(i) / float(timesteps)
            s = 1.0 - float(i + 1) / float(timesteps)
            return s, t

        def forward_sample(self, x0, t_tensor, pair_mask):
            keep = (torch.rand_like(x0.float()) > t_tensor.view(-1, 1, 1)).long()
            out = torch.where(keep.bool(), x0, torch.full_like(x0, self.mask_id))
            return symmetrize_matrix(out.float()).long() * pair_mask.long()

        def sample_step(self, zt, logits, s, t, pair_mask=None):
            pred = (torch.sigmoid(logits) >= 0.5).long()
            return symmetrize_matrix(pred.float()).long() * pair_mask.long()

        def decode(self, zt, logits, pair_mask=None):
            pred = (torch.sigmoid(logits) >= 0.5).long()
            return symmetrize_matrix(pred.float()).long() * pair_mask.long()

    class _DummyContactEBM(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def _pair_mask(self, rmask):
            B, L = rmask.shape
            pm = rmask[:, :, None] * rmask[:, None, :]
            eye = torch.eye(L, device=rmask.device)[None]
            return pm * (1.0 - eye)

        def _onehot(self, zt):
            return torch.nn.functional.one_hot(zt.clamp(0, 2), 3).float()

        def compute_outputs(self, x_oh, aatype, rmask, t_tensor, create_graph=False):
            logits = (x_oh[..., 1] - x_oh[..., 0]) * 2.0
            energy = -logits.mean(dim=(-1, -2)) + t_tensor * 0.1
            return logits, energy, None

        def energy(self, x_oh, aatype, rmask, t_tensor):
            _, e, _ = self.compute_outputs(x_oh, aatype, rmask, t_tensor)
            return e

        def score_energy(self, contact_tokens, aatype, rmask, t_eval=0.05):
            t_t = torch.full((contact_tokens.shape[0],), float(t_eval), device=contact_tokens.device)
            return self.energy(self._onehot(contact_tokens), aatype, rmask, t_t)

    cebm_sched.symmetrize_matrix = symmetrize_matrix
    cebm_cebm.ContactEBM = _DummyContactEBM
    cebm_train.build_diffusion = lambda cfg: (_DummyDiffusion(), 3)
    sys.modules["contact_ebm"] = cebm_pkg
    sys.modules["contact_ebm.diffusion"] = cebm_diff
    sys.modules["contact_ebm.diffusion.schedules"] = cebm_sched
    sys.modules["contact_ebm.model"] = cebm_mod
    sys.modules["contact_ebm.model.contact_ebm"] = cebm_cebm
    sys.modules["contact_ebm.scripts"] = cebm_scripts
    sys.modules["contact_ebm.scripts.train"] = cebm_train


_install_stubs()

import toporef_regimes_bench as trb  # noqa: E402
from proteinfoundation.utils.c2c_dump import write_atom14_pdb  # noqa: E402
from synth_tri_bench import structure_atom37  # noqa: E402


def test_pack_unpack_upper_triangle():
    rng = np.random.default_rng(42)
    for L in (17, 32, 65):
        raw = rng.integers(0, 2, size=(5, L, L), dtype=np.uint8)
        for i in range(5):
            raw[i] = np.triu(raw[i], 1) + np.triu(raw[i], 1).T
        packed = trb.pack_upper_triangle(raw, L)
        expected_bytes = math.ceil((L * (L - 1) // 2) / 8)
        assert packed.dtype == np.uint8 and packed.shape == (5, expected_bytes)
        unpacked = trb.unpack_upper_triangle(packed, L)
        assert np.array_equal(unpacked.astype(np.uint8), raw)


def test_coords14_to_atom37_matches_pdb_roundtrip():
    L = 12
    aatype = torch.tensor([0, 7, 1, 2, 3, 4, 7, 5, 14, 8, 9, 10], dtype=torch.long)
    torch.manual_seed(7)
    coords14 = torch.randn(L, 14, 3) * 5.0
    full, amask = trb.coords14_to_atom37(coords14, aatype)
    # Glycine (aatype == 7) at positions 1 and 6 has no CB (slot 3 in atom37)
    assert not bool(amask[1, 3]) and not bool(amask[6, 3])
    assert bool(amask[0, 3]) and bool(amask[2, 3])
    # N, CA, C, O present for all residues
    for slot in (0, 1, 2, 4):
        assert bool(amask[:, slot].all())
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.pdb")
        write_atom14_pdb(p, coords14.numpy(), aatype, torch.ones(L))
        _, full_pdb, amask_pdb = structure_atom37(p, chain="A")
        assert torch.equal(amask, amask_pdb)
        assert float((full - full_pdb).abs().max()) < 2e-3


def test_aa_compatibility_and_cebm_reverse():
    with tempfile.TemporaryDirectory() as td:
        stems = ["ch0", "ch1"]
        L = 24
        for idx, st in enumerate(stems):
            g = types.SimpleNamespace()
            aa = torch.tensor([(i + idx) % 20 for i in range(L)], dtype=torch.long)
            cm = torch.zeros(L, L, dtype=torch.float16)
            for i in range(L - 8):
                cm[i, i + 7] = 0.8
                cm[i + 7, i] = 0.8
            g.residue_type = aa
            g.contact_map_confind = cm
            torch.save(g, os.path.join(td, f"{st}.pt"))
        stats = trb.build_natural_aa_pair_stats(td, stems)
        assert stats["n_chains"] == 2
        assert stats["log_odds"].shape == (20, 20)

        g0 = torch.load(os.path.join(td, "ch0.pt"), weights_only=False)
        sc_nat = trb.score_map_aa_compatibility((g0.contact_map_confind.float() >= 0.01).numpy(), g0.residue_type, stats)
        assert np.isfinite(sc_nat["soft_log_odds"]) and sc_nat["jsd_vs_nat"] >= 0.0

        # Test ContactEBM reverse trajectory with K=0 and K=20
        dev = torch.device("cpu")
        cebm_model = trb.ContactEBM()
        cebm_diff, _ = trb.build_diffusion({})
        init_bmap = (g0.contact_map_confind.float().numpy() >= 0.01).astype(np.uint8)
        for t0 in (0.3, 0.4, 0.5):
            for K in (0, 20):
                out = trb.cebm_reverse_sample_from_t0(
                    cebm_model, cebm_diff, init_bmap, g0.residue_type, L, t0=t0, K_corrector=K, dev=dev, seed=0
                )
                assert out["packed_clean_steps"].dtype == np.uint8
                assert out["packed_clean_steps"].shape[0] == out["n_calls"]
                unp = trb.unpack_upper_triangle(out["packed_clean_steps"], L)
                assert unp.shape == (out["n_calls"], L, L)
                if K == 20:
                    assert out["n_calls"] > 20


if __name__ == "__main__":
    test_pack_unpack_upper_triangle()
    test_coords14_to_atom37_matches_pdb_roundtrip()
    test_aa_compatibility_and_cebm_reverse()
    print("ALL TOPOREF REGIMES BENCH CPU TESTS PASSED")
