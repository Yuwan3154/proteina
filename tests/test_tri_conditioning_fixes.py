"""Unit tests for the three tri-arm validation-sampling fixes. Stdlib + torch + omegaconf, no pytest.

Run: python tests/test_tri_conditioning_fixes.py   (exits non-zero on any failure)

The bugs these lock down were all invisible to the existing suite because each one only changed
what the SAMPLING path was conditioned on, never whether it ran:

  BUG 1  contact_map_tri_30M.yaml never declared topology_cond, and _build_self_reference_topology
         gates on it, so the tri arm sampled with no topology reference at all.
  BUG 2  with topology=None the trainer fell through to the variable-length branch and overwrote
         every chain's mask with one random L, so generated maps and ground truth described
         different proteins.
  BUG 3  ContactMapTriSiT never masked its output logits, unlike ContactMapHierSiT.

Each test asserts the FIXED behaviour and would fail against the code as it stood.
"""

import os
import sys
import traceback

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase

_FAILURES = []

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRI_NN_CFG = os.path.join(
    REPO, "configs", "experiment_config", "model", "nn", "contact_map_tri_30M.yaml"
)
HIER_NN_CFG = os.path.join(
    REPO, "configs", "experiment_config", "model", "nn", "contact_map_hier_topology_30M.yaml"
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def case(fn):
    try:
        fn()
        print(f"  PASS  {fn.__name__}")
    except Exception as e:
        _FAILURES.append((fn.__name__, traceback.format_exc()))
        print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")


def _tiny_tri():
    """Small enough to forward on CPU; the (L+T)^2 grid is what makes the real model heavy."""
    torch.manual_seed(0)
    m = ContactMapTriSiT(
        pair_dim=16, tri_hidden=16, n_blocks=1, dim_cond=16,
        max_topology_he_len=4, max_rel_pos=8, topology_vocab_size=65,
    )
    # The output head is zero-init, so an untrained model masks trivially and BUG 3 would look
    # fixed even without the fix. Give it a real bias so padded cells carry a nonzero constant.
    torch.nn.init.normal_(m.out.weight, std=0.5)
    torch.nn.init.constant_(m.out.bias, 0.7)
    return m


def _tiny_batch(B=2, L=10, valid=6, T=4, with_topology=True):
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :valid] = True
    batch = {
        "contact_map_t": torch.rand(B, L, L),
        "mask": mask,
        "t": torch.full((B,), 0.4),
        "residue_type": torch.randint(0, 20, (B, L)),
    }
    if with_topology:
        he = torch.full((B, T), -1, dtype=torch.long)
        he[:, :3] = torch.tensor([5, 9, 12])
        batch["topology_he_tokens"] = he
        batch["topology_he_pos_raw"] = torch.tensor([[1.0, 4.0, 8.0, 0.0]]).expand(B, T).clone()
        from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES

        batch["topology_he_feat"] = torch.rand(B, T, T, N_PAIR_FEATURES)
    return batch


# ── BUG 3: tri must mask its output logits, like hier ────────────────────────────────────────
def t_tri_output_is_masked_outside_valid_region():
    m = _tiny_tri()
    b = _tiny_batch(valid=6)
    out = m(b)["contact_map_logits"]
    check(out.shape == (2, 10, 10), f"unexpected logits shape {tuple(out.shape)}")
    pad = out[:, 6:, :]
    check(torch.all(pad == 0.0), f"padded rows must be exactly 0, max |v|={pad.abs().max():.4e}")
    pad_c = out[:, :, 6:]
    check(torch.all(pad_c == 0.0), f"padded cols must be exactly 0, max |v|={pad_c.abs().max():.4e}")


def t_tri_masked_output_gives_half_probability_in_padding():
    """The published figure's padding artifact: hier's padding is sigmoid(0)=0.5, tri's was not."""
    m = _tiny_tri()
    b = _tiny_batch(valid=6)
    p = m(b)["contact_map_pred"]
    check(
        torch.allclose(p[:, 6:, :], torch.full_like(p[:, 6:, :], 0.5)),
        "padded probabilities must be exactly 0.5 so both arms render identically",
    )


def t_tri_valid_region_is_untouched_by_the_mask_fix():
    """Masking must not zero anything inside the valid region."""
    m = _tiny_tri()
    b = _tiny_batch(valid=6)
    out = m(b)["contact_map_logits"]
    check(out[:, :6, :6].abs().max() > 0, "valid region must not be zeroed")


def t_tri_per_sample_masks_are_respected():
    """Two chains of different lengths in one batch must be masked independently."""
    m = _tiny_tri()
    b = _tiny_batch(B=2, L=10, valid=6)
    b["mask"][1, 4:] = False  # sample 1 is shorter
    out = m(b)["contact_map_logits"]
    check(torch.all(out[1, 4:, :] == 0.0), "sample 1 must be masked from 4")
    check(out[0, 4:6, :6].abs().max() > 0, "sample 0 must NOT be masked from 4")


# ── BUG 1: the topology reference must actually reach the network and change the output ──────
def t_tri_nn_config_declares_topology_cond():
    cfg = OmegaConf.load(TRI_NN_CFG)
    check(
        bool(cfg.get("topology_cond", False)) is True,
        "contact_map_tri_30M.yaml must declare topology_cond: True or "
        "_build_self_reference_topology returns None and sampling is UNCONDITIONED",
    )


def t_both_arms_declare_topology_cond():
    """The gate is read off the nn config, so the two arms must agree or the comparison is unfair."""
    tri = OmegaConf.load(TRI_NN_CFG)
    hier = OmegaConf.load(HIER_NN_CFG)
    check(
        bool(tri.get("topology_cond", False)) == bool(hier.get("topology_cond", False)),
        "the two arms must have the same topology_cond setting",
    )


def t_tri_output_changes_when_topology_is_supplied():
    """Runtime proof that the reference is consumed, not merely accepted."""
    m = _tiny_tri()
    torch.manual_seed(1)
    with_topo = _tiny_batch(with_topology=True)
    without = {k: v for k, v in with_topo.items() if not k.startswith("topology_")}
    a = m(with_topo)["contact_map_logits"]
    b = m(without)["contact_map_logits"]
    check(not torch.allclose(a, b), "supplying a topology reference must change the prediction")


def t_tri_output_changes_when_topology_content_changes():
    """A different reference must give a different map, not just a different tensor shape."""
    m = _tiny_tri()
    b1 = _tiny_batch(with_topology=True)
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b1.items()}
    b2["topology_he_tokens"] = b2["topology_he_tokens"].clone()
    b2["topology_he_tokens"][:, :3] = torch.tensor([20, 31, 44])
    a = m(b1)["contact_map_logits"]
    c = m(b2)["contact_map_logits"]
    check(not torch.allclose(a, c), "a different topology reference must change the prediction")


# ── BUG 2: a fixed chain set must never have its length redrawn ──────────────────────────────
def _varlen(cfg_dict, min_l=50, max_l=384):
    return ModelTrainerBase._use_variable_length_validation_sampling(
        None, OmegaConf.create(cfg_dict), min_l, max_l
    )


def t_fixed_chain_list_disables_length_redraw():
    check(
        _varlen({"fixed_chain_list": "/path/to/val_fixed32_max256.txt"}) is False,
        "a fixed chain set must keep each chain's own length",
    )


def t_fixed_chain_list_beats_explicit_variable_length_flag():
    check(
        _varlen({
            "fixed_chain_list": "/path/to/val_fixed32_max256.txt",
            "variable_length_sampling": True,
        }) is False,
        "fixed_chain_list must win over an explicit variable_length_sampling=True",
    )


def t_variable_length_still_works_without_a_fixed_chain_list():
    """The fix must not disable the feature for the length-based generation case it exists for."""
    check(_varlen({}) is True, "bounds available and no fixed set -> redraw stays enabled")
    check(_varlen({"variable_length_sampling": True}) is True, "explicit True must still enable")
    check(_varlen({"variable_length_sampling": False}) is False, "explicit False must still disable")


def t_variable_length_disabled_when_bounds_are_missing():
    check(_varlen({}, min_l=None, max_l=None) is False, "no bounds -> no redraw")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    print(f"running {len(tests)} tri-conditioning-fix tests")
    for fn in tests:
        case(fn)
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILED:")
        for n, tb in _FAILURES:
            print(f"--- {n} ---\n{tb}")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")
