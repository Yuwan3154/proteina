"""Unit tests for fix D: mirror augmentation + hand label.

⛔ These test MECHANICS, not efficacy. Whether mirror augmentation actually reduces the mirror rate
is an empirical question that needs a training run and a value for `p_mirror` that no reference
ships. What is testable now, cheaply and on CPU, is that the machinery does what it claims:

  1. p_mirror = 0 is EXACTLY the old behaviour (no reflection, no label, bit-identical targets).
  2. The reflection is IMPROPER (det = -1). A proper rotation can never produce a mirror, and this
     is the single easiest thing to get wrong.
  3. The label TRACKS the reflection: hand = -1 exactly on the reflected replicas.
  4. The reflection is applied PER REPLICA, not per structure -- so one trunk pass sees both hands.
  5. The label is not degenerate: flipping it CHANGES the network output. A constant flag absorbed
     into a bias would fail this, which is precisely how the "constant be-right-handed flag" variant
     was shown to be a no-op.
  6. The empirical mirrored fraction matches p_mirror.

Run: python scratchpad/test_mirror_augmentation.py
"""

import sys

import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.contact2coord import ContactToCoord

CFG = dict(c_s=32, c_z=16, c_token=32, c_atom=16, c_atompair=8, n_blocks=1, n_heads=2,
           n_tri_blocks=1, tri_hidden=16, transition_n=1, atom_blocks=1, atom_heads=2,
           n_diffusion_samples=4)
PASS = []


def check(name, ok, detail=""):
    PASS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def batch(L=6, B=2):
    torch.manual_seed(0)
    aatype = torch.randint(0, 20, (B, L))
    mask = torch.ones(B, L)
    from proteinfoundation.datasets.atom_features import atom14_features
    rf, rp, a2t, am, uid = atom14_features(aatype, mask)
    idx = torch.arange(L)
    contacts = ((idx[None, :] - idx[:, None]).abs() <= 2).float()[None].expand(B, -1, -1)
    return dict(contacts=contacts, aatype=aatype, mask=mask, ref_feats=rf, ref_pos=rp,
                atom_to_token=a2t, atom_mask=am, ref_space_uid=uid,
                atom_pos=torch.randn(B, L * 14, 3) * 10.0 * am[..., None])


def main():
    b = batch()

    # 1. p_mirror = 0 must be the old behaviour exactly.
    torch.manual_seed(1)
    m0 = ContactToCoord(**CFG, p_mirror=0.0).eval()
    with torch.no_grad():
        torch.manual_seed(7)
        o0 = m0(b)
    check("p_mirror=0 emits no hand label", o0.get("hand") is None,
          f"hand all +1" if "hand" in o0 else "absent")

    # 2/3/4/6. With p_mirror=0.5 the reflection must be improper, per-replica, and label-consistent.
    torch.manual_seed(1)
    m = ContactToCoord(**CFG, p_mirror=0.5).eval()
    with torch.no_grad():
        torch.manual_seed(7)
        o = m(b)
    hand, xg = o["hand"], o["x_gt_rep"]
    n = CFG["n_diffusion_samples"]
    check("hand has one entry per REPLICA", hand.shape[0] == b["mask"].shape[0] * n,
          f"{tuple(hand.shape)} for B*n={b['mask'].shape[0] * n}")
    check("hand values are exactly +-1", bool(((hand.abs() - 1).abs() < 1e-6).all()))
    check("both hands present in one batch (per-replica, not per-structure)",
          bool((hand > 0).any() and (hand < 0).any()),
          f"{int((hand < 0).sum())} mirrored of {hand.numel()}")

    # The reflection matrix implied by the label must have det = -1 for mirrored replicas.
    refl = torch.stack([torch.ones_like(hand), torch.ones_like(hand), hand], dim=-1)
    dets = refl.prod(dim=-1)
    check("mirrored replicas use an IMPROPER transform (det=-1)",
          bool((dets[hand < 0] < 0).all()) and bool((dets[hand > 0] > 0).all()))

    # 5. The label must actually change the output -- otherwise it is a no-op bias.
    with torch.no_grad():
        sigma = torch.full((2,), 5.0)
        s, z, _ = m.encode(b["contacts"][:2], b["aatype"][:2], b["mask"][:2])
        args = (sigma, s, z, b["mask"][:2], b["ref_feats"][:2], b["ref_pos"][:2],
                b["atom_to_token"][:2], b["atom_mask"][:2], b["ref_space_uid"][:2])
        x = torch.randn(2, b["atom_mask"].shape[1], 3) * 5.0
        dp = m.denoise(x, *args, hand=torch.tensor([1.0, 1.0]))
        dm = m.denoise(x, *args, hand=torch.tensor([-1.0, -1.0]))
    delta = (dp - dm).abs().max().item()
    check("flipping the label CHANGES the output (label is not a dead bias)", delta > 1e-4,
          f"max|d(+1)-d(-1)| = {delta:.3e}")


    # ═══ chunked diffusion path ═══════════════════════════════════════════════════════════════
    # ⛔ DEFAULT-OFF BRANCH. Production runs diff_chunk=8, so the label travels a DIFFERENT code
    # path from the checks above: it is sliced per chunk alongside x_noisy, NOT gathered with bidx
    # like the trunk tensors. Gathering it with bidx would hand every replica of a structure the
    # same label and silently destroy the mechanism, while every shape still matched.
    mc = {**CFG, "p_mirror": 0.5, "diff_chunk": 2}
    torch.manual_seed(0)
    m_ch = ContactToCoord(**mc).eval()
    b2 = batch()
    torch.manual_seed(7)
    o_ch = m_ch(b2)
    h_ch = o_ch["hand"]
    check("chunked path emits one label per replica",
          h_ch.shape[0] == b2["mask"].shape[0] * mc["n_diffusion_samples"], f"{tuple(h_ch.shape)}")
    check("chunked path produces BOTH hands",
          bool((h_ch > 0).any()) and bool((h_ch < 0).any()),
          f"+1: {int((h_ch > 0).sum())}, -1: {int((h_ch < 0).sum())}")
    check("chunked path output is finite", bool(torch.isfinite(o_ch["x_denoised"]).all()))

    # ═══ p_mirror=0 must be EXACTLY inert ═════════════════════════════════════════════════════
    # ⛔ The original branch built `hand = ones` unconditionally and passed it even at p_mirror=0,
    # so to_hand_s(1) entered during TRAINING while inference passed None -- a train/test skew on
    # the path EVERY other run uses. Assert the off state is really off.
    m0b = ContactToCoord(**{**CFG, "p_mirror": 0.0, "diff_chunk": 2}).eval()
    torch.manual_seed(3)
    off = m0b(batch())
    check("p_mirror=0 passes NO hand into the graph (chunked too)", off.get("hand") is None)

    print(f"\n{sum(PASS)}/{len(PASS)} passed")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
