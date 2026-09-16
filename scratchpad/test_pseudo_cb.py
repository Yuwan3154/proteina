"""Unit tests for the RoseTTAFold-style pseudo-CB fill in ContactMapTransform.

Covers the two things that can silently go wrong:
  1. GEOMETRY -- is the virtual CB actually where a real CB would be? Tested against REAL residues
     that HAVE a CB: recompute it from their N/CA/C and compare. A formula that is subtly wrong
     (mixed-up cross product, missing normalisation) still produces plausible-looking numbers, so
     this is checked against ground truth rather than eyeballed.
  2. MISSING BACKBONE -- the construction needs N, CA and C. Every combination of missing atoms is
     exercised, because the failure mode is a NaN or a silently wrong point entering the contact
     map, not an exception.

⛔ The real .pt files are in PROTEINA PDB ordering (N,CA,C,O,CB) and the transform expects OPENFOLD
ordering (N,CA,C,CB,O). pdb_data.py converts before transforms run, so this test must convert too --
otherwise it would validate the formula against the backbone OXYGEN and pass while meaning nothing.
This exact confusion already broke DSSP training once (see DSSPTargetTransform's docstring).

Run: python scratchpad/test_pseudo_cb.py
"""

import os
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.transforms import ContactMapTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

REAL_PT = "/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed_parking/5w3e_E.pt"
N_I, CA_I, C_I, CB_I = 0, 1, 2, 3

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def make_graph(coords, mask):
    g = Data()
    g.coords = coords
    g.coord_mask = mask
    return g


# ═══ A. geometry against REAL residues that have a CB ═════════════════════════════════════════
print("== A. virtual CB vs the real CB, on real residues ==")
if not os.path.exists(REAL_PT):
    print(f"  [SKIP] {REAL_PT} not present")
else:
    g = torch.load(REAL_PT, weights_only=False)
    coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
    cmask = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].float()

    # sanity: confirm the conversion really put CB at 3 and O at 4 before trusting anything below
    d3 = (coords[:, 3] - coords[:, 1]).norm(dim=-1)
    d4 = (coords[:, 4] - coords[:, 1]).norm(dim=-1)
    m3, m4 = cmask[:, 3] > 0.5, cmask[:, 4] > 0.5
    check("post-conversion index 3 is CB (~1.53 A from CA)", abs(float(d3[m3].mean()) - 1.53) < 0.1,
          f"{float(d3[m3].mean()):.3f} A")
    check("post-conversion index 4 is O (~2.4 A from CA)", abs(float(d4[m4].mean()) - 2.40) < 0.15,
          f"{float(d4[m4].mean()):.3f} A")

    have = (cmask[:, N_I] > 0.5) & (cmask[:, CA_I] > 0.5) & (cmask[:, C_I] > 0.5) & (cmask[:, CB_I] > 0.5)
    real_cb = coords[have, CB_I, :]
    # force the fill path on residues whose CB we are deliberately hiding
    filled = ContactMapTransform._fill_missing_cb_pseudo(
        coords[:, CB_I, :].clone(), coords, cmask, have)
    err = (filled[have] - real_cb).norm(dim=-1)
    check("virtual CB reproduces the real CB", float(err.mean()) < 0.25,
          f"mean {float(err.mean()):.3f} A, max {float(err.max()):.3f} A, n={int(have.sum())}")
    check("virtual CB sits 1.522 A from CA",
          abs(float((filled[have] - coords[have, CA_I, :]).norm(dim=-1).mean()) - 1.522) < 1e-3)

    # the thing the switch exists for
    gly = (cmask[:, CB_I] < 0.5) & (cmask[:, N_I] > 0.5) & (cmask[:, CA_I] > 0.5) & (cmask[:, C_I] > 0.5)
    if int(gly.sum()):
        f2 = ContactMapTransform._fill_missing_cb_pseudo(
            coords[:, CB_I, :].clone(), coords, cmask, cmask[:, CB_I] < 0.5)
        off = (f2[gly] - coords[gly, CA_I, :]).norm(dim=-1)
        check("glycine no longer collapses onto CA", float(off.min()) > 1.5,
              f"min offset {float(off.min()):.3f} A over n={int(gly.sum())} GLY")

# ═══ B. missing-backbone edge cases ═══════════════════════════════════════════════════════════
print("\n== B. missing backbone atoms fall back to CA, never NaN ==")
base = torch.tensor([[[0.0, 0, 0], [1.46, 0, 0], [2.0, 1.4, 0], [9, 9, 9], [0, 0, 0]]])  # N,CA,C,CB,O
for label, drop in [("N missing", N_I), ("C missing", C_I), ("N and C missing", None)]:
    coords = base.clone()
    mask = torch.ones(1, 5)
    mask[0, CB_I] = 0.0                       # CB absent -> fill path
    if drop is None:
        mask[0, N_I] = 0.0
        mask[0, C_I] = 0.0
    else:
        mask[0, drop] = 0.0
    out = ContactMapTransform._fill_missing_cb_pseudo(
        coords[:, CB_I, :].clone(), coords, mask, mask[:, CB_I] < 0.5)
    check(f"{label} -> falls back to CA", torch.allclose(out[0], coords[0, CA_I, :]))
    check(f"{label} -> no NaN", bool(torch.isfinite(out).all()))

print("\n== C. degenerate geometry does not produce NaN ==")
coords = torch.zeros(1, 5, 3)               # every backbone atom at the origin
mask = torch.ones(1, 5); mask[0, CB_I] = 0.0
out = ContactMapTransform._fill_missing_cb_pseudo(
    coords[:, CB_I, :].clone(), coords, mask, mask[:, CB_I] < 0.5)
check("coincident N/CA/C -> finite output", bool(torch.isfinite(out).all()), str(out[0].tolist()))

# ═══ D. the switch itself ═════════════════════════════════════════════════════════════════════
print("\n== D. switch semantics ==")
t_default = ContactMapTransform()
check("default is the historical 'ca' fill", t_default.cb_fill == "ca")
try:
    ContactMapTransform(cb_fill="rosetta")
    check("bad cb_fill raises", False)
except ValueError:
    check("bad cb_fill raises", True)

# default path must be byte-identical to the old behaviour
coords = torch.randn(6, 5, 3) * 5
mask = torch.ones(6, 5)
mask[[1, 4], CB_I] = 0.0
g = make_graph(coords, mask)
old_expected = coords[:, CB_I, :].clone()
old_expected[mask[:, CB_I] < 0.5] = coords[mask[:, CB_I] < 0.5, CA_I, :]
cm_ca = ContactMapTransform(cb_fill="ca")._contact_map_from_distance(g)
d = torch.linalg.norm(old_expected[None] - old_expected[:, None], dim=-1)
check("cb_fill='ca' matches the pre-change computation", torch.equal(cm_ca, (d <= 8.0).to(coords.dtype)))

cm_ps = ContactMapTransform(cb_fill="pseudo_cb")._contact_map_from_distance(g)
check("pseudo_cb produces a valid binary map", bool(torch.isfinite(cm_ps).all())
      and set(cm_ps.unique().tolist()) <= {0.0, 1.0})
check("pseudo_cb map is symmetric", torch.equal(cm_ps, cm_ps.T))
check("the two fills differ (the switch does something)", not torch.equal(cm_ca, cm_ps))

print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
