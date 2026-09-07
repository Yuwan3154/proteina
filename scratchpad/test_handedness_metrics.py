"""Gate for the mirror-detecting validation metrics.

⛔⛔ The metric these replace, `chirality_agree`, was logged for weeks as the mirror detector and
CANNOT detect a mirror: it tests per-residue stereocentres, which stay correctly L in a mirrored
generation (0.999 measured over 122 mirrored chains). A monitoring metric that reports "all good"
through a 48% failure is worse than no metric, so this gate pins the new ones against a KNOWN
mirror, a KNOWN correct structure, and a proper rotation.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.utils.c2c_dump import handedness_metrics

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


rng = np.random.default_rng(0)


def helix(n=40, rise=1.5, radius=2.3, turn=100.0, left=False):
    """An ideal alpha-helix CA trace. turn=+100 deg per residue is right-handed."""
    t = np.arange(n) * np.radians(turn) * (-1.0 if left else 1.0)
    return np.stack([radius * np.cos(t), radius * np.sin(t), rise * np.arange(n)], axis=-1)


def rand_rot():
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0            # force a PROPER rotation
    return q


print("\n=== 1. an ideal helix against itself ===")
R = helix()
m = handedness_metrics(R, R)
check("identical structure: rmsd_proper ~ 0", m["rmsd_proper"] < 1e-6, f"{m['rmsd_proper']:.2e}")
check("identical structure: not flagged mirrored", m["is_mirrored"] == 0.0)

print("\n=== 2. a proper rotation must NOT read as a mirror ===")
m = handedness_metrics(R @ rand_rot(), R)
check("rotated: rmsd_proper ~ 0", m["rmsd_proper"] < 1e-6, f"{m['rmsd_proper']:.2e}")
check("rotated: not flagged mirrored", m["is_mirrored"] == 0.0)

print("\n=== 3. a REFLECTION must read as a mirror ===")
Rm = R.copy()
Rm[:, 2] *= -1.0
m = handedness_metrics(Rm, R)
check("reflected: flagged mirrored", m["is_mirrored"] == 1.0)
check("reflected: proper rmsd >> reflected rmsd",
      m["rmsd_proper"] > 2.0 * m["rmsd_reflected"],
      f"proper {m['rmsd_proper']:.2f} vs reflected {m['rmsd_reflected']:.2f}")

print("\n=== 4. helix_pos_frac separates the two hands ===")
mr = handedness_metrics(helix(left=False), R)
ml = handedness_metrics(helix(left=True), R)
check("right-handed and left-handed helices give different helix_pos_frac",
      abs(mr.get("helix_pos_frac", 0.5) - ml.get("helix_pos_frac", 0.5)) > 0.5,
      f"right {mr.get('helix_pos_frac'):.3f} vs left {ml.get('helix_pos_frac'):.3f}")
check("the two hands land on opposite sides of 0.5",
      (mr.get("helix_pos_frac", 0.5) - 0.5) * (ml.get("helix_pos_frac", 0.5) - 0.5) < 0)

print("\n=== 5. degenerate input must not crash ===")
check("too-short input returns an empty dict", handedness_metrics(R[:4], R[:4]) == {})

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
