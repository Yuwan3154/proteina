"""What sign does _ca_dihedrals give for a RIGHT-handed alpha helix? Settle the convention.

`ca_handedness_filter.py` docstring claims "POSITIVE (~+50 deg) in a right-handed helix". The
measured native population says otherwise: 254 native chains gave median helix_pos_frac = 0.0815
(mean 0.1225, only 2.0% above 0.5). Real proteins ARE right-handed, so if positive meant
right-handed those numbers would sit near 0.9.

⛔ This matters beyond tidiness. helix_pos_frac and the 0.12/0.89 calibration are used to LABEL
structures as mirrored. A future reader trusting the docstring would conclude the detector is
inverted and "fix" a detector that is measuring correctly.

Ideal right-handed alpha helix geometry (textbook, not invented): radius 2.3 A, rise 1.5 A per
residue, +100 deg of turn per residue about +z, advancing along +z. Right-handedness is encoded by
the POSITIVE turn accompanying the POSITIVE rise -- that pairing IS the definition, so mirroring
through z gives the left-handed helix.
"""

import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.utils.c2c_dump import _ca_dihedrals

RADIUS, RISE, TURN_DEG = 2.3, 1.5, 100.0


def ideal_helix(n=12, handed="right"):
    sgn = 1.0 if handed == "right" else -1.0
    t = np.arange(n) * np.deg2rad(TURN_DEG) * sgn
    return np.stack([RADIUS * np.cos(t), RADIUS * np.sin(t), np.arange(n) * RISE], axis=-1)


def main():
    for handed in ("right", "left"):
        ca = ideal_helix(handed=handed)
        d = _ca_dihedrals(ca)
        sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
        frac = float((sel > 0).mean()) if len(sel) else float("nan")
        print(f"{handed:>5}-handed ideal helix: mean dihedral = {d.mean():+8.3f} deg   "
              f"helix_pos_frac = {frac:.3f}   (n_sel={len(sel)})")

    # Cross-check: mirroring through z must flip the sign, since a reflection negates every dihedral.
    ca = ideal_helix(handed="right")
    dm = _ca_dihedrals(ca @ np.diag([1.0, 1.0, -1.0]))
    d = _ca_dihedrals(ca)
    print(f"\nreflection check: right {d.mean():+.3f} -> mirrored {dm.mean():+.3f} "
          f"(sum {d.mean() + dm.mean():+.2e}, should be ~0)")

    verdict = "NEGATIVE" if d.mean() < 0 else "POSITIVE"
    print(f"\n=> _ca_dihedrals gives {verdict} dihedrals for a RIGHT-handed helix.")
    print(f"   docstring in ca_handedness_filter.py claims POSITIVE -> "
          f"{'WRONG' if verdict == 'NEGATIVE' else 'correct'}")
    print(f"   measured natives: median helix_pos_frac 0.0815 -> consistent with {verdict}")


if __name__ == "__main__":
    main()
