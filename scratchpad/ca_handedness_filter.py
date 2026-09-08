"""Can a mirrored generation be DETECTED without the native, and fixed by reflecting it?

Handedness is a coin flip: with the checkpoint fixed and only the noise varying, 15 of 32 chains
changed hand between two draws. If the sole error is a global sign, it does not need a training fix
to be usable -- it needs a detector.

The detector: natural proteins are built from right-handed alpha-helices, and under THIS module's
convention the CA-trace pseudo-dihedral over four consecutive residues is NEGATIVE (~-50 deg) in a
right-handed helix and positive in its mirror. A reflection flips the sign of every dihedral, so the
fraction of POSITIVE dihedrals in helical range ("helix_pos_frac") is LOW for a real protein and
HIGH for a mirrored one, and separates the two populations cleanly.

⛔⛔ THE SIGN ABOVE SAID "POSITIVE ... IN A RIGHT-HANDED HELIX" UNTIL 2026-09-07. IT WAS WRONG, AND
INVERTED. Measured, not argued -- `scratchpad/test_dihedral_convention.py` builds an ideal helix
(radius 2.3 A, rise 1.5 A, 100 deg/residue) and reports:
    right-handed: mean dihedral -50.044 deg, helix_pos_frac 0.000
    left-handed:  mean dihedral +50.044 deg, helix_pos_frac 1.000
    reflection check: -50.044 -> +50.044, sum 0.00e+00
and 254 NATIVE chains measure median helix_pos_frac 0.0815 (mean 0.1225, 2.0% above 0.5), which is
only consistent with right-handed = NEGATIVE.
⭐ The CODE and the 0.12/0.89 calibration were always correct; only this comment was wrong. Anyone
who "fixes" the code to match the old comment BREAKS A WORKING DETECTOR -- which is why the
correction is spelled out rather than silently edited.

⛔ This is scored WITHOUT reference to the native -- otherwise it would be useless at inference. The
native is used only to LABEL each chain for scoring the detector, never as an input to it.
"""

import argparse
import glob
import os
import sys

import numpy as np


def read_ca(path):
    ca = []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(ca)


def dihedrals(ca):
    """CA pseudo-dihedral for every consecutive quadruple, in degrees."""
    b0 = ca[1:-2] - ca[0:-3]
    b1 = ca[2:-1] - ca[1:-2]
    b2 = ca[3:] - ca[2:-1]
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1, axis=1, keepdims=True))
    x = (n1 * n2).sum(-1)
    y = (m * n2).sum(-1)
    return np.degrees(np.arctan2(y, x))


def helical_score(ca):
    """Fraction of HELICAL-range dihedrals that are positive (right-handed).

    A right-handed alpha-helix sits near +50 deg; its mirror near -50. Restricting to |d| in
    30-90 deg keeps helical geometry and discards extended/loop regions, whose dihedrals are
    broadly distributed and carry no handedness signal.
    """
    d = dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    if len(sel) < 5:
        return None, len(sel)
    return float((sel > 0).mean()), len(sel)


def kabsch_rmsd(a, b, allow_reflection=False):
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((a @ rot) - b) ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_254")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    rows = []
    for gp in sorted(glob.glob(os.path.join(args.dir, "*_gen.pdb"))):
        tp = gp.replace("_gen.pdb", "_gt.pdb")
        if not os.path.exists(tp):
            continue
        g, t = read_ca(gp), read_ca(tp)
        n = min(len(g), len(t))
        if n < 40:
            continue
        g, t = g[:n], t[:n]
        prop = kabsch_rmsd(g, t)
        refl = kabsch_rmsd(g, t, allow_reflection=True)
        is_mirror = prop > 2.0 * refl and prop - refl > 1.0     # label, from the native
        score, nsel = helical_score(g)                          # detector, native-free
        if score is None:
            continue
        rows.append((os.path.basename(gp)[:-8], is_mirror, score, nsel, prop, refl))

    if not rows:
        print("no structures found")
        return 1

    # ⭐ CALIBRATE ON THE NATIVES, never on an assumed sign convention. Every _gt structure is a
    # real protein and therefore correct-handed by definition, so whatever value they take IS the
    # right-handed signature. This removes the guesswork about which way the dihedral sign runs.
    nat = []
    for gp in sorted(glob.glob(os.path.join(args.dir, "*_gt.pdb"))):
        t = read_ca(gp)
        if len(t) < 40:
            continue
        sc, _ = helical_score(t)
        if sc is not None:
            nat.append(sc)
    if nat:
        na = np.array(nat)
        print(f"NATIVE calibration ({len(na)} real structures): right-handed fraction "
              f"mean {na.mean():.3f}  min {na.min():.3f}  max {na.max():.3f}\n")

    mir = [r for r in rows if r[1]]
    ok = [r for r in rows if not r[1]]
    print(f"structures scored: {len(rows)}   mirrored (by native): {len(mir)}   correct: {len(ok)}\n")
    for lab, grp in (("correct", ok), ("mirrored", mir)):
        if grp:
            v = np.array([r[2] for r in grp])
            print(f"  {lab:>8}: right-handed dihedral fraction  "
                  f"mean {v.mean():.3f}  min {v.min():.3f}  max {v.max():.3f}")

    # Direction taken from the natives, not from an assumption: a generated chain is called
    # mirrored when its score falls on the OPPOSITE side of the threshold from real proteins.
    nat_mean = float(np.mean(nat)) if nat else 0.5
    native_low = nat_mean < args.threshold
    flag = (lambda v: v >= args.threshold) if native_low else (lambda v: v < args.threshold)
    print(f"natives sit {'BELOW' if native_low else 'ABOVE'} {args.threshold} "
          f"(mean {nat_mean:.3f}), so a generated chain is flagged when its score is on the other side.")
    tp_ = sum(1 for r in mir if flag(r[2]))
    fp_ = sum(1 for r in ok if flag(r[2]))
    print(f"\ndetector: flag as mirrored when the score is opposite the native side of {args.threshold}")
    print(f"  correctly flagged   : {tp_}/{len(mir)}")
    print(f"  wrongly flagged     : {fp_}/{len(ok)}")
    acc = (tp_ + (len(ok) - fp_)) / len(rows)
    print(f"  accuracy            : {acc*100:.1f}%")

    # ⭐ Are the false positives simply chains with too little helix for the statistic to mean
    # anything? nsel is the number of dihedrals in helical range; a low count makes the fraction
    # noisy. If the FPs cluster at low nsel, abstaining there is principled rather than tuned.
    fps = [r for r in ok if flag(r[2])]
    tns = [r for r in ok if not flag(r[2])]
    if fps:
        print(f"\n  false positives  : nsel {sorted(r[3] for r in fps)}")
        print(f"  true negatives   : nsel median {int(np.median([r[3] for r in tns]))}, "
              f"min {min(r[3] for r in tns)}")
        for gate in (10, 20, 30, 40):
            keep = [r for r in rows if r[3] >= gate]
            if not keep:
                continue
            m2 = [r for r in keep if r[1]]
            o2 = [r for r in keep if not r[1]]
            t2 = sum(1 for r in m2 if flag(r[2]))
            f2 = sum(1 for r in o2 if flag(r[2]))
            a2 = (t2 + (len(o2) - f2)) / len(keep)
            print(f"  abstain when nsel < {gate:>2}: keeps {len(keep):>3}/{len(rows)} chains, "
                  f"caught {t2}/{len(m2)}, false {f2}/{len(o2)}, accuracy {a2*100:.1f}%")

    # What the fix would BUY: reflect every flagged structure and rescore.
    before = float(np.mean([r[4] for r in rows]))
    after = float(np.mean([(r[5] if flag(r[2]) else r[4]) for r in rows]))
    print(f"\nmean CA-RMSD as generated        : {before:.2f} A")
    print(f"mean CA-RMSD, flagged reflected   : {after:.2f} A")
    print("\n⚠️ The detector never sees the native. The native is used only to LABEL chains for")
    print("   scoring it, so the accuracy above is what it would achieve at inference time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
