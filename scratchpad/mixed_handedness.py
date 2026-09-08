"""Is our mirror failure a PURE GLOBAL sign flip, or are chains internally MIXED?

This decides whether the existing detector-plus-reflection route is provably sufficient. A global
mirror is repairable by one reflection; a chain whose helices disagree with each other is NOT
repairable by any global operation, and would force a training-time fix. ProtDiff reported exactly
this mixed-handedness mode, so it is not hypothetical.

⛔ THRESHOLD-FREE BY CONSTRUCTION. Two deliberate choices:
  - "purity" = max(f, 1-f) over a chain's helical dihedral signs. 1.0 = perfectly uniform hand,
    0.5 = maximally mixed. No cutoff is applied anywhere; the DISTRIBUTION is the result.
  - the sliding window is SWEPT over several lengths rather than fixed, because any single window
    length would be an invented parameter and could manufacture or hide mixing on its own.
Natives are included as a control: whatever "pure" looks like must be read off them, not assumed.

Uses the repo's own `_ca_dihedrals`, imported not reimplemented. Convention (measured, see
test_dihedral_convention.py): right-handed helix = NEGATIVE dihedral, so a native chain has
helix_pos_frac near 0 and a mirrored chain near 1.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.utils.c2c_dump import _ca_dihedrals

WINDOWS = [6, 10, 16, 24]


def read_ca(path):
    ca = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(ca, dtype=np.float64)


def helical_signs(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return (sel > 0).astype(float)


def analyse(paths, label):
    fracs, purities, win_dis = [], [], {w: [] for w in WINDOWS}
    skipped = 0
    for p in paths:
        s = helical_signs(read_ca(p))
        if len(s) < 5:
            skipped += 1
            continue
        f = float(s.mean())
        fracs.append(f)
        purities.append(max(f, 1.0 - f))
        for w in WINDOWS:
            if len(s) >= 2 * w:
                # Per-window hand, then the spread ACROSS windows. A pure chain has every window
                # agreeing (spread 0); a mixed chain has windows on both sides.
                wm = np.array([s[i:i + w].mean() for i in range(0, len(s) - w + 1)])
                win_dis[w].append(float(((wm > 0.5).mean()) * (1 - (wm > 0.5).mean()) * 4.0))
    a, pur = np.asarray(fracs), np.asarray(purities)
    print(f"\n=== {label} ===  n={len(a)} (skipped {skipped})")
    if not len(a):
        print("  ⛔ NOTHING SCORED"); return
    print(f"  helix_pos_frac : mean {a.mean():.4f}  median {np.median(a):.4f}")
    print(f"  PURITY max(f,1-f) : mean {pur.mean():.4f}  median {np.median(pur):.4f}  "
          f"min {pur.min():.4f}")
    print("  purity distribution (1.0 = uniformly handed, 0.5 = maximally mixed):")
    hist, edges = np.histogram(pur, bins=10, range=(0.5, 1.0))
    for h, lo in zip(hist, edges[:-1]):
        print(f"    {lo:.2f}-{lo + 0.05:.2f} {h:5d} {'#' * int(50 * h / max(hist.max(), 1))}")
    print("  within-chain window disagreement (0 = all windows agree, 1 = half-and-half):")
    for w in WINDOWS:
        v = np.asarray(win_dis[w])
        if len(v):
            print(f"    window {w:3d}: mean {v.mean():.4f}  median {np.median(v):.4f}  "
                  f"frac>0.1 {float((v > 0.1).mean()):.3f}  (n={len(v)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_254")
    args = ap.parse_args()
    import glob
    gen = sorted(glob.glob(f"{args.gen_dir}/*_gen.pdb"))
    gt = sorted(glob.glob(f"{args.gen_dir}/*_gt.pdb"))
    print(f"generated: {len(gen)}   natives: {len(gt)}")
    analyse(gt, "NATIVES (control -- defines what 'pure' looks like)")
    analyse(gen, "c2c GENERATED")


if __name__ == "__main__":
    main()
