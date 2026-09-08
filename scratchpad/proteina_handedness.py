"""Does PROTEINA mirror? Run OUR detector on Proteina's own generations.

⛔ This tests a premise my whole explanation rests on and that I had NOT measured -- I took
"Proteina gets around this fine" from the user's framing. If Proteina also mirrors, the
contact-map-conditioning story is wrong and has to be thrown out.

Uses `helix_pos_frac`, the native-FREE half of handedness_metrics: the fraction of helical-range
CA pseudo-dihedrals that are positive. Calibration already established on our data:
    real proteins ~= 0.12      mirrored ~= 0.89
⭐ `_ca_dihedrals` is IMPORTED, not reimplemented, so this is byte-for-byte the same detector that
produced the 48% figure for c2c. A reimplementation could differ in sign convention and would make
the comparison meaningless.

Proteina samples are CA-only backbones, which is exactly what this consumes.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.utils.c2c_dump import _ca_dihedrals


def read_ca(path):
    ca = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(ca, dtype=np.float64)


def helix_pos_frac(ca):
    if len(ca) < 8:
        return None
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    if len(sel) < 5:
        return None
    return float((sel > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True, help="file with one PDB path per line")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    with open(args.filelist) as fh:
        paths = [l.strip() for l in fh if l.strip()]

    fracs, skipped = [], 0
    for p in paths:
        ca = read_ca(p)
        f = helix_pos_frac(ca)
        if f is None:
            skipped += 1
            continue
        fracs.append(f)

    a = np.asarray(fracs)
    n = len(a)
    print(f"=== {args.label} ===")
    print(f"  structures scored : {n}   (skipped, too short / too few helical dihedrals: {skipped})")
    if n == 0:
        print("  ⛔ NOTHING SCORED -- do not read this as a result.")
        return
    # The 0.5 split is the midpoint of the calibrated 0.12 / 0.89 modes, not a tuned threshold.
    mirrored = int((a > 0.5).sum())
    print(f"  helix_pos_frac    : mean {a.mean():.4f}  median {np.median(a):.4f}  "
          f"min {a.min():.4f}  max {a.max():.4f}")
    print(f"  MIRRORED (>0.5)   : {mirrored}/{n} = {100.0 * mirrored / n:.1f}%")
    # ⛔ Lead with the modal split, not the mean: a bimodal set makes the mean meaningless.
    hist, edges = np.histogram(a, bins=10, range=(0.0, 1.0))
    print("  distribution:")
    for h, lo in zip(hist, edges[:-1]):
        bar = "#" * int(60 * h / max(hist.max(), 1))
        print(f"    {lo:.1f}-{lo + 0.1:.1f} {h:6d} {bar}")


if __name__ == "__main__":
    main()
