"""Do mirrored generations sit in FORBIDDEN Ramachandran space? A physics-grounded second detector.

This follows from the chimera result: mirrored samples keep L residues (frac-L 0.9988) but build a
left-handed fold. For a genuine L-amino acid that combination is sterically strained -- the
right-handed alpha region is phi<0, and its mirror (phi>0) is essentially forbidden except for
glycine. So a mirrored chain should show anomalously many POSITIVE-phi residues.

Two payoffs if it holds:
  1. A second detector INDEPENDENT of the CA pseudo-dihedral, so the two can be combined -- useful
     because the CA detector false-positives on ~2% of natives.
  2. Evidence that a PHYSICS term (Ramachandran / clash) would be chirality-sensitive, which matters
     because such a term is grounded in stereochemistry rather than needing an invented weight.

⛔ Threshold-free reporting: the phi<0 fraction is reported per population and glycine is EXCLUDED
(it is achiral and genuinely populates phi>0). Natives define the reference rate; nothing is assumed.
"""

import argparse
import glob
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.utils.c2c_dump import _ca_dihedrals


def read_residues(path):
    res, cur, key, name = [], {}, None, None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            rid = line[22:27]
            if rid != key:
                if cur:
                    res.append((name, cur))
                cur, key, name = {}, rid, line[17:20].strip()
            cur[line[12:16].strip()] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if cur:
        res.append((name, cur))
    return res


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def phi_psi(path):
    res = read_residues(path)
    phis = []
    for i in range(1, len(res)):
        nm_prev, prev = res[i - 1]
        nm, cur = res[i]
        if nm == "GLY":
            continue  # achiral: genuinely populates phi>0, would dilute the signal
        if not all(a in prev for a in ("C",)) or not all(a in cur for a in ("N", "CA", "C")):
            continue
        p = dihedral(np.asarray(prev["C"]), np.asarray(cur["N"]),
                     np.asarray(cur["CA"]), np.asarray(cur["C"]))
        phis.append(p)
    return np.asarray(phis)


def ca_trace(path):
    return np.asarray([c["CA"] for _, c in read_residues(path) if "CA" in c])


def hand(path):
    d = _ca_dihedrals(ca_trace(path))
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def summarise(paths, label):
    rows = []
    for p in paths:
        ph = phi_psi(p)
        h = hand(p)
        if not len(ph) or np.isnan(h):
            continue
        rows.append((h, float((ph > 0).mean()), float(np.median(ph))))
    a = np.asarray(rows)
    if not len(a):
        print(f"=== {label} === NOTHING SCORED"); return None
    mir = a[:, 0] > 0.5
    print(f"\n=== {label} ===  n={len(a)}  mirrored-by-CA-detector: {int(mir.sum())}")
    for sel, nm in ((~mir, "fold RIGHT-handed"), (mir, "fold MIRRORED")):
        if sel.sum():
            print(f"  {nm:<20} n={int(sel.sum()):4d}  frac phi>0 : mean {a[sel,1].mean():.4f}  "
                  f"median {np.median(a[sel,1]):.4f}   median phi {np.median(a[sel,2]):+7.1f} deg")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_254")
    args = ap.parse_args()
    summarise(sorted(glob.glob(f"{args.gen_dir}/*_gt.pdb")), "NATIVES (reference phi>0 rate)")
    g = summarise(sorted(glob.glob(f"{args.gen_dir}/*_gen.pdb")), "c2c GENERATED")
    if g is not None:
        # How well does phi>0 alone separate the two fold classes the CA detector found?
        mir = g[:, 0] > 0.5
        if mir.sum() and (~mir).sum():
            from numpy import argsort
            x, y = g[:, 1], mir.astype(float)
            order = argsort(x)
            xs, ys = x[order], y[order]
            # AUC via rank statistic -- no threshold chosen.
            ranks = np.arange(1, len(xs) + 1)
            n1, n0 = ys.sum(), (1 - ys).sum()
            auc = (ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            print(f"\n  AUC of 'frac phi>0' for predicting the CA-detector's mirrored label: {auc:.4f}")
            print("  (0.5 = no information, 1.0 = perfect separation; no threshold is chosen here)")


if __name__ == "__main__":
    main()
