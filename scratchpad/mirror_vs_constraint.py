"""Is the mirror rate independent of how CONSTRAINED the chain is? A prediction of the mechanism.

If the fold hand is read out of the input noise (measured: the sampler is reflection-equivariant),
then it must NOT depend on how much the conditioning constrains the structure. A denser contact map
or a longer chain pins the fold more tightly in every achiral respect, but pins the HAND not at all.

So this is a falsification test, not a fishing expedition:
  - mechanism TRUE  -> mirror rate flat vs length and vs native contact density
  - mechanism FALSE -> sparsely-constrained chains mirror more, and "require denser maps" would be a
                       cheap practical mitigation worth having

Contact density is computed from the NATIVE CA coordinates (pairs within 8 A, |i-j|>=3) as a proxy
for how constrained the conditioning is. It is a proxy, not the ConFind map actually used for
conditioning -- stated so the number is not over-read.

⛔ Reported as binned rates with counts, not a correlation coefficient alone: a single r over a
bimodal outcome hides which bins actually move. Counts are shown so small bins are visible as small.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.utils.c2c_dump import _ca_dihedrals


def ca_from_pdb(path):
    ca = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(ca, dtype=np.float64)


def hand(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def contact_density(ca):
    """Fraction of |i-j|>=3 CA pairs within 8 A -- a proxy for conditioning constraint."""
    n = len(ca)
    if n < 8:
        return float("nan")
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None]) >= 3
    return float((d[sep] < 8.0).mean())


def binned(x, y, name, nbins=5):
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    print(f"\n  mirror rate vs {name} (quintiles):")
    for i in range(nbins):
        lo, hi = qs[i], qs[i + 1]
        sel = (x >= lo) & (x <= hi if i == nbins - 1 else x < hi)
        if sel.sum():
            print(f"    {name:>16} {lo:8.3f}-{hi:8.3f}  n={int(sel.sum()):4d}  "
                  f"mirrored {100*float((y[sel] > 0.5).mean()):5.1f}%")
    r = float(np.corrcoef(x, (y > 0.5).astype(float))[0, 1])
    print(f"    point-biserial r({name}, mirrored) = {r:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_254")
    args = ap.parse_args()
    rows = []
    for g in sorted(glob.glob(f"{args.gen_dir}/*_gen.pdb")):
        t = g.replace("_gen.pdb", "_gt.pdb")
        if not os.path.exists(t):
            continue
        cg, ct = ca_from_pdb(g), ca_from_pdb(t)
        h = hand(cg)
        if np.isnan(h):
            continue
        rows.append((len(ct), contact_density(ct), h))
    a = np.asarray(rows)
    print(f"n = {len(a)} chains   overall mirrored: "
          f"{100*float((a[:,2] > 0.5).mean()):.1f}%")
    binned(a[:, 0], a[:, 2], "length")
    binned(a[:, 1], a[:, 2], "contact_density")
    print("\n  Mechanism predicts BOTH to be flat: the conditioning is achiral, so it cannot")
    print("  determine the hand no matter how tightly it constrains everything else.")


if __name__ == "__main__":
    main()
