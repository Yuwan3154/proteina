"""Matched-step comparison of overfit chirality arms, over the FULL trajectory.

Supersedes /orcd/scratch/orcd/011/chenxiou/arm_compare.py, whose BINS stopped at 1200 and whose
pooled z-test looked only at steps 0-600. Both arms now run past 1900, so the old tool silently
truncated most of the evidence.

⛔⛔ Two traps this fixes, both of which have burned this project already:
 1. NEVER QUOTE A STAT BEFORE THE FULL RANGE. Two earlier claims here ("won't self-resolve",
    "global handedness is not improving") were made on short windows and BOTH flipped sign when the
    range was extended. The bins now run to the end of the data.
 2. SAME POPULATION BEFORE COMPARING CURVES. The arms have different max steps (the baseline
    finished, the treatment is still running), so a pooled comparison over "everything" would
    compare different step distributions. Every pooled test here is restricted to the COMMON step
    range, and the per-arm n is printed for every cell so an unbalanced bin cannot hide.

⛔ Judged on the REFLECTION-GAP SIGN (is the improper superposition the better fit?), which is
independent of the CA-dihedral statistic the chiral loss trains on. helix_pos_frac would improve by
construction and would prove nothing.

Usage: arm_compare_full.py <name1> <name2> ...   (dirs under c2c_store; samples/ is appended)
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri")
sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad")

from ca_handedness_filter import read_ca
from proteinfoundation.utils.c2c_dump import handedness_metrics

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"
BINS = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600), (600, 800),
        (800, 1200), (1200, 1600), (1600, 2000), (2000, 2400)]


def load(name):
    rows = []
    for rd in sorted(glob.glob(os.path.join(STORE, name, "samples", "step*"))):
        st = int(os.path.basename(rd).replace("step", ""))
        for gp in sorted(glob.glob(os.path.join(rd, "*_gen.pdb"))):
            tp = gp.replace("_gen.pdb", "_gt.pdb")
            if not os.path.exists(tp):
                continue
            g, t = read_ca(gp), read_ca(tp)
            n = min(len(g), len(t))
            if n < 10:
                continue
            h = handedness_metrics(g[:n], t[:n])
            if not h:
                continue
            dm = float(np.abs(np.linalg.norm(g[:n, None] - g[None, :n], axis=-1)
                              - np.linalg.norm(t[:n, None] - t[None, :n], axis=-1)).mean())
            rows.append((st, h["rmsd_proper"] - h["rmsd_reflected"], h["is_mirrored"], dm))
    return np.array(rows) if rows else np.zeros((0, 4))


def ztest(p1, n1, p2, n2):
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = (pool * (1 - pool) * (1 / n1 + 1 / n2)) ** 0.5
    return (p1 - p2) / se if se > 0 else 0.0


arms = {n: load(n) for n in sys.argv[1:]}
for n, a in arms.items():
    print(f"{n}: {len(a)} generations" +
          (f", steps {a[:,0].min():.0f}-{a[:,0].max():.0f}" if len(a) else ""))

names = [n for n in arms if len(arms[n])]
assert len(names) >= 2, f"need >=2 non-empty arms, got {len(names)} -- refusing a vacuous comparison"

# ⛔ The common range, not the union. Beyond it only one arm has data.
COMMON = min(arms[n][:, 0].max() for n in names)
print(f"\n⛔ COMMON step range across all arms: 0-{COMMON:.0f} "
      f"(max per arm: {', '.join(f'{n}={arms[n][:,0].max():.0f}' for n in names)})")

print(f"\n{'bin':<12}" + "".join(f"{n[-8:]:>24}" for n in names))
print(f"{'':12}" + "".join(f"{'refl-sign   n   dmae':>24}" for _ in names))
for lo, hi in BINS:
    if lo > COMMON:
        continue
    cells = []
    for n in names:
        a = arms[n]
        s = (a[:, 0] >= lo) & (a[:, 0] < hi)
        cells.append(f"{'-':>24}" if s.sum() == 0 else
                     f"{np.mean(a[s,1] > 0):>11.3f} {int(s.sum()):>4} {a[s,3].mean():>7.2f}")
    print(f"{f'{lo}-{hi}':<12}" + "".join(cells))

# ── pooled tests ─────────────────────────────────────────────────────────────────────────────
# Two windows, reported TOGETHER and on purpose. 0-600 is where the baseline has not yet reached
# zero and an acceleration could be visible; it is also the window quoted before, so it stays
# comparable. The full common range is the honest whole-trajectory answer.
for label, lo, hi in [("0-600 (the window where acceleration could show)", 0, 600),
                      (f"0-{COMMON:.0f} (FULL common range)", 0, COMMON + 1)]:
    print(f"\n=== pooled over steps {label} ===")
    base = arms[names[0]]
    bs = (base[:, 0] >= lo) & (base[:, 0] < hi)
    if bs.sum() == 0:
        print("  baseline has no data in this window")
        continue
    p1, n1 = float(np.mean(base[bs, 1] > 0)), int(bs.sum())
    for n in names[1:]:
        a = arms[n]
        s = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if s.sum() == 0:
            continue
        p2, n2 = float(np.mean(a[s, 1] > 0)), int(s.sum())
        z = ztest(p1, n1, p2, n2)
        print(f"  {names[0]} {p1:.3f} (n={n1})  vs  {n} {p2:.3f} (n={n2})   z={z:+.2f}  "
              f"{'SIGNIFICANT' if abs(z) > 1.96 else 'not significant'}")

# ⭐ The quantity the user actually cares about: does the treatment reach zero SOONER?
print(f"\n=== first step bin at which the reflection-gap sign rate hits 0.000 ===")
for n in names:
    a = arms[n]
    hit = None
    for lo, hi in BINS:
        s = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if s.sum() and np.mean(a[s, 1] > 0) == 0.0:
            hit = (lo, hi, int(s.sum()))
            break
    print(f"  {n}: " + (f"bin {hit[0]}-{hit[1]} (n={hit[2]})" if hit else "never reaches 0.000"))
