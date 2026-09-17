"""Compare the mirror rate between runs AT MATCHED FIT QUALITY, not at matched step.

⛔⛔ WHY THIS TOOL EXISTS. The mirror statistics are FIT-QUALITY-BIASED. `is_mirrored` needs
proper > 2x reflected, so a poor structure -- where both superpositions are bad and neither
dominates -- cannot trigger it, and `gap_frac` is likewise compressed when both fits are bad. So a
run whose dist_mae is WORSE will look LESS mirrored for purely mechanical reasons.

That is exactly the situation at the first FAPE read: local step 500 showed is_mirrored 0.312 and
gap_frac 0.250 against the parent's 0.562 / 0.472 -- but dist_mae had degraded 1.13 -> 2.11, because
the arm warm-started with a fresh optimizer and is 25% through warmup=2000. Every mirror number there
is confounded with fit quality, in the direction that flatters the treatment.

⭐ THE FIX: bin every generation by its OWN dist_mae and compare runs WITHIN a bin. Inside a bin the
structures are comparably good, so a difference in mirror rate cannot be a fit-quality artefact.

⛔ Reports the per-bin n for BOTH runs. A bin where one run has 2 generations and the other has 60
is not a comparison, and the counts are the only way to see that.

Usage: mirror_vs_fitquality.py c2c_cb8_tbeta c2c_cb8_fape [--min_step 5000]
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri")
sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad")

from ca_handedness_filter import read_ca
from proteinfoundation.utils.c2c_dump import handedness_metrics

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
# ⛔ Early rounds are unusable for ANY mirror question: before the model folds at all both
# superpositions are ~equally bad and the reflection sign is noise about nothing.
ap.add_argument("--min_step", type=int, default=0,
                help="ignore rounds below this LOCAL step (per run)")
ap.add_argument("--bins", default="0,1.25,1.5,2,3,5,1e9",
                help="dist_mae bin edges, in Angstrom")
args = ap.parse_args()

EDGES = [float(v) for v in args.bins.split(",")]


def load(run):
    rows = []
    for rd in sorted(glob.glob(os.path.join(STORE, run, "samples", "step*"))):
        st = int(os.path.basename(rd).replace("step", ""))
        if st < args.min_step:
            continue
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
            pr, rf = h["rmsd_proper"], h["rmsd_reflected"]
            rows.append((st, dm, 1 if pr > rf else 0, h["is_mirrored"],
                         (pr - rf) / pr if pr > 0 else 0.0, pr, rf))
    return np.array(rows) if rows else np.zeros((0, 7))


data = {r: load(r) for r in args.runs}
for r, a in data.items():
    print(f"{r}: {len(a)} generations" +
          (f", local steps {a[:,0].min():.0f}-{a[:,0].max():.0f}, "
           f"dist_mae {a[:,1].min():.2f}-{a[:,1].max():.2f}" if len(a) else ""))

live = [r for r in args.runs if len(data[r])]
assert len(live) >= 2, (f"need >=2 runs with generations, got {len(live)} ({live}). "
                        f"This is a DATA state, not a result.")

print(f"\n{'dist_mae bin':>16}" + "".join(f"{r[-12:]:>30}" for r in live))
print(f"{'':16}" + "".join(f"{'n   refl-sign  is_mirr':>30}" for _ in live))
for lo, hi in zip(EDGES, EDGES[1:]):
    cells = []
    for r in live:
        a = data[r]
        sel = (a[:, 1] >= lo) & (a[:, 1] < hi)
        if sel.sum() == 0:
            cells.append(f"{'-':>30}")
        else:
            cells.append(f"{int(sel.sum()):>5} {a[sel,2].mean():>10.3f} {a[sel,3].mean():>10.3f}")
    lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
    print(f"{lab:>16}" + "".join(cells))

print("\n⭐ Compare DOWN a column only where BOTH runs have a usable n in the SAME row. A difference")
print("   in a shared bin cannot be a fit-quality artefact, because the structures are comparably")
print("   good by construction. A difference visible only ACROSS rows is exactly the artefact.")
print("⚠️ If the runs do not OVERLAP in any bin, the comparison cannot be made yet -- that is the")
print("   expected state while a warm-restarted arm is still climbing back to the parent's dist_mae.")
