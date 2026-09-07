"""The honest number: accuracy on NOVEL-FOLD chains, after correcting handedness.

Two separate problems distort every headline figure this project has produced:
  - 90.2% of validation clusters have a same-fold training neighbour, so aggregate TM is fold recall;
  - ~86% of chains are coin flips on handedness, so half the samples are reflections.

Neither is visible in a mean TM. This strips both: restrict to chains with no close training fold,
and apply the native-free handedness detector, then report what is left.

⛔ Reports RMSD, not TM. RMSD after reflection is already computed (the improper-rotation Kabsch);
recomputing TM would need USalign over reflected PDBs, which is a separate job. Do not quote a
"corrected TM" from this script -- it does not produce one.
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_handedness_filter import helical_score, kabsch_rmsd, read_ca  # noqa: E402

MAPROW = re.compile(r"^\s*(\d+)\s+(gen\d+)\s+(\S+)\s+(\d+)\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_254")
    ap.add_argument("--order_log",
                    default="/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/valids-22216500.out")
    ap.add_argument("--novel",
                    default="/orcd/scratch/orcd/011/chenxiou/novel_fold_val_chains.txt")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    order = {}
    for line in open(args.order_log):
        m = MAPROW.match(line)
        if m:
            order[m.group(2)] = (m.group(3), int(m.group(4)))
    novel = {l.strip() for l in open(args.novel) if l.strip()}

    # Native side of the threshold, calibrated on real structures rather than assumed.
    nat = []
    for tp in sorted(glob.glob(os.path.join(args.dir, "*_gt.pdb"))):
        t = read_ca(tp)
        if len(t) >= 40:
            sc, _ = helical_score(t)
            if sc is not None:
                nat.append(sc)
    native_low = float(np.mean(nat)) < args.threshold
    flag = (lambda v: v >= args.threshold) if native_low else (lambda v: v < args.threshold)

    rows = []
    for gp in sorted(glob.glob(os.path.join(args.dir, "*_gen.pdb"))):
        label = os.path.basename(gp)[:-8]
        tp = gp.replace("_gen.pdb", "_gt.pdb")
        if label not in order or not os.path.exists(tp):
            continue
        cid, L_map = order[label]
        g, t = read_ca(gp), read_ca(tp)
        n = min(len(g), len(t))
        if n < 40:
            continue
        g, t = g[:n], t[:n]
        if n != L_map:
            print(f"ABORT: {label} length {n} != map {L_map}; loader orders differ", file=sys.stderr)
            return 1
        sc, _ = helical_score(g)
        if sc is None:
            continue
        prop = kabsch_rmsd(g, t)
        refl = kabsch_rmsd(g, t, allow_reflection=True)
        corrected = refl if flag(sc) else prop
        rows.append((cid, n, prop, corrected, cid in novel))

    allr = rows
    nov = [r for r in rows if r[4]]
    con = [r for r in rows if not r[4]]

    print(f"{'set':>26} {'n':>5} {'RMSD as generated':>19} {'RMSD hand-corrected':>21}")
    for lab, grp in (("all validation clusters", allr),
                     ("contaminated (fold-mate)", con),
                     ("NOVEL FOLD (honest)", nov)):
        if grp:
            b = np.mean([r[2] for r in grp])
            a = np.mean([r[3] for r in grp])
            print(f"{lab:>26} {len(grp):>5} {b:>19.2f} {a:>21.2f}")

    if nov:
        print(f"\nnovel-fold chains, hand-corrected RMSD, worst to best:")
        for cid, L, prop, corr, _ in sorted(nov, key=lambda x: -x[3]):
            print(f"  {cid:>9}  L={L:>3}  {prop:>6.2f} -> {corr:>6.2f} A")
        good = sum(1 for r in nov if r[3] < 3.0)
        print(f"\n  {good}/{len(nov)} novel-fold chains under 3 A after correction")
    print("\n⚠️ RMSD only. A corrected TM would need USalign rerun over reflected PDBs.")
    print("⚠️ The novel-fold set is itself a LOWER bound on novelty: only 13152 of 259k training")
    print("   chains were searched, so some of these may have a closer neighbour that was not seen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
