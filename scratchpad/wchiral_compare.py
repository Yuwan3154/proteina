"""Compare the production w_chiral arms against their parent trajectory, aligned on the LINEAGE.

The arms (c2c_cb8_wch03, c2c_cb8_wch10) fork from a FROZEN branch point of c2c_cb8_tbeta, so an
arm's local step 0 is NOT lineage step 0. Plotting local step against the parent's step would
compare an arm's first hour against the parent's first hour and call the difference an effect.

⛔⛔ THIS EXACT TRAP HAS ALREADY FLIPPED A SIGN ON THIS PROJECT. On a previous warm-started pair the
naive axis said the child led 14/15 rounds (+0.00545); aligned on the lineage it led 2/15 (-0.00447).
The sign REVERSED. See [[feedback_align_warm_started_runs_on_the_lineage]].

⛔ The offset is READ, never assumed: each arm's own job log prints
`[warm-start] ... from <ckpt> (step N)`, and N is the offset actually used. If no arm log carries
that line the script refuses to run rather than guessing 5500.

⛔ Judged on the REFLECTION-GAP SIGN and on the dist_mae / reflected-RMSD pair. NOT on the
CA-dihedral statistic the chiral loss itself trains -- that would be circular.
⚠️ Round rates are noisy: validation draws DIFFERENT chains each round (measured, job 22866078), so
a single round's rate is not comparable across runs. Bin over lineage steps and print n per cell.

Usage: wchiral_compare.py c2c_cb8_tbeta c2c_cb8_wch03 c2c_cb8_wch10
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ca_handedness_filter import read_ca
from proteinfoundation.utils.c2c_dump import handedness_metrics

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"
LOGS = os.path.join(STORE, "logs")

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="first is the PARENT (offset 0), rest are the arms")
ap.add_argument("--bin", type=int, default=500, help="lineage-step bin width")
args = ap.parse_args()


def lineage_offset(run):
    """Offset = the step the arm warm-started FROM, read out of its own logs. 0 for the parent."""
    best = None
    for lg in sorted(glob.glob(os.path.join(LOGS, f"{run}-*.out"))):
        for line in open(lg, errors="ignore"):
            m = re.search(r"\[warm-start\].*\(step (\d+)\)", line)
            if m:
                v = int(m.group(1))
                # every segment re-prints the same branch point; they must agree
                if best is not None and v != best:
                    raise SystemExit(f"⛔ {run}: conflicting warm-start steps {best} vs {v} "
                                     f"-- cannot align, resolve before comparing")
                best = v
    return best


def load(run, offset):
    rows = []
    for rd in sorted(glob.glob(os.path.join(STORE, run, "samples", "step*"))):
        local = int(os.path.basename(rd).replace("step", ""))
        per = []
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
            per.append((1 if h["rmsd_proper"] > h["rmsd_reflected"] else 0,
                        h["rmsd_proper"], h["rmsd_reflected"], dm))
        if per:
            a = np.array(per, dtype=float)
            rows.append((local + offset, local, len(a), a[:, 0].mean(),
                         a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()))
    return rows


parent, arms = args.runs[0], args.runs[1:]
offsets, data = {}, {}
for i, run in enumerate(args.runs):
    off = 0 if i == 0 else lineage_offset(run)
    if i > 0 and off is None:
        raise SystemExit(f"⛔ {run}: no '[warm-start] ... (step N)' line in its logs. Refusing to "
                         f"assume an offset -- the comparison would be meaningless if wrong.")
    offsets[run] = off
    data[run] = load(run, off)
    print(f"{run}: offset {off:+d}, {len(data[run])} rounds"
          + (f", lineage steps {data[run][0][0]}-{data[run][-1][0]}" if data[run] else ""))

live = [r for r in args.runs if data[r]]
if len(live) < 2:
    raise SystemExit(f"\n⛔ only {len(live)} run(s) have dumped structures "
                     f"({', '.join(live) or 'none'}). The arms have not produced validation output "
                     f"yet -- nothing to compare. This is a DATA state, not a null result.")

lo = max(min(r[0] for r in data[run]) for run in live)
hi = min(max(r[0] for r in data[run]) for run in live)
print(f"\n⛔ OVERLAPPING lineage range across all arms: {lo}-{hi}")
if hi <= lo:
    raise SystemExit("⛔ no overlapping lineage range yet -- the arms have not caught up to the "
                     "parent's trajectory. Comparing outside the overlap compares different "
                     "training amounts, not different losses.")

edges = list(range(int(lo), int(hi) + args.bin, args.bin))
print(f"\n{'lineage bin':<16}" + "".join(f"{r[-10:]:>26}" for r in live))
print(f"{'':16}" + "".join(f"{'refl-sign  n  refl  dmae':>26}" for _ in live))
for a, b in zip(edges, edges[1:]):
    cells = []
    for run in live:
        sel = [r for r in data[run] if a <= r[0] < b]
        if not sel:
            cells.append(f"{'-':>26}")
            continue
        n = sum(r[2] for r in sel)
        rs = float(np.average([r[3] for r in sel], weights=[r[2] for r in sel]))
        rf = float(np.average([r[5] for r in sel], weights=[r[2] for r in sel]))
        dm = float(np.average([r[6] for r in sel], weights=[r[2] for r in sel]))
        cells.append(f"{rs:>10.3f} {n:>3} {rf:>6.2f} {dm:>5.2f}")
    print(f"{f'{a}-{b}':<16}" + "".join(cells))

print("\n⚠️ Read the dist_mae / reflected-RMSD pair, not a single round's rate: validation draws")
print("   different chains each round, so per-round rates are not comparable across runs.")
print("⚠️ Every arm here also crossed the pseudo-CB contact-definition change; the parent crossed it")
print("   at lineage step 7076, the arms started after it. Do not attribute a step at 7076 to w_chiral.")
