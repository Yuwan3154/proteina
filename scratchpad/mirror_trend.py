"""Per-validation-round mirror table for a c2c run, from its own dumped structures.

⛔ Reads the dumped PDBs, NOT wandb. Same source as the step-6432 table already reported, so the
numbers stay directly comparable; and it needs no API token, so it runs as an ordinary CPU job.

⛔ Reports the COUNT per round beside every rate. With a handful of chains per round a rate moves in
steps of 1/n, so one round's rate cannot be quoted as "the" mirror rate -- that is how a 0.625 and a
0.500 get read as a trend when they differ by one structure.

⛔⛔ PSEUDO-CB CHANGEOVER. The contact definition (the c2c INPUT conditioning) changed mid-run. Pass
--changeover <step> and any round at or after it is flagged, because a move across that line has a
second candidate explanation and must not be attributed to the science unexamined.

Usage: mirror_trend.py <run_name> [--changeover 7076]
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
ap.add_argument("run")
ap.add_argument("--changeover", type=int, default=None,
                help="step at which the contact definition changed; rounds >= it are flagged")
args = ap.parse_args()

rounds = sorted(glob.glob(os.path.join(STORE, args.run, "samples", "step*")))
assert rounds, f"no sample rounds under {STORE}/{args.run}/samples"

rows = []
skipped = []
for rd in rounds:
    st = int(os.path.basename(rd).replace("step", ""))
    per = []
    for gp in sorted(glob.glob(os.path.join(rd, "*_gen.pdb"))):
        tp = gp.replace("_gen.pdb", "_gt.pdb")
        if not os.path.exists(tp):
            skipped.append((os.path.basename(gp), "no _gt.pdb"))
            continue
        g, t = read_ca(gp), read_ca(tp)
        n = min(len(g), len(t))
        if n < 10:
            skipped.append((os.path.basename(gp), f"n={n}"))
            continue
        h = handedness_metrics(g[:n], t[:n])
        if not h:
            skipped.append((os.path.basename(gp), "handedness_metrics returned nothing"))
            continue
        pr, rf = h["rmsd_proper"], h["rmsd_reflected"]
        dm = float(np.abs(np.linalg.norm(g[:n, None] - g[None, :n], axis=-1)
                          - np.linalg.norm(t[:n, None] - t[None, :n], axis=-1)).mean())
        per.append((pr, rf, h["is_mirrored"], (pr - rf) / pr if pr > 0 else 0.0, dm))
    if per:
        a = np.array(per)
        rows.append((st, len(a), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(),
                     a[:, 3].mean(), a[:, 4].mean(), float((a[:, 0] > a[:, 1]).mean())))

# ⛔ Every skip printed, not counted. A count is not evidence that the skips were benign.
print(f"[skips] {len(skipped)}")
for name, why in skipped[:20]:
    print(f"    {name}: {why}")

assert rows, "no scorable rounds -- the table would be vacuous, not clean"
print(f"\n=== {args.run}: mirror trend over {len(rows)} validation rounds ===")
print(f"{'step':>7} {'n':>3} {'proper':>8} {'reflect':>8} {'is_mirr':>8} {'gap_frac':>9} "
      f"{'dist_mae':>9} {'refl-sign':>10}")
for st, n, pr, rf, im, gf, dm, rs in rows:
    flag = ""
    if args.changeover is not None and st >= args.changeover:
        flag = "  <- at/after pseudo-CB changeover"
    print(f"{st:>7} {n:>3} {pr:>8.2f} {rf:>8.2f} {im:>8.3f} {gf:>9.3f} {dm:>9.2f} {rs:>10.3f}{flag}")

tot = sum(r[1] for r in rows)
print(f"\npooled over all rounds: {tot} generations")
print("  refl-sign = fraction where the REFLECTED superposition fits better (the non-circular read);")
print("  is_mirrored is stricter (needs proper > 2x reflected) and under-reports on poor structures.")
if args.changeover is not None:
    pre = [r for r in rows if r[0] < args.changeover]
    post = [r for r in rows if r[0] >= args.changeover]
    print(f"\n⛔ rounds before changeover: {len(pre)}   at/after: {len(post)}")
    if pre and post:
        print(f"   refl-sign  before {np.mean([r[7] for r in pre]):.3f}  "
              f"after {np.mean([r[7] for r in post]):.3f}   "
              f"-- ⚠️ a move here has TWO candidate causes, do not attribute it to training alone")
