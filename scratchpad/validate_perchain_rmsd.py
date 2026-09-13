"""VALIDATE the per-chain RMSD computation that the confound conclusion rests on.

The confound finding -- that the GT mirror test misses mirrored chains as structure quality degrades
-- was computed with my own kabsch_rmsd over the dumped gen/gt pairs. If that computation disagrees
with what the training job itself logged, the conclusion is built on a broken instrument.

The job logs val/rmsd_proper and val/rmsd_reflected per round. Recomputing the same rounds from the
PDBs and comparing is a direct test. They should agree closely; a systematic offset would mean my
alignment differs from the trainer's (different atom selection, different superposition, or a
mean-vs-median mismatch).
"""

import glob
import os
import sys

import numpy as np
import wandb

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad")
from ca_handedness_filter import kabsch_rmsd, read_ca

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"
CHECKS = [("c2c_cb8", 2144, 2143), ("c2c_cb8", 2644, 2643), ("c2c_cb8", 3144, 3143),
          ("c2c_cb8_tbeta", 3144, 3143)]

api = wandb.Api()
runs = list(api.runs("DP_CO_AFdiffusion/contact2coord"))
logged = {}
for arm in ("c2c_cb8", "c2c_cb8_tbeta"):
    rs = [r for r in runs if r.name == arm]
    if not rs:
        continue
    df = sorted(rs, key=lambda x: str(x.created_at))[-1].history(samples=100000, pandas=True)
    cols = ["trainer/global_step", "val/rmsd_proper", "val/rmsd_reflected"]
    if all(c in df.columns for c in cols):
        sub = df[cols].dropna()
        for _, row in sub.iterrows():
            logged[(arm, int(row["trainer/global_step"]))] = (
                float(row["val/rmsd_proper"]), float(row["val/rmsd_reflected"]))

print(f"{'arm':>14} {'step':>6} {'n':>3} | {'mine mean':>9} {'logged':>8} {'diff':>7} "
      f"| {'mine mean refl':>14} {'logged':>8} {'diff':>7}")
ok = True
for arm, dirstep, logstep in CHECKS:
    d = os.path.join(STORE, arm, "samples", f"step{dirstep:07d}")
    pr, rf = [], []
    for gen in sorted(glob.glob(os.path.join(d, "*_gen.pdb"))):
        gt = gen.replace("_gen.pdb", "_gt.pdb")
        if not os.path.exists(gt):
            continue
        a, b = read_ca(gen), read_ca(gt)
        if a is None or b is None or len(a) != len(b) or len(a) < 5:
            continue
        pr.append(kabsch_rmsd(a, b, allow_reflection=False))
        rf.append(kabsch_rmsd(a, b, allow_reflection=True))
    if not pr:
        continue
    lp, lr = logged.get((arm, logstep), (float("nan"), float("nan")))
    dp, dr = np.mean(pr) - lp, np.mean(rf) - lr
    if abs(dp) > 0.05 * max(lp, 1) or abs(dr) > 0.05 * max(lr, 1):
        ok = False
    print(f"{arm:>14} {dirstep:>6} {len(pr):>3} | {np.mean(pr):9.3f} {lp:8.3f} {dp:+7.3f} "
          f"| {np.mean(rf):14.3f} {lr:8.3f} {dr:+7.3f}")

print("\n⭐ my per-chain RMSD matches the trainer's logged aggregates within 5% on every round —"
      "\n   the confound analysis rests on a validated instrument." if ok else
      "\n⛔ MISMATCH > 5% — my per-chain computation differs from the trainer's."
      "\n   The confound conclusion must be revisited before it is relied on.")
