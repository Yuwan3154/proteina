"""DOUBLE-CHECK the control's mirror rate with a SECOND, independent definition.

The control's 39.6% is the anchor the whole t_beta experiment will be judged against, and so far it
comes from ONE definition: val/is_mirrored, which is GT-based (proper RMSD > 2x the
reflection-allowed RMSD AND gap > 1 A). If that definition were biased, the comparison inherits it.

The CA-dihedral detector is genuinely independent -- native-free, different input (dumped PDB
geometry rather than RMSD scalars), different code path. Agreement between them is corroboration;
disagreement would mean the anchor needs care.

⛔ Only RESOLVED rounds (step >= 2143) are counted. Pooling the amorphous phase gave the retracted
17.5%.
"""

import glob
import os
import sys

import numpy as np
import wandb

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad")
from ca_handedness_filter import helical_score, read_ca

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"
ARM = "c2c_cb8"
RESOLVED_STEP = 2143
MIRROR_CUT = 0.65      # calibrated: natives median 0.0815, mirrored generations ~0.893
N_PER_ROUND = 16

api = wandb.Api()
rs = [r for r in api.runs("DP_CO_AFdiffusion/contact2coord") if r.name == ARM]
r = sorted(rs, key=lambda x: str(x.created_at))[-1]
df = r.history(samples=100000, pandas=True)
sub = df[["trainer/global_step", "val/is_mirrored"]].dropna()
gt = {int(a): float(b) for a, b in zip(sub["trainer/global_step"], sub["val/is_mirrored"])}

print(f"{'step':>7} {'GT is_mirrored':>15} {'detector >0.65':>15}")
gt_tot = det_tot = n_tot = 0
for d in sorted(glob.glob(os.path.join(STORE, ARM, "samples", "step*"))):
    step = int(os.path.basename(d).replace("step", ""))
    if step < RESOLVED_STEP:
        continue
    scores = []
    for f in sorted(glob.glob(os.path.join(d, "*_gen.pdb"))):
        ca = read_ca(f)
        if ca is None or len(ca) < 5:
            continue
        sc = helical_score(ca)
        sc = sc[0] if isinstance(sc, (tuple, list)) else sc
        if sc == sc:
            scores.append(float(sc))
    if not scores:
        continue
    g = np.array(scores)
    det_n = int((g > MIRROR_CUT).sum())
    gtv = gt.get(step, gt.get(step - 1))
    gt_n = round(gtv * N_PER_ROUND) if gtv is not None else None
    print(f"{step:>7} {f'{gt_n}/{len(g)}':>15} {f'{det_n}/{len(g)}':>15}")
    if gt_n is not None:
        gt_tot += gt_n
    det_tot += det_n
    n_tot += len(g)

print(f"\nGT-based   : {gt_tot}/{n_tot} = {100*gt_tot/n_tot:.1f}%")
print(f"detector   : {det_tot}/{n_tot} = {100*det_tot/n_tot:.1f}%")
print(f"difference : {abs(gt_tot - det_tot)} chain(s) of {n_tot}")
se = (0.5 * 0.5 / n_tot) ** 0.5
print(f"\n1 SE at n={n_tot} is {100*se:.1f}pp, so the two definitions agree well inside sampling noise."
      if abs(gt_tot - det_tot) / n_tot < se else
      f"\n⚠️ the two definitions differ by more than 1 SE ({100*se:.1f}pp) -- the anchor needs care.")
