"""Paired mirror-rate comparison: c2c_cb8 (standard noise) vs c2c_cb8_tbeta (high-noise training).

The question is whether emphasising the high-noise regime -- where the model must GENERATE a hand
rather than copy one out of the noised input -- reduces the ~50% mirror coin flip. Both arms are
lDDT-OFF and share every other setting, and their effective batch is identical (8 structures x 48
samples per step: 2 GPU x accum 4 vs 1 GPU x accum 8), so MATCHED STEP means matched data seen.

Three traps this exists to avoid, each of which has already produced a wrong reading once:

 1. ⛔ AMORPHOUS vs BIMODAL. Early in training every helix_pos_frac sits at 0.50 +/- 0.04 because
    there is no resolvable handedness yet, and `is_mirrored` then reads 0/16 -- which is NOT "no
    mirroring". Pooling those rounds gave "17.5% mirrored" for a run that is at the coin flip.
    Handedness did not resolve in c2c_cb8 until ~step 2143.
 2. ⛔ n=16 per round, so a rate moves in 6.25% steps and one round cannot separate 50% from 62%.
    Lead with COUNTS and pool only rounds that are past the amorphous phase.
 3. ⛔ Matched steps, not matched wall-clock. The t_beta arm runs on 1 GPU and is ~2x slower.

Calibration (memory project_c2c_mirror_coinflip_and_detector): natives median helix_pos_frac 0.0815;
mirrored generations ~0.893; reference first-draw mirror rate 51.8% at n=254.
"""

import glob
import os
import sys

import numpy as np
import wandb

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad")
from ca_handedness_filter import helical_score, read_ca

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"
ARMS = {"c2c_cb8": "standard noise (control)", "c2c_cb8_tbeta": "t_beta 1.3,2.0 (high noise)"}
RESOLVED_STEP = 2143      # measured: where handedness first resolved in c2c_cb8
NATIVE, MIRROR = 0.35, 0.65
N_PER_ROUND = 16


def detector_rounds(arm):
    """Per-round handedness distribution from the dumped structures."""
    out = []
    for d in sorted(glob.glob(os.path.join(STORE, arm, "samples", "step*"))):
        step = int(os.path.basename(d).replace("step", ""))
        g = []
        for f in sorted(glob.glob(os.path.join(d, "*_gen.pdb"))):
            ca = read_ca(f)
            if ca is None or len(ca) < 5:
                continue
            sc = helical_score(ca)
            sc = sc[0] if isinstance(sc, (tuple, list)) else sc
            if sc == sc:
                g.append(float(sc))
        if g:
            out.append((step, np.array(g)))
    return out


def wandb_rounds(arm):
    api = wandb.Api()
    rs = [r for r in api.runs("DP_CO_AFdiffusion/contact2coord") if r.name == arm]
    if not rs:
        return []
    import pandas as pd
    frames = [r.history(samples=100000, pandas=True) for r in rs]
    df = pd.concat([f for f in frames if "trainer/global_step" in f.columns], ignore_index=True)
    if "val/is_mirrored" not in df.columns:
        return []
    sub = df[["trainer/global_step", "val/is_mirrored"]].dropna().sort_values("trainer/global_step")
    return list(zip(sub["trainer/global_step"].to_numpy(), sub["val/is_mirrored"].to_numpy()))


for arm, label in ARMS.items():
    print(f"\n{'='*74}\n{arm} -- {label}\n{'='*74}")
    det = detector_rounds(arm)
    wb = dict(wandb_rounds(arm))
    if not det:
        print("  no dumped samples yet")
        continue
    print(f"  {'step':>7} {'n':>3} {'mean':>6} {'native':>7} {'MIRROR':>7} {'mid':>4} {'spread':>7}"
          f" {'is_mirrored':>12}  phase")
    resolved = []
    for step, g in det:
        lo, hi = int((g < NATIVE).sum()), int((g > MIRROR).sum())
        mid = len(g) - lo - hi
        amorph = mid > lo + hi
        wv = wb.get(step, wb.get(step - 1))
        wstr = f"{round(wv*N_PER_ROUND):>2}/{N_PER_ROUND}" if wv is not None else "   -"
        print(f"  {step:>7} {len(g):>3} {g.mean():6.3f} {lo:>7} {hi:>7} {mid:>4} {g.std():7.3f}"
              f" {wstr:>12}  {'AMORPHOUS' if amorph else 'resolved'}")
        if not amorph and step >= RESOLVED_STEP and wv is not None:
            resolved.append(wv)
    if resolved:
        n = len(resolved) * N_PER_ROUND
        rate = float(np.mean(resolved))
        se = (rate * (1 - rate) / n) ** 0.5
        print(f"\n  POOLED over {len(resolved)} RESOLVED rounds (step >= {RESOLVED_STEP}) "
              f"= {n} samples: {round(rate*n)}/{n} = {100*rate:.1f}% +/- {100*se:.1f}% (1 SE)")
        print(f"  reference coin flip 51.8% (n=254) | natives 2.0% | Proteina 5.7%")
        if rate + 2 * se < 0.40:
            print("  ⭐ BELOW the coin flip by >2 SE -- high-noise training moved it")
        else:
            print("  → NOT distinguishable from the coin flip at this sample size")
    else:
        print(f"\n  ⛔ NO RESOLVED ROUNDS past step {RESOLVED_STEP} yet -- verdict NOT readable.")
        print("     Quoting a rate from the amorphous phase would repeat the 17.5% error.")
