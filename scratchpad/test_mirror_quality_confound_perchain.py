"""TEST the quality-confound claim PER CHAIN, instead of arguing it from round-level aggregates.

I claimed the GT is_mirrored test (proper RMSD > 2x reflection-allowed, gap > 1 A) fires less often on
poorly-formed structures, because both RMSDs then grow and the RATIO collapses toward 1. That was
argued from four round-level means. It is testable directly: the dumped gen/gt PDB pairs give a
per-chain proper and reflected RMSD, so the claim becomes a per-chain question.

If the confound is real:
  - chains the GT test flags as mirrored should have SMALLER proper RMSD than chains it does not, at
    equal true mirroring -- i.e. among chains the NATIVE-FREE detector calls mirrored, the GT test
    should miss the poor-quality ones;
  - the miss rate should rise with proper RMSD.

The detector is the reference here precisely because it never touches the native, so it cannot share
the ratio bias.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad")
from ca_handedness_filter import helical_score, kabsch_rmsd, read_ca

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"
MIRROR_CUT = 0.65
ROUNDS = [("c2c_cb8", 2144), ("c2c_cb8", 2644), ("c2c_cb8", 3144),
          ("c2c_cb8_tbeta", 3144)]

rows = []
for arm, step in ROUNDS:
    d = os.path.join(STORE, arm, "samples", f"step{step:07d}")
    for gen in sorted(glob.glob(os.path.join(d, "*_gen.pdb"))):
        gt = gen.replace("_gen.pdb", "_gt.pdb")
        if not os.path.exists(gt):
            continue
        a, b = read_ca(gen), read_ca(gt)
        if a is None or b is None or len(a) != len(b) or len(a) < 5:
            continue
        proper = kabsch_rmsd(a, b, allow_reflection=False)
        refl = kabsch_rmsd(a, b, allow_reflection=True)
        sc = helical_score(a)
        sc = sc[0] if isinstance(sc, (tuple, list)) else sc
        gt_flag = (proper > 2 * refl) and (proper - refl > 1.0)   # the val/is_mirrored rule
        rows.append((arm, step, proper, refl, sc, gt_flag))

print(f"{len(rows)} chains with both gen and gt\n")
det_mirror = [r for r in rows if r[4] == r[4] and r[4] > MIRROR_CUT]
print(f"chains the NATIVE-FREE detector calls mirrored: {len(det_mirror)}")
if det_mirror:
    caught = [r for r in det_mirror if r[5]]
    missed = [r for r in det_mirror if not r[5]]
    print(f"   of those, the GT ratio test CAUGHT   {len(caught)}")
    print(f"   of those, the GT ratio test MISSED   {len(missed)}")
    if caught and missed:
        pc = np.median([r[2] for r in caught])
        pm = np.median([r[2] for r in missed])
        print(f"\n   median proper RMSD of CAUGHT chains : {pc:8.2f} A")
        print(f"   median proper RMSD of MISSED chains : {pm:8.2f} A")
        if pm > pc:
            print(f"   ⭐ MISSED chains are {pm/pc:.1f}x WORSE -- the confound is REAL and per-chain.")
        else:
            print("   ⚠️ missed chains are NOT worse -- the confound claim is NOT supported per-chain.")
    elif not missed:
        print("   the GT test caught every detector-mirrored chain -- no confound visible here.")
    else:
        print("   the GT test caught NONE of them.")

print(f"\n{'arm':>14} {'step':>6} {'n':>3} {'det mirror':>11} {'GT flag':>8} {'med proper':>11}")
for arm, step in ROUNDS:
    sub = [r for r in rows if r[0] == arm and r[1] == step]
    if not sub:
        continue
    dm = sum(1 for r in sub if r[4] == r[4] and r[4] > MIRROR_CUT)
    gf = sum(1 for r in sub if r[5])
    mp = np.median([r[2] for r in sub])
    print(f"{arm:>14} {step:>6} {len(sub):>3} {dm:>11} {gf:>8} {mp:>11.2f}")
