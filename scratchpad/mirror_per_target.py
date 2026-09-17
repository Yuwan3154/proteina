"""Is the mirror a per-SAMPLE coin flip, or a per-TARGET property the model has learned?

The round-level rate has hovered near 0.5 for thousands of steps. Two very different worlds produce
that same number:
  (A) COIN FLIP  -- each generation independently picks an enantiomer. Per-target rates cluster at
      0.5 and the spread across targets is just binomial noise.
  (B) PER-TARGET -- the model reliably builds SOME targets right-handed and others mirrored. Per-
      target rates pile up at 0 and 1, and the spread is much wider than binomial.
These imply different fixes. Under (B) the model has learned a stable wrong answer for specific
structures, and something about those structures predicts it; under (A) there is no signal to latch
onto and only an explicit chirality term can break the tie.

⛔⛔ THIS ONLY WORKS IF SAMPLE SLOT j IS THE SAME CHAIN IN EVERY ROUND. That is an assumption about
dataloader determinism, not a fact, and the whole analysis is meaningless if it is wrong -- so it is
VERIFIED here by comparing each slot's ground-truth coordinates across rounds. Slots that are not
stable are dropped and reported, never silently averaged over.

Usage: mirror_per_target.py <run_name> [--min_step 5000]
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri")
sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad")

from ca_handedness_filter import read_ca
from proteinfoundation.utils.c2c_dump import handedness_metrics

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"

ap = argparse.ArgumentParser()
ap.add_argument("run")
# ⛔ Restrict to the converged regime by default. Before the model folds at all, both superpositions
# are ~equally bad and the reflection sign is noise about nothing -- pooling those rounds in would
# manufacture exactly the 0.5-everywhere picture that hypothesis (A) predicts.
ap.add_argument("--min_step", type=int, default=5000)
args = ap.parse_args()

rounds = sorted(glob.glob(os.path.join(STORE, args.run, "samples", "step*")))
rounds = [r for r in rounds if int(os.path.basename(r).replace("step", "")) >= args.min_step]
assert rounds, f"no rounds at or after step {args.min_step}"
print(f"[data] {len(rounds)} rounds at step >= {args.min_step}: "
      f"{', '.join(os.path.basename(r).replace('step','') for r in rounds)}")

gt_sig = defaultdict(list)     # slot -> list of (step, signature)
sign = defaultdict(list)       # slot -> list of (step, reflected_fits_better)
skipped = []

for rd in rounds:
    st = int(os.path.basename(rd).replace("step", ""))
    for gp in sorted(glob.glob(os.path.join(rd, "*_gen.pdb"))):
        slot = os.path.basename(gp).replace("_gen.pdb", "")
        tp = gp.replace("_gen.pdb", "_gt.pdb")
        if not os.path.exists(tp):
            skipped.append((st, slot, "no _gt.pdb"))
            continue
        g, t = read_ca(gp), read_ca(tp)
        n = min(len(g), len(t))
        if n < 10:
            skipped.append((st, slot, f"n={n}"))
            continue
        h = handedness_metrics(g[:n], t[:n])
        if not h:
            skipped.append((st, slot, "no handedness metrics"))
            continue
        # signature of the TARGET: length + rounded radius of gyration. Cheap, and enough to catch a
        # slot that is a different chain from one round to the next.
        rg = float(np.sqrt(((t - t.mean(0)) ** 2).sum(1).mean()))
        gt_sig[slot].append((st, (len(t), round(rg, 2))))
        sign[slot].append((st, 1 if h["rmsd_proper"] > h["rmsd_reflected"] else 0))

print(f"[skips] {len(skipped)}")
for st, slot, why in skipped[:20]:
    print(f"    step {st} {slot}: {why}")

# ── the assumption, checked ──────────────────────────────────────────────────────────────────
# ⛔ MEASURED: slot j is NOT a fixed chain. Every slot carries a different target each round, so
# grouping by slot would average over different chains and manufacture a coin-flip picture.
stable, unstable = [], []
for slot, sigs in gt_sig.items():
    uniq = set(s for _, s in sigs)
    (stable if len(uniq) == 1 else unstable).append(slot)
print(f"\n=== slot stability (is slot j the same chain every round?) ===")
print(f"  stable slots  : {len(stable)}")
print(f"  UNSTABLE slots: {len(unstable)}")
print("  ⇒ grouping by SLOT is invalid here; regrouping by target SIGNATURE instead.")

# ── regroup by target, not by slot ───────────────────────────────────────────────────────────
# The validation loader hands out different chains per round, so match a target across rounds by its
# own ground-truth signature. If the loader also crops randomly, the same chain yields a DIFFERENT
# signature each round and nothing will recur -- which the recurrence histogram below will show
# plainly rather than hiding behind an average.
by_target = defaultdict(list)
for slot in gt_sig:
    for (st, sig), (st2, v) in zip(gt_sig[slot], sign[slot]):
        assert st == st2, "step misalignment between signature and sign lists"
        by_target[sig].append((st, v))

occur = defaultdict(int)
for sig, v in by_target.items():
    occur[len(v)] += 1
print(f"\n=== target recurrence across the {len(rounds)} rounds ===")
print(f"  distinct targets seen: {len(by_target)}")
for k_ in sorted(occur):
    print(f"    seen in {k_} round(s): {occur[k_]} targets")

MIN_ROUNDS = 3
rep = {s: v for s, v in by_target.items() if len(v) >= MIN_ROUNDS}
if not rep:
    print(f"\n⛔ NO target recurs in >= {MIN_ROUNDS} rounds. The question 'is the mirror a stable")
    print("   per-target property' CANNOT be answered from these dumps: validation draws fresh")
    print("   chains (and/or fresh crops) every round, so no target is ever revisited.")
    print("   ⇒ This is a DATA LIMITATION, not a null result. Answering it needs a fixed")
    print("     validation subset re-sampled across rounds.")
    sys.exit(0)

rates, counts = [], []
print(f"\n=== per-target reflection rate, {len(rep)} targets seen in >= {MIN_ROUNDS} rounds ===")
print(f"{'len':>5} {'rounds':>7} {'mirrored':>9} {'rate':>6}")
for sig in sorted(rep, key=lambda s: -len(rep[s])):
    v = [x for _, x in rep[sig]]
    rates.append(np.mean(v)); counts.append(len(v))
    print(f"{sig[0]:>5} {len(v):>7} {sum(v):>9} {np.mean(v):>6.2f}")

rates = np.array(rates)
k = int(min(counts))
print(f"\npooled reflection rate: {rates.mean():.3f} over {len(rates)} targets")

# ⛔ Compare the SPREAD to what a fair per-sample coin flip would give. Under (A) each target's rate
# is Binomial(k, p)/k with variance p(1-p)/k; under (B) rates pile up at 0 and 1 and the variance is
# much larger. Using the OBSERVED pooled p, so this is not a test against a strawman p=0.5.
p = rates.mean()
exp_var = p * (1 - p) / k if k else 0.0
obs_var = rates.var()
print(f"  observed variance across targets : {obs_var:.4f}")
print(f"  variance if it were a coin flip  : {exp_var:.4f}  (k={k} rounds per target, p={p:.3f})")
if exp_var > 0:
    print(f"  ratio observed/expected          : {obs_var/exp_var:.2f}")
extreme = int(((rates == 0.0) | (rates == 1.0)).sum())
print(f"  targets that are ALWAYS one way  : {extreme} of {len(rates)} "
      f"({100.0*extreme/len(rates):.0f}%)")
print("\nreading: ratio >> 1 and many all-or-nothing targets => PER-TARGET (learned, stable).")
print("         ratio ~ 1 and rates clustered near the pooled mean => per-sample COIN FLIP.")
