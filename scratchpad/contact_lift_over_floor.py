"""Is the single-step contact metric measuring LEARNING, or measuring COPYING?

contact_precision_at_L_single_step last reads 1.0000 while its last-quarter median was 0.460, so the
last point alone proves nothing. What matters is the metric against its OWN noisy floor: the floor is
the precision of the NOISED INPUT itself, so (metric - floor) is all the model actually contributes.
A high metric over a high floor means the answer was already in the input -- the same copy-vs-generate
distinction that drives the c2c chirality mechanism.
"""

import numpy as np
import wandb

api = wandb.Api()
r = api.run("kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5")
df = r.history(samples=100000, pandas=True)

M = "validation_loss/contact_precision_at_L_single_step"
F = "validation_loss/contact_precision_at_L_noisy_floor"
# ⛔ `global_step` is the TRUE optimizer step (it matches the diag log and the on-box watcher).
# `trainer/global_step` runs ~1.28x higher on this trainer and reading it overstates every step
# label by ~28% -- the generative rounds sat at real steps 1000/2001/3002/4003/5004 while
# trainer/global_step called them 1279/2559/3839/5119/6399, one of which was AHEAD of the run.
# `global_step` is logged on far fewer rows than the metrics, so an inner join is EMPTY. Sort by
# wandb's own row counter and forward-fill, giving each metric row the most recent true step.
if "global_step" in df.columns and "_step" in df.columns:
    df = df.sort_values("_step")
    df["_true_step"] = df["global_step"].ffill()
    STEP = "_true_step"
else:
    STEP = "trainer/global_step"
sub = df[[STEP, M, F]].dropna()
m, f = sub[M].to_numpy(), sub[F].to_numpy()
st = sub[STEP].to_numpy()
print(f"{len(sub)} paired points, steps {int(st.min())}..{int(st.max())} (axis: {STEP})")
for lo, hi, name in [(0, len(m)//4, "first quarter"), (3*len(m)//4, len(m), "last quarter")]:
    mm, ff = m[lo:hi], f[lo:hi]
    print(f"  {name:14} metric median {np.median(mm):.4f}  floor median {np.median(ff):.4f}  "
          f"LIFT {np.median(mm) - np.median(ff):+.4f}")
print(f"\n  last 8 paired values (metric / floor / lift):")
for a, b, c in zip(st[-8:], m[-8:], f[-8:]):
    print(f"      step {int(a):>6}: {b:.4f} / {c:.4f} / {b - c:+.4f}")

# Derive the contact DENSITY, which is a random predictor's precision@L, from a round where the
# model predicted essentially nothing: recall ~ 0 means accuracy ~ the true-negative fraction.
acc = df["validation_sampling/contact_accuracy_mean"].dropna()
rec = df["validation_sampling/contact_recall_mean"].dropna()
if len(acc) and len(rec):
    print(f"\n  generative rounds (accuracy, recall):")
    for a, b in zip(acc.to_numpy(), rec.to_numpy()):
        print(f"      accuracy {a:.4f}  recall {b:.4f}"
              + ("   <- recall~0, so accuracy ~= 1 - density" if b < 0.005 else ""))
    a0 = acc.to_numpy()[0]
    print(f"  => implied contact density ~ {1 - a0:.4f}, which is a RANDOM predictor's precision@L")
