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
sub = df[["trainer/global_step", M, F]].dropna()
m, f = sub[M].to_numpy(), sub[F].to_numpy()
st = sub["trainer/global_step"].to_numpy()
print(f"{len(sub)} paired points, steps {int(st.min())}..{int(st.max())}")
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
