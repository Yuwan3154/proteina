"""Did the chain resume restore the MODEL STATE, or only the step counter?

The log says "Restoring states from the checkpoint path at .../last.ckpt", and the wandb series is
monotone -- but neither shows that the optimizer, EMA and weights actually came back. If any of those
were lost, training would effectively restart from a worse point and the metrics would JUMP at the
segment boundary while the step counter marched on regardless.

Test: compare metric values immediately before the timeout (step ~14,342) and immediately after the
resume (step ~14,352). Continuous values mean the state genuinely came back; a jump means it did not.

⛔ The metrics are noisy per round, so compare a WINDOW either side, not single points, and use a
metric whose round-to-round scatter is known to be modest.
"""

import numpy as np
import wandb

BOUNDARY = 14347          # between the last pre-timeout step and the resumed step
HALFWIDTH = 900           # steps either side to average over

api = wandb.Api()
r = api.run("kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5")
df = r.history(samples=200000, pandas=True)
if "global_step" in df.columns and "_step" in df.columns:
    df = df.sort_values("_step")
    df["_true"] = df["global_step"].ffill()

METRICS = [
    "validation_loss/contact_map_loss_epoch",
    "validation_loss/align_precision_at_q_epoch",
    "validation_loss/mlm_acc_epoch",
    "validation_loss/contact_precision_at_L_single_step",
]

print(f"boundary at step ~{BOUNDARY}, comparing +/-{HALFWIDTH} steps\n")
print(f"{'metric':>52} {'before':>9} {'after':>9} {'delta':>9}")
worst = 0.0
for m in METRICS:
    if m not in df.columns:
        print(f"{m:>52} {'ABSENT':>9}")
        continue
    sub = df[["_true", m]].dropna()
    s = sub["_true"].to_numpy()
    v = sub[m].to_numpy()
    pre = v[(s >= BOUNDARY - HALFWIDTH) & (s < BOUNDARY)]
    post = v[(s > BOUNDARY) & (s <= BOUNDARY + HALFWIDTH)]
    if not len(pre) or not len(post):
        print(f"{m:>52} {'n/a':>9} {'n/a':>9}  (pre={len(pre)} post={len(post)})")
        continue
    a, b = np.median(pre), np.median(post)
    rel = abs(b - a) / max(abs(a), 1e-9)
    worst = max(worst, rel)
    print(f"{m:>52} {a:9.4f} {b:9.4f} {b-a:+9.4f}   ({100*rel:.0f}% rel, n={len(pre)}/{len(post)})")

print(f"\nlargest relative shift across the boundary: {100*worst:.0f}%")
if worst < 0.30:
    print("⭐ No jump beyond ordinary round-to-round scatter -- model/optimizer/EMA state came back.")
else:
    print("⛔ A metric shifted sharply at the boundary -- the restore may have dropped state.")
