"""Redo the top-k check against the metric the callback ACTUALLY monitors.

train.py:706 monitors `validation_sampling/contact_precision_at_L_median` -- the GENERATIVE metric,
logged only at the every-1000-step evaluation rounds. My first check compared against
`validation_loss/contact_precision_at_L_single_step`, the noisy per-round DENOISING metric, and so
"found" that selection looked arbitrary. The check was wrong, not the callback.

Correct test: the retained best-checkpoint steps should be exactly the argmax-5 of the monitored
metric.
"""

import re
import subprocess

import numpy as np
import wandb

LS = subprocess.run(
    ["ls", "-1", "/orcd/scratch/orcd/011/chenxiou/proteina_tri/store/tri_cb8synth_v5/checkpoints"],
    capture_output=True, text=True)
best = sorted({int(m.group(1)) for n in LS.stdout.split("\n")
               if "best_contact" in n for m in [re.search(r"step=0*(\d+)", n)] if m})

api = wandb.Api()
r = api.run("kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5")
df = r.history(samples=100000, pandas=True)
M = "validation_sampling/contact_precision_at_L_median"
if M not in df.columns:
    raise SystemExit(f"{M} ABSENT")
if "global_step" in df.columns and "_step" in df.columns:
    df = df.sort_values("_step")
    df["_true"] = df["global_step"].ffill()
sub = df[["_true", M]].dropna()
s = sub["_true"].to_numpy().astype(int)
v = sub[M].to_numpy()

print(f"monitored metric: {M}")
print(f"{len(sub)} evaluation rounds\n")
print(f"{'step':>7} {'median p@L':>11}  retained?")
for st, val in zip(s, v):
    mark = "  <- RETAINED" if any(abs(st - b) <= 60 for b in best) else ""
    print(f"{st:>7} {val:11.4f}{mark}")

order = np.argsort(-v)
top5 = sorted(int(s[i]) for i in order[:5])
print(f"\nargmax-5 steps of the monitored metric: {top5}")
print(f"retained best-checkpoint steps:         {best}")
match = all(any(abs(t - b) <= 60 for b in best) for t in top5) and len(best) == len(top5)
print("⭐ EXACT MATCH — top-k selection is tracking the right metric." if match
      else "⚠️ MISMATCH — investigate the callback.")
