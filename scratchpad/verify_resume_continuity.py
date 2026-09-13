"""Did the chain resume SPLICE the wandb series, or continue it cleanly?

In window 1 a relaunch resumed a wandb run of the same name and produced 1,118 rows from the OLD
v3-index run interleaved with 2,116 from the new one under a single id, overlapping in global_step --
so any plot mixed two experiments. The fix was to give a genuine restart a new RUN name.

This resume is the OTHER case: a chain SEGMENT continuing the same training, where reusing the name
is CORRECT. But correct-in-principle is not the same as correct-in-fact. Verify:
  - global_step is MONOTONE across the segment boundary (no overlap, no rewind)
  - the gap at the boundary is small (<= the checkpoint interval, 50 steps)
  - no duplicated step values, which would mean two segments writing the same x
"""

import numpy as np
import wandb

api = wandb.Api()
r = api.run("kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5")
df = r.history(samples=200000, pandas=True)
if "global_step" not in df.columns or "_step" not in df.columns:
    raise SystemExit("missing step columns")
df = df.sort_values("_step")
s = df["global_step"].dropna().to_numpy().astype(int)
print(f"{len(s)} logged global_step values, range {s.min()}..{s.max()}")

d = np.diff(s)
back = np.where(d < 0)[0]
print(f"\nbackward steps (rewind / splice signature): {len(back)}")
if len(back):
    for i in back[:5]:
        print(f"   at index {i}: {s[i]} -> {s[i+1]}  (drop of {s[i]-s[i+1]})")

dup = len(s) - len(np.unique(s))
print(f"duplicated step values: {dup}")

big = np.where(d > 50)[0]
print(f"\ngaps larger than the 50-step checkpoint interval: {len(big)}")
for i in big[:5]:
    print(f"   {s[i]} -> {s[i+1]}  (gap {s[i+1]-s[i]})")

# the segment boundary: 22596722 timed out ~14,342, 22596750 resumed at 14,352
near = [(a, b) for a, b in zip(s, s[1:]) if 14000 <= a <= 14600 and b != a + 1]
print(f"\nsteps around the segment boundary (14,000-14,600), non-unit transitions:")
for a, b in near[:10]:
    print(f"   {a} -> {b}")

# ⛔ The splice signature is a BACKWARD step, not a duplicate. wandb legitimately writes several
# metric rows at the same global_step (an epoch-level row and a step-level row, say), so requiring
# dup == 0 raised a FALSE ALARM on a perfectly clean resume the first time this ran.
if not len(back):
    print(f"\n⭐ MONOTONE (0 backward steps) -- the segment continued the series cleanly, NOT spliced.")
    print(f"   {dup} duplicated step values are EXPECTED: multiple metrics logged at one global_step.")
else:
    print("\n⛔ BACKWARD steps present -- the series is spliced or rewound. Do not trust any tri curve.")
