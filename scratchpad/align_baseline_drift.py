"""Is the alignment head's lead over its baseline real, or is the BASELINE drifting down?

align_precision_at_q is now 0.624 against a positional baseline of 0.448. But the code comment
records that baseline MEASURED OFFLINE at 0.555 (job 22507049) on a pre-v4 index. A position-only
baseline depends on the DATA, not the model, so it should be roughly flat within a run -- if it is
falling, part of the head's apparent lead is the bar moving down rather than the head moving up.

Plot both trajectories over the run.
"""

import numpy as np
import wandb

api = wandb.Api()
r = api.run("kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5")
df = r.history(samples=100000, pandas=True)
if "global_step" in df.columns and "_step" in df.columns:
    df = df.sort_values("_step")
    df["_true"] = df["global_step"].ffill()

H = "validation_loss/align_precision_at_q_epoch"
B = "validation_loss/align_precision_at_q_pos_baseline_epoch"
sub = df[["_true", H, B]].dropna()
s = sub["_true"].to_numpy()
h = sub[H].to_numpy()
b = sub[B].to_numpy()
print(f"{len(sub)} paired epoch points, steps {int(s.min())}..{int(s.max())}\n")
print(f"{'band':>18} {'head':>7} {'baseline':>9} {'lead':>7}")
edges = np.linspace(s.min(), s.max(), 6)
for lo, hi in zip(edges, edges[1:]):
    m = (s >= lo) & (s < hi)
    if m.sum():
        print(f"{f'{int(lo)}-{int(hi)}':>18} {np.median(h[m]):7.4f} {np.median(b[m]):9.4f} "
              f"{np.median(h[m]) - np.median(b[m]):+7.4f}")

print(f"\nbaseline first-quarter median {np.median(b[:len(b)//4]):.4f} -> "
      f"last-quarter {np.median(b[-len(b)//4:]):.4f}")
print(f"head     first-quarter median {np.median(h[:len(h)//4]):.4f} -> "
      f"last-quarter {np.median(h[-len(h)//4:]):.4f}")
drift = np.median(b[-len(b)//4:]) - np.median(b[:len(b)//4])
gain = np.median(h[-len(h)//4:]) - np.median(h[:len(h)//4])
print(f"\nbaseline drift {drift:+.4f} | head gain {gain:+.4f}")
if abs(drift) < 0.02:
    print("⭐ the baseline is essentially FLAT -- the head's lead is the head improving, not the bar falling.")
elif drift < 0:
    print(f"⚠️ the baseline FELL by {abs(drift):.4f}; that much of the apparent lead is the bar moving down.")
else:
    print(f"⭐ the baseline ROSE by {drift:.4f}; the head's lead is achieved against a RISING bar.")
print("\n⛔ The offline 0.555 (job 22507049) was measured on a PRE-v4 index -- do not mix it with")
print("   today's numbers; the template set changed. Only within-run comparisons are valid.")
