"""Is tri's precision@L plateauing, or still climbing under round-to-round noise?

Three ticks ago I flagged "precision@L FLAT ~0.30 for 4 rounds" and said per-round gains had shrunk
from ~0.05 to ~0.008. That was eyeballed. With 15 rounds it is testable: fit a trend to the LATE
rounds and ask whether its slope is distinguishable from zero given the residual scatter.

⛔ The rounds are NOT evenly spaced -- the resumed segment logged one only 351 steps after the
previous, where the others sit ~1000 apart. Regress on STEP, not on round index, or that point gets
~3x the leverage it deserves.
"""

import numpy as np
import wandb

api = wandb.Api()
r = api.run("kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5")
df = r.history(samples=200000, pandas=True)
if "global_step" in df.columns and "_step" in df.columns:
    df = df.sort_values("_step")
    df["_true"] = df["global_step"].ffill()

M = "validation_sampling/contact_precision_at_L_mean"
sub = df[["_true", M]].dropna()
s = sub["_true"].to_numpy()
v = sub[M].to_numpy()
print(f"{len(sub)} rounds, steps {int(s.min())}..{int(s.max())}")
print("  " + "  ".join(f"{int(a)}:{b:.3f}" for a, b in zip(s, v)))

for label, lo in (("all rounds", 0), ("late (>= step 6000)", 6000), ("last 5 rounds", s[-5])):
    m = s >= lo
    if m.sum() < 3:
        continue
    x, y = s[m], v[m]
    # least squares slope per 1000 steps, with its standard error
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym + slope * (x - xm))
    s_err = np.sqrt((resid ** 2).sum() / max(n - 2, 1))
    se_slope = s_err / np.sqrt(sxx)
    t = slope / se_slope if se_slope else float("nan")
    print(f"\n{label}: n={n}")
    print(f"   slope       {1000*slope:+.4f} per 1000 steps  (SE {1000*se_slope:.4f}, t={t:+.2f})")
    print(f"   residual sd {s_err:.4f}  <- the round-to-round scatter my 'flat' claim was reading")
    print("   " + ("⭐ slope distinguishable from zero (|t| > 2): STILL CLIMBING"
                   if abs(t) > 2 else
                   "→ slope NOT distinguishable from zero at this n: cannot call climb or plateau"))
