"""Why do the 54 validation_sampling rows map to global_step 1279..6463 when the run is at 51,647?

The sampling rows sit at history _step 1504..76208, i.e. spread across the whole run, yet
ffill(trainer/global_step) put them all below 6,500. Either trainer/global_step is non-monotone in
the returned frame (resume resets), or it is NaN across the sampling rows and ffill is carrying a
stale value. Print the raw columns for those rows and settle it -- a wrong x-axis would make the
convergence panel actively misleading.
"""

import wandb

RUN = "kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5"
KEY = "validation_sampling/contact_precision_at_L_mean"

api = wandb.Api()
run = api.run(RUN)
df = run.history(samples=200000, pandas=True)
print(f"[history] {len(df)} rows")

gs = df["trainer/global_step"]
print(f"[trainer/global_step] non-null={gs.notna().sum()} monotonic={gs.dropna().is_monotonic_increasing}")
print(f"[trainer/global_step] min={gs.min()} max={gs.max()}")
d = gs.dropna()
drops = (d.diff() < 0).sum()
print(f"[trainer/global_step] backward jumps: {drops}")
if drops:
    idx = d.index[(d.diff() < 0)][:10]
    for i in idx:
        print(f"    backward at _step={i}: {d.loc[:i].iloc[-2]} -> {d.loc[i]}")

sub = df[df[KEY].notna()]
print(f"\n[sampling rows] n={len(sub)}")
print(f"{'_step':>8} {'raw gstep':>11} {'ffill gstep':>12} {'epoch':>8} {'value':>8}")
gsf = gs.ffill()
ef = df["epoch"].ffill()
for i in list(sub.index)[:6] + list(sub.index)[-6:]:
    raw = df.at[i, "trainer/global_step"]
    print(f"{i:>8} {str(raw):>11} {str(gsf.at[i]):>12} {str(ef.at[i]):>8} {df.at[i, KEY]:>8.4f}")
