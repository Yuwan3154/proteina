"""Which contact-precision keys actually exist in tri_cb8synth_v5's HISTORY (not just its summary)?

Two scan_history attempts returned n=0 for every one-step P@L series while the same keys are
plainly present in run.summary. Rather than guess a third time, enumerate: pull the history frame
and report, for every precision-ish column, how many non-null rows it has and where they sit.

A key in summary but absent from history is a real distinction -- summary can be written directly,
and a metric logged only at validation time may land differently from a per-step metric.
"""

import sys

import wandb

RUN = "kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5"

api = wandb.Api()
run = api.run(RUN)
print(f"[run] {run.name} state={run.state}")

df = run.history(samples=200000, pandas=True)
print(f"[history] {len(df)} rows x {len(df.columns)} cols")

print("\n=== every column containing 'precision' or 'single_step' or 'noisy_floor' ===")
hits = [c for c in df.columns
        if "precision" in c or "single_step" in c or "noisy_floor" in c]
for c in sorted(hits):
    s = df[c].dropna()
    where = f"steps {int(s.index.min())}..{int(s.index.max())}" if len(s) else "-"
    last = f"{s.iloc[-1]:.5f}" if len(s) else "-"
    print(f"  {c:70} n={len(s):>6}  last={last:>9}  {where}")

print("\n=== step-like columns present in history ===")
for c in sorted(df.columns):
    if c in ("_step", "epoch", "global_step", "trainer/global_step", "_runtime"):
        s = df[c].dropna()
        print(f"  {c:70} n={len(s):>6}  last={s.iloc[-1] if len(s) else '-'}")

print("\n=== are the summary-only keys real? compare summary vs history ===")
for k in sorted(run.summary.keys()):
    if "precision_at_L" in k:
        inhist = k in df.columns
        n = int(df[k].notna().sum()) if inhist else 0
        print(f"  {k:70} summary={run.summary[k]!s:>10}  in_history={inhist} n={n}")
