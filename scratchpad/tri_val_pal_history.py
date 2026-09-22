"""Export tri_cb8synth_v5's validation P@L history for the training-status report.

Two metric families, both requested by the user 2026-09-22:
  1. one-step P@L STRATIFIED BY NOISE LEVEL -- validation_loss/contact_precision_at_L_single_step
     and its _tlow/_tmid/_thigh variants. The strata are the run's OWN logged keys; no binning is
     invented. The matching contact_precision_at_L_noisy_floor* series is exported alongside
     because the single-step number is uninterpretable without it: the floor is what copying the
     NOISED input already achieves, so P@L 0.97 at thigh against a 0.73 floor is a far smaller win
     than the raw number suggests.
  2. sampling P@L -- validation_sampling/contact_precision_at_L_{mean,median} plus the L2 / L5 /
     range-resolved variants.

⛔ USE run.history(), NOT run.scan_history(). Two scan_history attempts returned n=0 for all eight
one-step series even though they hold 1032 rows each. scan_history(keys=K) returns only rows
carrying EVERY key in K, and these validation scalars do not co-occur with trainer/global_step --
the join silently emptied the result, and a bare scan of the keys alone returned nothing either.
history() is the reliable full view. A failed query is not absence.

⛔ THE VALIDATION P@L KEYS ARE PER-BATCH, NOT PER-ROUND. Measured counts: tlow 240 + tmid 538 +
thigh 254 = 1032 = exactly the count of the unsuffixed key, i.e. every validation batch is logged
into exactly ONE stratum and the unsuffixed key is that same per-batch value, not an aggregate.
That is why run.summary reports 0.874 for it while the last history row reads 0.315 -- the summary
is just whichever batch happened to log last. Quoting the summary value as "the" one-step P@L
would be wrong. => aggregate per EPOCH (the run's own index, matching what the train side already
does with its _epoch variants) and carry n per point so thin epochs stay visible.
"""

import json
import sys

import wandb

RUN = "kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5"

STRATA = ["", "_tlow", "_tmid", "_thigh"]
ONESTEP = [f"validation_loss/contact_precision_at_L_single_step{s}" for s in STRATA]
FLOOR = [f"validation_loss/contact_precision_at_L_noisy_floor{s}" for s in STRATA]

SAMPLING = [
    "validation_sampling/contact_precision_at_L_mean",
    "validation_sampling/contact_precision_at_L_median",
    "validation_sampling/contact_precision_at_L2_mean",
    "validation_sampling/contact_precision_at_L5_mean",
    "validation_sampling/contact_long_range_precision_at_L5_mean",
    "validation_sampling/contact_medium_range_precision_at_L5_mean",
    "validation_sampling/contact_f1_mean",
    "validation_sampling/contact_recall_mean",
]

api = wandb.Api()
run = api.run(RUN)
print(f"[run] {run.name} state={run.state}", file=sys.stderr)

df = run.history(samples=200000, pandas=True)
print(f"[history] {len(df)} rows x {len(df.columns)} cols", file=sys.stderr)

df["epoch"] = df["epoch"].ffill()
df["trainer/global_step"] = df["trainer/global_step"].ffill()

out = {
    "run": run.name,
    "state": run.state,
    "global_step": float(df["trainer/global_step"].dropna().iloc[-1]),
    "epoch": float(df["epoch"].dropna().iloc[-1]),
    "onestep_by_epoch": {},
    "sampling": {},
    "strata_counts": {},
}

# ---- one-step P@L + noisy floor, aggregated per epoch -----------------------------------------
for key in ONESTEP + FLOOR:
    sub = df[[key, "epoch", "trainer/global_step"]].dropna(subset=[key])
    out["strata_counts"][key] = int(len(sub))
    if not len(sub):
        print(f"  {key:70} EMPTY", file=sys.stderr)
        out["onestep_by_epoch"][key] = []
        continue
    g = sub.groupby("epoch").agg(
        mean=(key, "mean"), n=(key, "size"), step=("trainer/global_step", "max"))
    out["onestep_by_epoch"][key] = [
        [int(e), round(float(r["mean"]), 6), int(r["n"]), int(r["step"])]
        for e, r in g.iterrows()
    ]
    print(f"  {key:70} rows={len(sub):>5} epochs={len(g):>4} last={g['mean'].iloc[-1]:.4f}",
          file=sys.stderr)

# ---- sampling P@L, already one row per validation-sampling round -------------------------------
for key in SAMPLING:
    sub = df[[key, "trainer/global_step"]].dropna(subset=[key])
    out["sampling"][key] = [[int(s), round(float(v), 6)]
                            for v, s in zip(sub[key], sub["trainer/global_step"])]
    print(f"  {key:70} n={len(sub)}", file=sys.stderr)

empty = [k for k, v in list(out["onestep_by_epoch"].items()) + list(out["sampling"].items()) if not v]
print(f"[check] {len(empty)} empty series" + (": " + ", ".join(empty) if empty else ""),
      file=sys.stderr)

print(json.dumps(out))
