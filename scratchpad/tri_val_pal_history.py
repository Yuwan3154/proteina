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

# ⛔ trainer/global_step is NOT exported as a run-level figure: it resets on resume, so its last
# value is a SEGMENT offset, not the run's step count. The cumulative step belongs to the lineage
# and must be read from the live job, not from this column.
out = {
    "run": run.name,
    "state": run.state,
    "epoch": float(df["epoch"].dropna().iloc[-1]),
    "onestep_by_epoch": {},
    "sampling": {},
    "strata_counts": {},
}

# ---- one-step P@L + noisy floor, aggregated per epoch -----------------------------------------
for key in ONESTEP + FLOOR:
    sub = df[[key, "epoch"]].dropna(subset=[key])
    out["strata_counts"][key] = int(len(sub))
    if not len(sub):
        print(f"  {key:70} EMPTY", file=sys.stderr)
        out["onestep_by_epoch"][key] = []
        continue
    # no trainer/global_step here either -- see the x-axis note below; epoch IS the axis
    g = sub.groupby("epoch").agg(mean=(key, "mean"), n=(key, "size"))
    out["onestep_by_epoch"][key] = [
        [int(e), round(float(r["mean"]), 6), int(r["n"])] for e, r in g.iterrows()
    ]
    print(f"  {key:70} rows={len(sub):>5} epochs={len(g):>4} last={g['mean'].iloc[-1]:.4f}",
          file=sys.stderr)

# ---- sampling P@L, already one row per validation-sampling round -------------------------------
# ⛔⛔ X-AXIS IS **epoch**, NOT trainer/global_step. Measured 2026-09-22: trainer/global_step RESETS
# on every resume and this run has eight of them -- at history _step 68696 it reads 63.0 while the
# epoch there is 113, and the column has backward jumps from the very start (99->64, 127->99, ...).
# Plotting against it would have drawn the last third of training on top of the first. epoch is
# monotone 0..127 and is already the x-axis the report's other tri charts use.
# Same family as [[feedback_align_warm_started_runs_on_the_lineage]]: a resumed run's step counter
# is not the lineage's step counter.
for key in SAMPLING:
    sub = df[[key, "epoch"]].dropna(subset=[key])
    out["sampling"][key] = [[int(e), round(float(v), 6)]
                            for v, e in zip(sub[key], sub["epoch"])]
    print(f"  {key:70} n={len(sub)}", file=sys.stderr)

empty = [k for k, v in list(out["onestep_by_epoch"].items()) + list(out["sampling"].items()) if not v]
print(f"[check] {len(empty)} empty series" + (": " + ", ".join(empty) if empty else ""),
      file=sys.stderr)

print(json.dumps(out))
