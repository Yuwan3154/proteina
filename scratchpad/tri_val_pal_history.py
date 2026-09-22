"""Export tri_cb8synth_v5's validation P@L history for the training-status report.

Two metric families, both requested by the user 2026-09-22:
  1. one-step  P@L STRATIFIED BY NOISE LEVEL -- validation_loss/contact_precision_at_L_single_step
     plus its _tlow/_tmid/_thigh variants. The strata are the run's OWN logged keys; no binning is
     invented here. The matching contact_precision_at_L_noisy_floor* series is exported alongside
     because the single-step number is meaningless without it: the floor is what you get by copying
     the NOISED input, so "P@L 0.87 at thigh" only counts as learning if it beats the 0.73 floor.
  2. sampling P@L -- validation_sampling/contact_precision_at_L_{mean,median} and the L2 / L5 /
     range-resolved variants.

scan_history(keys=...) is used rather than history(): it streams only the requested columns and
returns EVERY row, where history(samples=N) subsamples and would smooth away exactly the
non-monotone excursions the report exists to show.
"""

import json
import sys

import wandb

RUN = "kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5"

STEP_KEYS = ["trainer/global_step", "global_step", "epoch"]

ONESTEP = [
    "validation_loss/contact_precision_at_L_single_step",
    "validation_loss/contact_precision_at_L_single_step_tlow",
    "validation_loss/contact_precision_at_L_single_step_tmid",
    "validation_loss/contact_precision_at_L_single_step_thigh",
    "validation_loss/contact_precision_at_L_noisy_floor",
    "validation_loss/contact_precision_at_L_noisy_floor_tlow",
    "validation_loss/contact_precision_at_L_noisy_floor_tmid",
    "validation_loss/contact_precision_at_L_noisy_floor_thigh",
]

SAMPLING = [
    "validation_sampling/contact_precision_at_L_mean",
    "validation_sampling/contact_precision_at_L_median",
    "validation_sampling/contact_precision_at_L2_mean",
    "validation_sampling/contact_precision_at_L5_mean",
    "validation_sampling/contact_long_range_precision_at_L5_mean",
    "validation_sampling/contact_medium_range_precision_at_L5_mean",
    "validation_sampling/contact_f1_mean",
    "validation_sampling/contact_recall_mean",
    "validation_sampling/contact_n_samples",
]

api = wandb.Api()
run = api.run(RUN)
print(f"[run] {run.name} id={run.id} state={run.state}", file=sys.stderr)

out = {"run": run.name, "state": run.state,
       "summary_global_step": run.summary.get("trainer/global_step"),
       "series": {}}

# scan_history(keys=K) returns ONLY rows carrying every key in K. Asking for
# trainer/global_step alongside the validation_loss/* keys returned ZERO rows for all eight of
# them -- the validation scalars are logged on rows that do not carry trainer/global_step, so the
# join killed the query. The metrics were there all along (the run summary shows them).
# => scan each family on its OWN keys, keyed by _step (always present), and build the
# _step -> global_step map from a separate scan. A failed query is not absence.
stepmap = {}
for r in run.scan_history(keys=["trainer/global_step"], page_size=10000):
    if r.get("_step") is not None and r.get("trainer/global_step") is not None:
        stepmap[int(r["_step"])] = int(r["trainer/global_step"])
print(f"[scan] step map: {len(stepmap)} rows", file=sys.stderr)


def x_for(row):
    """global_step for a row: exact if logged, else the nearest earlier mapped _step."""
    s = row.get("_step")
    if s is None:
        return None
    s = int(s)
    if s in stepmap:
        return stepmap[s]
    earlier = [k for k in stepmap if k <= s]
    return stepmap[max(earlier)] if earlier else None


for label, keys in (("onestep", ONESTEP), ("sampling", SAMPLING)):
    rows = list(run.scan_history(keys=keys, page_size=10000))
    print(f"[scan] {label}: {len(rows)} rows", file=sys.stderr)
    series = {k: [] for k in keys}
    for r in rows:
        x = x_for(r)
        if x is None:
            continue
        for k in keys:
            v = r.get(k)
            if v is not None:
                series[k].append([int(x), float(v)])
    for k in keys:
        print(f"    {k:66} n={len(series[k])}", file=sys.stderr)
    out["series"].update(series)

# ⛔ an empty series here means the QUERY is wrong, not that the metric is missing -- fail loudly
empty = [k for k, v in out["series"].items() if not v]
print(f"[check] {len(empty)} empty series" + (": " + ", ".join(empty) if empty else ""),
      file=sys.stderr)

# every series is reported, including empty ones -- an absent metric must be visible as absent
# rather than silently missing from the payload
print(json.dumps(out))
