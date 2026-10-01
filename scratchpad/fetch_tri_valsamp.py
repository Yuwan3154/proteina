# Validation-sampling P@L history of the two tris (val_fixed32_max256, mostly training chains), per global step.
import json
import math
import sys

import wandb

api = wandb.Api(timeout=180)
K = ["validation_sampling/contact_precision_at_L_mean", "validation_sampling/contact_precision_at_L_median", "validation_sampling/contact_precision_at_L5_mean",
     "validation_sampling/contact_long_range_precision_at_L5_mean", "trainer/global_step", "epoch"]
out = {}
for ent in ("kryst3154-massachusetts-institute-of-technology", "DP_CO_AFdiffusion"):
    if "protein_transformer_big_runs" not in {p.name for p in api.projects(ent)}:
        print(f"[entity] {ent}: project absent"); continue
    for name in ("tri_cb8synth_v5", "tri_confindsynth_ft"):
            for r in api.runs(f"{ent}/protein_transformer_big_runs", filters={"display_name": name}):
                rows = [x for x in r.scan_history(keys=K[:1] + ["trainer/global_step"], page_size=2000)]
                full = [x for x in r.scan_history(keys=K, page_size=2000)] or rows
                print(f"[run] {ent} {name} {r.id} {r.state} rows {len(full)}", flush=True)
                out.setdefault(name, []).extend({k: x.get(k) for k in K} for x in full)
json.dump(out, open(sys.argv[1], "w"))
for n, v in out.items():
    v = sorted((x for x in v if x["trainer/global_step"] is not None), key=lambda x: x["trainer/global_step"])
    print(n, len(v), "last:", v[-3:] if v else None)
