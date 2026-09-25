"""Fetch every training-status-report series in one go (user 2026-09-25: "update the training metrics report on
everything"). Generalises the three .tmp fetchers (fetch_c2c.py, extract_tri.py, tri_val_pal_history.py) to all runs:

  c2c  (segments concatenated on trainer/global_step, deduped; both wandb entities):  c2c_cb8_tbeta, c2c_confind_tbeta
       -> c2c_steps_<name>.json
  tri  (one run id per model, epoch is the monotone axis; trainer/global_step resets on resume):
       tri_cb8synth_v5, tri_confindsynth_ft -> tri_epochs_<run>.json + tri_val_pal_<run>.json

Uses run.history(), not scan_history (the latter silently drops validation keys that do not co-occur with the
step key -- see tri_val_pal_history.py). Writes to OUT (default: $S/.tmp/curves/report_<date>/).

Usage: python scratchpad/fetch_report_data.py OUT_DIR
"""

import json
import math
import os
import sys

import wandb

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
ENTITIES = ("DP_CO_AFdiffusion", "kryst3154-massachusetts-institute-of-technology")
TRI_ENTITY = "kryst3154-massachusetts-institute-of-technology"
C2C_RUNS = ("c2c_cb8_tbeta", "c2c_confind_tbeta")
TRI_RUNS = ("tri_cb8synth_v5", "tri_confindsynth_ft")
C2C_VAL = ["val/loss", "val/diffusion", "val/distogram", "val/rmsd", "val/dist_mae_sampled", "val/rmsd_proper",
           "val/rmsd_reflected", "val/rmsd_refl_gap", "val/is_mirrored", "val/helix_pos_frac"]
C2C_TRAIN = ["train/loss", "train/diffusion", "train/distogram", "train/rmsd"]
TRI_KEYS = {"train_contact": "train/contact_map_loss_epoch", "val_contact": "validation_loss/contact_map_loss_epoch",
            "train_align": "train/align_loss_epoch", "val_align": "validation_loss/align_loss_epoch",
            "train_p": "train/contact_precision_at_L_single_step_epoch",
            "val_p": "validation_loss/contact_precision_at_L_single_step", "lr": "lr-Adam", "step": "global_step"}
STRATA = ["", "_tlow", "_tmid", "_thigh"]
ONESTEP = [f"validation_loss/contact_precision_at_L_single_step{s}" for s in STRATA]
FLOOR = [f"validation_loss/contact_precision_at_L_noisy_floor{s}" for s in STRATA]
SAMPLING = ["validation_sampling/contact_precision_at_L_mean", "validation_sampling/contact_precision_at_L_median",
            "validation_sampling/contact_precision_at_L2_mean", "validation_sampling/contact_precision_at_L5_mean",
            "validation_sampling/contact_long_range_precision_at_L5_mean",
            "validation_sampling/contact_medium_range_precision_at_L5_mean", "validation_sampling/contact_f1_mean",
            "validation_sampling/contact_recall_mean"]

api = wandb.Api(timeout=180)


def num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# ---- c2c ------------------------------------------------------------------------------------------------------
for name in C2C_RUNS:
    runs = []
    for ent in ENTITIES:
        has = "contact2coord" in {p.name for p in api.projects(ent)}
        got = [x for x in api.runs(ent + "/contact2coord") if x.name == name] if has else []
        print(f"[c2c {name}] {ent}: project {'present' if has else 'ABSENT'}, {len(got)} segments")
        runs += got
    by_step = {}
    for x in runs:
        for rec in x.scan_history(keys=None, page_size=2000):
            s = num(rec.get("trainer/global_step"))
            if s is None:
                continue
            d = by_step.setdefault(int(s), {"step": int(s)})
            e = num(rec.get("epoch"))
            if e is not None:
                d["epoch"] = int(e)
            for k in C2C_VAL + C2C_TRAIN:
                v = num(rec.get(k))
                if v is not None:
                    d[k] = v
    series = [by_step[s] for s in sorted(by_step)]
    rng = f"{series[0]['step']} -> {series[-1]['step']}" if series else "EMPTY"
    print(f"[c2c {name}] {len(runs)} segments, {len(series)} distinct steps, range {rng}; "
          f"val/rmsd n={sum(1 for d in series if 'val/rmsd' in d)}")
    json.dump(series, open(os.path.join(OUT, f"c2c_steps_{name}.json"), "w"))

# ---- tri ------------------------------------------------------------------------------------------------------
for rid in TRI_RUNS:
    run = api.run(f"{TRI_ENTITY}/protein_transformer_big_runs/{rid}")
    df = run.history(samples=200000, pandas=True)
    df["epoch"] = df["epoch"].ffill()
    print(f"[tri {rid}] state={run.state} history {len(df)} rows x {len(df.columns)} cols")
    rows = {}
    for _, rec in df.iterrows():
        e = num(rec.get("epoch"))
        if e is None:
            continue
        cur = rows.setdefault(int(e), {})
        for out, col in TRI_KEYS.items():
            v = num(rec.get(col)) if col in df.columns else None
            if v is not None:
                cur[out] = v  # last non-null value in the epoch, as extract_tri.py
    series = []
    for e in sorted(rows):
        rows[e]["epoch"] = e
        series.append(rows[e])
    print(f"[tri {rid}] epochs {len(series)}; " + ", ".join(f"{k} n={sum(1 for d in series if k in d)}" for k in TRI_KEYS))
    json.dump(series, open(os.path.join(OUT, f"tri_epochs_{rid}.json"), "w"))
    pal = {"run": run.name, "state": run.state, "epoch": float(df["epoch"].dropna().iloc[-1]) if len(df) else None,
           "onestep_by_epoch": {}, "sampling": {}, "strata_counts": {}}
    for key in ONESTEP + FLOOR:
        sub = df[[key, "epoch"]].dropna(subset=[key]) if key in df.columns else df.iloc[0:0]
        pal["strata_counts"][key] = int(len(sub))
        g = sub.groupby("epoch").agg(mean=(key, "mean"), n=(key, "size")) if len(sub) else None
        pal["onestep_by_epoch"][key] = [] if g is None else [[int(e), round(float(r["mean"]), 6), int(r["n"])]
                                                            for e, r in g.iterrows()]
    for key in SAMPLING:
        sub = df[[key, "epoch"]].dropna(subset=[key]) if key in df.columns else df.iloc[0:0]
        pal["sampling"][key] = [[int(e), round(float(v), 6)] for v, e in zip(sub[key], sub["epoch"])]
    empty = [k for k, v in list(pal["onestep_by_epoch"].items()) + list(pal["sampling"].items()) if not v]
    print(f"[tri {rid}] pal: {len(empty)} empty series")
    json.dump(pal, open(os.path.join(OUT, f"tri_val_pal_{rid}.json"), "w"))
print("FETCH_DONE")
