import json
import os

import pandas as pd
import wandb

ENTITY = "kryst3154-massachusetts-institute-of-technology"
PROJECT = "protein_transformer_big_runs"
OUT_DIR = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
RUN_IDS = os.environ["RUN_IDS"].split(",")

# Same key set as fetch_localattn_curves.py -- do not invent new metric names.
KEYS = [
    "train/loss_epoch",
    "validation_loss/loss_epoch",
    "train/contact_map_loss_epoch",
    "validation_loss/contact_map_loss_epoch",
    "train/contact_precision_at_L_single_step_epoch",
    "validation_loss/contact_precision_at_L_single_step",
    "validation_loss/contact_precision_at_L_noisy_floor",
    "validation_sampling/contact_precision_at_L_mean",
    "validation_sampling/contact_precision_at_L_median",
]
PRECISION_HIGHER_IS_BETTER = "precision"

os.makedirs(OUT_DIR, exist_ok=True)
api = wandb.Api(timeout=180)

for run_id in RUN_IDS:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    df = pd.DataFrame(list(run.scan_history(keys=None, page_size=2000)))
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(f"{OUT_DIR}/{run_id}_full_history.csv", index=False)
    print(f"\n===== {run_id} =====")
    print(f"state={run.state}  lastHistoryStep={run.lastHistoryStep}  history_rows={len(df)}")
    if "epoch" not in df.columns:
        print("  no 'epoch' column -- nothing epoch-indexed to report")
        continue
    print(f"  max epoch logged: {df['epoch'].max()}")
    report = {}
    for key in KEYS:
        if key not in df.columns:
            print(f"  {key:58s} ABSENT")
            continue
        sub = df[["epoch", key]].dropna().sort_values("epoch")
        if len(sub) < 2:
            print(f"  {key:58s} only {len(sub)} point(s)")
            continue
        y = sub[key].values
        x = sub["epoch"].values
        higher = PRECISION_HIGHER_IS_BETTER in key
        best_i = int(y.argmax() if higher else y.argmin())
        # last 20% of logged points, to separate trend from single-point noise
        tail = y[max(0, len(y) - max(2, len(y) // 5)) :]
        report[key] = {
            "n": len(y), "last_epoch": float(x[-1]), "final": float(y[-1]),
            "best": float(y[best_i]), "best_epoch": float(x[best_i]),
            "tail_mean": float(tail.mean()), "tail_std": float(tail.std()),
        }
        print(f"  {key:58s} final={y[-1]:.4f}  best={y[best_i]:.4f}@ep{x[best_i]:.0f}  "
              f"tail_mean={tail.mean():.4f}  n={len(y)}")
    with open(f"{OUT_DIR}/{run_id}_convergence.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  wrote {OUT_DIR}/{run_id}_convergence.json")
