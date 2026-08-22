import os

import pandas as pd

CURVES = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
RUN_IDS = os.environ["RUN_IDS"].split(",")

for run_id in RUN_IDS:
    df = pd.read_csv(f"{CURVES}/{run_id}_full_history.csv", low_memory=False)
    cols = [c for c in ("epoch", "trainer/global_step", "_runtime") if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["epoch"])
    print(f"\n===== {run_id} =====")
    print(f"rows with numeric epoch: {len(sub)}   epoch range: {sub['epoch'].min()} .. {sub['epoch'].max()}")
    # first row of each epoch: what global_step and wall-clock did that epoch begin at
    first = sub.sort_values("_runtime").groupby("epoch").first().reset_index()
    print(first.to_string(index=False))
    if len(first) > 2:
        span_h = (first["_runtime"].iloc[-1] - first["_runtime"].iloc[0]) / 3600.0
        n_ep = first["epoch"].iloc[-1] - first["epoch"].iloc[0]
        print(f"\n  {n_ep:.0f} epochs over {span_h:.2f} h  ->  {span_h / max(n_ep, 1):.3f} h/epoch")
        d_step = first["trainer/global_step"].iloc[-1] - first["trainer/global_step"].iloc[0]
        print(f"  global_step advanced {d_step:.0f} over those epochs -> {d_step / max(n_ep, 1):.1f} steps/epoch")
