"""Merge the 4 WS1 shards and report slim-vs-stock AF2Rank calibration:
mean per-target Spearman(score, tm_ref_template) over all targets, for composite/ptm/plddt."""
import glob
import os

import pandas as pd
from scipy.stats import spearmanr

OUT = "/home/jupyter-chenxi/runs/ws1_af2rank"


def per_target_spearman(df, col, truth="tm_ref_template"):
    rhos = []
    for tid, g in df.groupby("target"):
        ok = g[[col, truth]].dropna()
        if len(ok) >= 3 and ok[col].nunique() > 1 and ok[truth].nunique() > 1:
            rhos.append(spearmanr(ok[col], ok[truth]).correlation)
    return rhos


rows = []
for mode in ("slim", "stock"):
    parts = glob.glob(f"{OUT}/part_*/calibration_decoys_{mode}.csv")
    if not parts:
        print(f"no {mode} parts"); continue
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(f"{OUT}/calibration_decoys_{mode}_ALL.csv", index=False)
    for col in ("composite", "ptm", "plddt"):
        rhos = per_target_spearman(df, col)
        rows.append({"mode": mode, "score": col, "mean_spearman": sum(rhos) / len(rhos) if rhos else float("nan"),
                     "median_spearman": pd.Series(rhos).median() if rhos else float("nan"), "n_targets": len(rhos)})

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/calibration_summary_ALL.csv", index=False)
print(res.to_string(index=False))
print("\n=== slim vs stock (composite = the AF2Rank score) ===")
piv = res[res.score == "composite"].set_index("mode")["mean_spearman"]
print(piv.to_string())
