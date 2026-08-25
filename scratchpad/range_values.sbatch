#!/bin/bash
#SBATCH --job-name=rangeval
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1
set -uo pipefail
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
cd /orcd/scratch/orcd/011/chenxiou/proteina_cmhier
python - <<'PY'
import pandas as pd, numpy as np
C = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
RUNS = [("localattn_full384", "local_attn"), ("tri_full384", "tri_mul")]
# All three are on the FIXED 32-chain set, so they are cross-arm comparable.
KEYS = [
    ("validation_sampling/contact_precision_at_L5_mean",              "all pairs   sep>=6"),
    ("validation_sampling/contact_medium_range_precision_at_L5_mean", "medium  12<=sep<24"),
    ("validation_sampling/contact_long_range_precision_at_L5_mean",   "long        sep>=24"),
]
EP_LO, EP_HI = 20, 38     # matched-epoch window (tri's max epoch is 38)

vals = {}
for run, lab in RUNS:
    df = pd.read_csv(f"{C}/{run}_full_history.csv", low_memory=False)
    vals[lab] = {}
    for k, name in KEYS:
        if k not in df.columns:
            vals[lab][name] = float("nan"); continue
        sub = df[["epoch", k]].apply(pd.to_numeric, errors="coerce").dropna()
        sel = sub[(sub["epoch"] >= EP_LO) & (sub["epoch"] <= EP_HI)]
        vals[lab][name] = float(sel[k].median()) if len(sel) else float("nan")

print(f"=== precision@L/5 by sequence separation, MATCHED epochs {EP_LO}-{EP_HI}, fixed 32 chains ===")
print(f"{'range':22s} {'local_attn':>11s} {'tri_mul':>11s} {'la / tri':>10s}")
for _, name in KEYS:
    a, b = vals["local_attn"][name], vals["tri_mul"][name]
    ratio = (a / b) if (b and not np.isnan(a) and not np.isnan(b) and b > 0) else float("nan")
    print(f"{name:22s} {a:>11.4f} {b:>11.4f} {ratio:>10.2f}x")
print("\nIf the ratio GROWS from short to long range, tri's confidence is concentrated")
print("near the diagonal and the deficit is specifically in long-range contacts.")
PY
