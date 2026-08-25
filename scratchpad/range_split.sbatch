#!/bin/bash
#SBATCH --job-name=rangesplit
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# Tests the diagonal hypothesis with ZERO GPU: _compute_contact_map_metrics already computes
# long-range (sep>=24) and medium-range (12<=sep<24) precision alongside the all-pairs number.
# If tri's deficit WIDENS as short-range pairs are excluded, its confidence really is
# concentrated near the diagonal.
source /etc/profile >/dev/null 2>&1
set -uo pipefail
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
cd /orcd/scratch/orcd/011/chenxiou/proteina_cmhier
export WANDB_SILENT=true RUN_IDS="localattn_full384,tri_full384"
python scratchpad/live_curve_stats.py >/dev/null 2>&1   # refresh cached history
python - <<'PY'
import pandas as pd, numpy as np
C = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
RUNS = [("localattn_full384", "local_attn"), ("tri_full384", "tri_mul")]
EP_LO, EP_HI = 20, 38          # matched-epoch window (tri's max is 38)

for run, lab in RUNS:
    df = pd.read_csv(f"{C}/{run}_full_history.csv", low_memory=False)
    print(f"\n=== {lab} ===")
    cand = sorted(c for c in df.columns if "range" in c.lower() or "at_L" in c)
    print("  range-resolved keys present:")
    for c in cand:
        n = int(df[c].notna().sum())
        if n:
            print(f"    {c}  (n={n})")
PY
