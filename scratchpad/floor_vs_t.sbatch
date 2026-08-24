#!/bin/bash
#SBATCH --job-name=floorvt2
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# The floor depends ONLY on (data, t). Both arms draw from the same 4158-chain val split, so
# their floor-vs-t curves MUST coincide. `contact_single_step_t` is logged alongside, so this
# tests the relationship directly on already-cached data -- no model, no GPU.
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
KT = "validation_loss/contact_single_step_t"
KF = "validation_loss/contact_precision_at_L_noisy_floor"
KS = "validation_loss/contact_precision_at_L_single_step"
RUNS = [("localattn_full384", "local_attn (batch 4)"), ("tri_full384", "tri_mul (batch 1)")]

for run, lab in RUNS:
    df = pd.read_csv(f"{C}/{run}_full_history.csv", low_memory=False)
    have = [k for k in (KT, KF, KS) if k in df.columns]
    print(f"=== {lab} ===  present: {[k.split('/')[-1] for k in have]}")
    if KT not in df.columns or KF not in df.columns:
        print("   cannot test: t or floor missing\n"); continue
    cols = [KT, KF, KS, "epoch"]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[(sub["epoch"] >= 20) & (sub["epoch"] <= 38)]   # MATCHED epoch window (tri max = 38)
    print(f"   [MATCHED epochs 20-38]  n={len(sub)}   t range [{sub[KT].min():.3f}, {sub[KT].max():.3f}]  mean t={sub[KT].mean():.3f}")
    # floor as a function of the RECORDED t -- this is the model-independent curve
    edges = np.arange(0.0, 1.0001, 0.1)
    print("     t-bin      n    floor(median)   signal(median)")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = sub[(sub[KT] >= lo) & (sub[KT] < hi)]
        if len(sel) >= 3:
            print(f"   [{lo:.1f},{hi:.1f})  {len(sel):5d}      {sel[KF].median():.4f}         {sel[KS].median():.4f}")
    print()
PY
