#!/bin/bash
#SBATCH --job-name=tbinchk
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
BINS = ["tlow", "tmid", "thigh"]
SIG = "validation_loss/contact_precision_at_L_single_step_{b}"
FLR = "validation_loss/contact_precision_at_L_noisy_floor_{b}"
RUNS = [("localattn_full384", "local_attn"), ("tri_full384", "tri_mul")]

print("The noisy floor MUST rise monotonically with t (at t->1 the input IS the true map).")
print("A violation means the binned estimate is unreliable, not that physics changed.\n")
for run, lab in RUNS:
    df = pd.read_csv(f"{C}/{run}_full_history.csv", low_memory=False)
    print(f"=== {lab} ===")
    for scope, lo, hi in (("ALL epochs", -1, 1e9), ("ep 34-42", 34, 42)):
        out = []
        for b in BINS:
            ks, kf = SIG.format(b=b), FLR.format(b=b)
            if kf not in df.columns: out.append(f"{b}=ABSENT"); continue
            sub = df[["epoch", ks, kf]].apply(pd.to_numeric, errors="coerce").dropna()
            sel = sub[(sub["epoch"] >= lo) & (sub["epoch"] <= hi)]
            if not len(sel): out.append(f"{b}=n/a"); continue
            out.append(f"{b}: floor={sel[kf].median():.4f} sig={sel[ks].median():.4f} "
                       f"gap={(sel[ks]-sel[kf]).median():+.4f} n={len(sel)}")
        print(f"  [{scope}]")
        for o in out: print("     ", o)
        fl = []
        for b in BINS:
            kf = FLR.format(b=b)
            if kf not in df.columns: continue
            sub = df[["epoch", kf]].apply(pd.to_numeric, errors="coerce").dropna()
            sel = sub[(sub["epoch"] >= lo) & (sub["epoch"] <= hi)]
            if len(sel): fl.append(sel[kf].median())
        if len(fl) == 3:
            print(f"      floor monotonic in t? {'YES' if fl[0] < fl[1] < fl[2] else '*** NO -- estimate unreliable ***'}"
                  f"  ({fl[0]:.3f} -> {fl[1]:.3f} -> {fl[2]:.3f})")
    print()
PY
