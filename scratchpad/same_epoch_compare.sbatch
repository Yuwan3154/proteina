#!/bin/bash
#SBATCH --job-name=sameep
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# local_attn is at epoch 145 and tri at epoch 38, because local_attn runs ~0.26 h/epoch vs
# tri's ~2.34 h. Comparing their ENDPOINTS would credit local_attn for 4x more training.
# The only fair read is at a COMMON epoch.
source /etc/profile >/dev/null 2>&1
set -uo pipefail
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
cd /orcd/scratch/orcd/011/chenxiou/proteina_cmhier
python - <<'PY'
import pandas as pd
C = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
KS = "validation_sampling/contact_precision_at_L_mean"
KM = "validation_sampling/contact_precision_at_L_median"
KG1, KG2 = ("validation_loss/contact_precision_at_L_single_step",
            "validation_loss/contact_precision_at_L_noisy_floor")

def at_epoch(run, ep, key, win=3.0):
    df = pd.read_csv(f"{C}/{run}_full_history.csv", low_memory=False)
    sub = df[["epoch", key]].apply(pd.to_numeric, errors="coerce").dropna()
    sel = sub[(sub["epoch"] >= ep - win) & (sub["epoch"] <= ep + win)]
    return (float(sel[key].mean()), len(sel)) if len(sel) else (float("nan"), 0)

def gap_at(run, ep, win=3.0):
    df = pd.read_csv(f"{C}/{run}_full_history.csv", low_memory=False)
    sub = df[["epoch", KG1, KG2]].apply(pd.to_numeric, errors="coerce").dropna()
    sel = sub[(sub["epoch"] >= ep - win) & (sub["epoch"] <= ep + win)]
    return (float((sel[KG1] - sel[KG2]).median()), len(sel)) if len(sel) else (float("nan"), 0)

EP = 38  # tri's latest epoch = the highest epoch where BOTH arms have data
print(f"=== SAME-EPOCH comparison at epoch {EP} +/- 3 (tri's current max) ===")
for run, lab in (("localattn_full384", "local_attn max384"), ("tri_full384", "tri_mul max384")):
    m, nm = at_epoch(run, EP, KS); md, _ = at_epoch(run, EP, KM); g, ng = gap_at(run, EP)
    print(f"  {lab:20s} sampled p@L mean={m:.4f} median={md:.4f} (n={nm})   single-step gap={g:+.4f} (n={ng})")
print()
print("=== ENDPOINT values (NOT comparable -- different amounts of training) ===")
for run, lab, ep in (("localattn_full384", "local_attn max384", 145),
                     ("tri_full384", "tri_mul max384", 38),
                     ("localattn_full512_1gpu", "local_attn max512 ref", 36)):
    m, _ = at_epoch(run, ep, KS, win=2.0); g, _ = gap_at(run, ep, win=2.0)
    print(f"  {lab:22s} @ep{ep:<4d} sampled p@L mean={m:.4f}  single-step gap={g:+.4f}")
PY
