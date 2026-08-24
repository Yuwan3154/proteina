#!/bin/bash
#SBATCH --job-name=trainprobe
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# Prerequisite for the train-vs-val trajectory experiment: a list of TRAINING chains.
# The CSV cannot be parsed with awk (commas inside quoted name/ligands fields), so read it
# with pandas and report what the `split` column actually contains before relying on it.
source /etc/profile >/dev/null 2>&1
set -uo pipefail
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
python - <<'PY'
import pandas as pd
D = "/orcd/pool/006/chenxiou/proteina/data/pdb_train"
F = f"{D}/df_pdb_f1_minl50_maxl384_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.csv"
df = pd.read_csv(F, low_memory=False)
print("rows:", len(df), "| cols:", list(df.columns))
print("\nsplit value counts (NaN included):")
print(df["split"].value_counts(dropna=False).head(10))
print("\nprocessed_length describe:")
print(df["processed_length"].describe())
# The val fixed set is <=256; report how many chains would be eligible for a matched probe.
elig = df[(df["processed_length"] >= 50) & (df["processed_length"] <= 256)]
print(f"\nchains with 50 <= processed_length <= 256: {len(elig)}")
PY
