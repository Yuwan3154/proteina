#!/bin/bash
#SBATCH --job-name=subfloor
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=14:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u

cd /orcd/scratch/orcd/011/chenxiou/proteina_cmhier

echo "################ local_attn (val-loss minimum: epoch 63) ################"
python scratchpad/floor_vs_model_over_time.py curves/overfit_localattn_full_history.csv 63

if [ -f curves/overfit_convnext_full_history.csv ]; then
  echo
  echo "################ conv_next (val-loss minimum: epoch 119) ################"
  python scratchpad/floor_vs_model_over_time.py curves/overfit_convnext_full_history.csv 119
fi
