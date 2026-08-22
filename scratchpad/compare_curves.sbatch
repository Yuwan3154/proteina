#!/bin/bash
#SBATCH --job-name=cmpcurves
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u
# refresh the cached history CSVs first so the figure is not plotted from a stale fetch
export WANDB_SILENT=true
export RUN_IDS="localattn_full512_1gpu,tri_full384"
python /orcd/scratch/orcd/011/chenxiou/proteina_cmhier/scratchpad/live_curve_stats.py
python /orcd/scratch/orcd/011/chenxiou/proteina_cmhier/scratchpad/compare_curves.py
