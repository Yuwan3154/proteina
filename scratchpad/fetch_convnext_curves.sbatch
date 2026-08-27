#!/bin/bash
#SBATCH --job-name=curves_cn
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# Must precede set -u: Lmod's init references unset variables.
source /etc/profile >/dev/null 2>&1
set -uo pipefail

module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u

export WANDB_SILENT=true
python /orcd/scratch/orcd/011/chenxiou/proteina_cmhier/scratchpad/fetch_convnext_curves.py
