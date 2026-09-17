#!/bin/bash
#SBATCH --job-name=ckptcad
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# CPU-only: constructs ModelCheckpoint callbacks and reads back their intervals. No GPU, no training.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
# activate.sh hard-requires this on Engaging and there is no nvidia-smi on a CPU node. Inert here.
export TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export CUE_HOST=engaging
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

python scratchpad/test_ckpt_cadence.py
echo "CKPTCAD_EXIT=$?"
