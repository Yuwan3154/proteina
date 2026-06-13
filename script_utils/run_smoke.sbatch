#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# Minimal end-to-end smoke: 4 optim steps, 1 GPU, batch 1, no wandb/ckpt.
# Submit with: sbatch --export=ALL,CONFIG_NAME=<exp_config> script_utils/run_smoke.sbatch
set -e
REPO="${REPO:-/orcd/pool/006/chenxiou/proteina}"
cd "$REPO"
module load miniforge 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cue_openfold
export DATA_PATH="${DATA_PATH:-$REPO/data}"
export PYTHONPATH="$REPO"
export SKIP_CONFIND_PRECOMPUTE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u proteinfoundation/train.py \
  --config_name "$CONFIG_NAME" \
  --single --nolog --batch_size 1 opt.max_steps=4 \
  af2_ipa_weights_path="$REPO/../params/params_model_1_ptm.npz"
