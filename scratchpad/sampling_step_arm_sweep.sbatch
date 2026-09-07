#!/bin/bash
#SBATCH --job-name=stepsweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Stratified validation of the tri contact model by conditioning class.
# ⛔ devices=1 -- the per-class metric keys are data-dependent and this repo has a documented
# 2-GPU DDP deadlock from exactly that shape.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_sh
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"

# ⛔ Copy the checkpoint: tri_sm120 is LIVE and rewrites last-EMA.ckpt; reading it mid-write gives
# a truncated file. Never read the running run's checkpoint in place.
SRC=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/store/tri_full384/checkpoints/last-EMA.ckpt
CK="$TMPDIR/nonself-$SLURM_JOB_ID.ckpt"
cp "$SRC" "$CK" || { echo "FATAL: cannot copy $SRC"; exit 77; }
echo "checkpoint copied from $SRC"

python scratchpad/sampling_step_arm_sweep.py --ema_ckpt "$CK" \
    --steps ${STEPS:-25 50 100 200} --arms ${ARMS:-self nonself}
rm -f "$CK"
echo "NONSELF_EXIT=$?"
