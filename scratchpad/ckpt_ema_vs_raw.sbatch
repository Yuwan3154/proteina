#!/bin/bash
#SBATCH --job-name=emavsraw
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# ⛔ CPU-ONLY on mit_quicktest (15 min max, no GRES there) -- this is tensor comparison, no model and
# no GPU. Deliberately NOT on a GPU partition: it must not compete for the preemptable gres/gpu=4
# cap that the training arms are already starved on, nor for mit_normal_gpu's 2-GPU cap.
# ⛔ Reads checkpoints IN PLACE, no copies: c2c_cb8_tbeta is CANCELLED so its directory is static and
# the branch point is frozen by design. Zero disk written.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"

python -c "import torch" || { echo "FATAL: torch unavailable"; exit 79; }

T=/orcd/scratch/orcd/011/chenxiou/c2c_store
python scratchpad/ckpt_ema_vs_raw.py \
    "$T/branch_points/c2c_cb8_tbeta_fape.ckpt" \
    "$T/c2c_cb8_tbeta/epoch=4-step=8000.ckpt" \
    "$T/c2c_cb8_tbeta/epoch=5-step=8500.ckpt" \
    "$T/c2c_cb8_tbeta/epoch=5-step=9750.ckpt" \
    "$T/c2c_cb8_fape/epoch=0-step=530.ckpt"
echo "EMAVSRAW_EXIT=$?"
