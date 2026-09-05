#!/bin/bash
#SBATCH --job-name=c2c
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# ⛔ mit_normal_gpu ONLY, and h200:2 is exactly that partition's per-user GPU cap
# (QOS mit_normal_gpu: gres/gpu=2). The generation array is pinned to mit_preemptable for the same
# reason -- if a genmaps shard landed here it would consume this cap and training could never start.
# pi_so3's H200s are held by another user; if they free, this can move there.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_sh
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
# ⛔ Short TMPDIR on the checkpoints' own device: 82 chars once cost this project a whole training
# chain via the 108-byte AF_UNIX sun_path limit, and it HANGS rather than crashing.
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR" /orcd/scratch/orcd/011/chenxiou/c2c_store/logs
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scratchpad/train_c2c.py --devices "${DEVICES:-2}" --accum "${ACCUM:-16}" ${EXTRA:-}
rc=$?
echo "C2C_EXIT=$rc"
exit $rc
