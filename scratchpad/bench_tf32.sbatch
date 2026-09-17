#!/bin/bash
#SBATCH --job-name=tf32bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# ⛔ NO --partition / --gres here ON PURPOSE: they are passed at submit time so the SAME script runs
# on rtx_pro_6000 and on h200. Baking them in is what made the previous comparison unrepeatable.
# ⛔ Submit BOTH on pi_so3 (which has rtx_pro_6000:2 AND h200:2) so the card is the only variable --
# mit_preemptable nodes are shared with other users and that contention confounded the first attempt.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader
echo "node=$(hostname)  other jobs on this node:"
squeue -w "$(hostname -s)" -h -O "JobID:11,UserName:11,tres-alloc:34" | grep -v "^$SLURM_JOB_ID" || true

python scratchpad/bench_tf32.py --steps "${STEPS:-6}"
echo "TF32BENCH_EXIT=$?"
