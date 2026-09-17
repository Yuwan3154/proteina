#!/bin/bash
#SBATCH --job-name=dlbench
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
# ⛔ This benchmark uses NO GPU -- it times the dataloader only. Submitting it with --gres put
# it behind QOSMaxGRESPerUser (preemptable cap 4, and tri alone holds 2), so it could never
# start. Run it GRES-free. activate.sh still demands TORCH_CUDA_ARCH_LIST, so default it when
# nvidia-smi is absent; it is inert here because nothing CUDA is compiled or launched.
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
[ -z "$TORCH_CUDA_ARCH_LIST" ] && export TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "(no GPU -- expected)"
echo "node=$(hostname)  other jobs on this node:"
squeue -w "$(hostname -s)" -h -O "JobID:11,UserName:11,tres-alloc:34" | grep -v "^$SLURM_JOB_ID" || true

python scratchpad/bench_dataloader.py --batches "${BATCHES:-40}" --workers "${WORKERS:-0,4,8,16}"
echo "DLBENCH_EXIT=$?"
