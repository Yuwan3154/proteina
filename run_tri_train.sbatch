#!/bin/bash
#SBATCH --job-name=tri_train
#SBATCH --partition=pi_so3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#
# Third architecture: 2D-only, triangle multiplicative updates, topology reference concatenated
# onto the token axis. The grid is (L+T)^2 rather than L^2 and every block materialises several
# [B, N, N, c] intermediates, so VRAM is the expected failure mode -- watch for OOM.
# cpus-per-task is deliberately small: pi_so3 nodes run 30-31 of 32 cores allocated, so a larger
# request will not start however many GPUs are free.
set -uo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
CONFIG="${CONFIG:-training_contact_tri_overfit_v1}"
RUN="${RUN:-overfit_tri}"

module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u

export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_tri
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_tri
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs"
# Fragmentation is the usual proximate cause of OOM on a model this activation-heavy.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?} config=$CONFIG run=$RUN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

CHAIN="${CHAIN:-0}"
if [ "$CHAIN" -gt 0 ]; then
    # The chained segment must inherit THIS segment's placement, not the #SBATCH defaults
    # above: a CLI-overridden launch (--partition=mit_normal_gpu --gres=gpu:l40s:1) silently
    # reverted to pi_so3/rtx_pro_6000 on resubmission and three segments died on sm_120.
    # TresPerNode is the REQUESTED gres, so a scheduler over-allocation does not propagate.
    JOBINFO=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n')
    CHAIN_GRES=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TresPerNode=gres/||p')
    CHAIN_TIME=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TimeLimit=||p')
    CHAIN=$((CHAIN - 1)) CONFIG="$CONFIG" RUN="$RUN" \
        sbatch --dependency=afterany:${SLURM_JOB_ID} \
               --partition="$SLURM_JOB_PARTITION" \
               --gres="$CHAIN_GRES" \
               --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
               --mem="${SLURM_MEM_PER_NODE}M" \
               --time="$CHAIN_TIME" \
               "$REPO/run_tri_train.sbatch" \
        && echo "[$(date)] queued next chain segment on $SLURM_JOB_PARTITION $CHAIN_GRES $CHAIN_TIME"
fi

exec torchrun --standalone --nproc_per_node=1 proteinfoundation/train.py \
    --config_name "$CONFIG" \
    --ngpus_per_node 1 \
    --nnodes 1 \
    run_name_="$RUN"
