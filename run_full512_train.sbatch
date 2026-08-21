#!/bin/bash
#SBATCH --job-name=full512
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#
# FULL dataset at max length 512, 2-GPU DDP. Replaces the overfit-2000 regime, whose two arms both
# peaked on validation early (local_attn epoch 63, conv_next 119) and then degraded 2.4x.
# Batch size is MEASURED, not assumed: hier at L=512 on a 44.4 GB L40S fits batch 6 (96% of card),
# and batch 4 sits at 65%, so the dataset config's batch 4 is the headroom-safe choice.
# 2 GPUs x batch 4 x accumulate 4 = effective batch 32, the invariant shared with the overfit arms.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
CONFIG="${CONFIG:-training_contact_hier_topology_full512_v1}"
RUN="${RUN:-localattn_full512}"
NGPU="${NGPU:-2}"

module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u

export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_full512
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_full512
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ⛔ SECOND, SEPARATE timeout. opt.dist_timeout_s governs the COLLECTIVE timeout; this one governs
# ProcessGroupNCCL's heartbeat monitor of its own watchdog THREAD, defaults to 480 s, and hard-
# aborts the process (SIGABRT, no Python traceback) 8 min later. Job 20873965 died exactly that
# way at 19:23:34 with ZERO collective timeouts, after surviving the 600 s collective limit.
# Both GPUs were measured at 100% utilisation while "stuck", which is the documented GIL-held-
# inside-a-CUDA-API case, not a deadlock -- so the monitor is firing on a slow startup.
# Matched to dist_timeout_s=3600 so the two limits cannot disagree.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?} config=$CONFIG run=$RUN ngpu=$NGPU"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

CHAIN="${CHAIN:-0}"
if [ "$CHAIN" -gt 0 ]; then
    # Inherit THIS segment's placement; a bare `sbatch` would revert to the #SBATCH defaults and
    # silently discard any CLI override, which already cost three segments on the tri launcher.
    JOBINFO=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n')
    CHAIN_GRES=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TresPerNode=gres/||p')
    CHAIN_TIME=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TimeLimit=||p')
    CHAIN=$((CHAIN - 1)) CONFIG="$CONFIG" RUN="$RUN" NGPU="$NGPU" \
        sbatch --dependency=afterany:${SLURM_JOB_ID} \
               --partition="$SLURM_JOB_PARTITION" \
               --gres="$CHAIN_GRES" \
               --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
               --mem="${SLURM_MEM_PER_NODE}M" \
               --time="$CHAIN_TIME" \
               "$REPO/run_full512_train.sbatch" \
        && echo "[$(date)] queued next chain segment on $SLURM_JOB_PARTITION $CHAIN_GRES $CHAIN_TIME (CHAIN=$((CHAIN - 1)) remaining)"
fi

exec torchrun --standalone --nproc_per_node="$NGPU" proteinfoundation/train.py \
    --config_name "$CONFIG" \
    --ngpus_per_node "$NGPU" \
    --nnodes 1 \
    run_name_="$RUN"
