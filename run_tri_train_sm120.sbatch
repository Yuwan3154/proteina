#!/bin/bash
#SBATCH --job-name=tri_sm120
#SBATCH --partition=pi_so3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#
# Triangle arm on RTX PRO 6000 (Blackwell, sm_120). Differs from run_tri_train.sbatch ONLY in the
# environment: the conda env `cue_openfold` is torch+cu126, which has no sm_120 kernels and dies on
# `no kernel image is available for execution on the device`. This uses the uv venv built against
# cu128, whose arch list carries sm_120 (verified on node4007, job 20782762).
# Must precede set -u: Lmod's init references unset variables.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
CONFIG="${CONFIG:-training_contact_tri_full384_v1}"
RUN="${RUN:-tri_full384}"
NGPU="${NGPU:-2}"

module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"

export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_sm120
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_sm120
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?} config=$CONFIG run=$RUN"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| arch', torch.cuda.get_arch_list())"

CHAIN="${CHAIN:-0}"
if [ "$CHAIN" -gt 0 ]; then
    JOBINFO=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n')
    CHAIN_GRES=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TresPerNode=gres/||p')
    CHAIN_TIME=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TimeLimit=||p')
    CHAIN=$((CHAIN - 1)) CONFIG="$CONFIG" RUN="$RUN" NGPU="$NGPU" OVERRIDES="${OVERRIDES:-}" \
        sbatch --dependency=afterany:${SLURM_JOB_ID} \
               --partition="$SLURM_JOB_PARTITION" \
               --gres="$CHAIN_GRES" \
               --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
               --mem="${SLURM_MEM_PER_NODE}M" \
               --time="$CHAIN_TIME" \
               "$REPO/run_tri_train_sm120.sbatch" \
        && echo "[$(date)] queued next chain segment on $SLURM_JOB_PARTITION $CHAIN_GRES $CHAIN_TIME"
fi

# unquoted on purpose: OVERRIDES is a space-separated list of hydra key=value pairs
exec torchrun --standalone --nproc_per_node="$NGPU" proteinfoundation/train.py \
    --config_name "$CONFIG" \
    --ngpus_per_node "$NGPU" \
    --nnodes 1 \
    run_name_="$RUN" \
    ${OVERRIDES:-}
