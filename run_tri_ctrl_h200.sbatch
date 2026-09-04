#!/bin/bash
#SBATCH --job-name=tri_ctrl
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_ot/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_ot/logs/%x-%j.err
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

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_ot
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
CONFIG="${CONFIG:-training_contact_tri_full384_ctrl_v1}"
RUN="${RUN:-tri_full384_ctrl}"
NGPU="${NGPU:-2}"

module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"

export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_sm120
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_sm120
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs"
# Lightning's atomic ckpt save renames a temp file and a cross-device rename is not atomic (see 07dbb58); it also made wandb copy rather than rename, which is what hit EDQUOT on 2026-09-02.
# AF_UNIX sun_path is 108 bytes and python appends ~32 for the DataLoader worker socket, so a long
# TMPDIR makes every worker->parent batch handoff fail and the run HANG (reads as a dataloader
# timeout, not a path error). Same device as the checkpoints is what the corruption fix needs; being
# UNDER the checkpoints dir was never required and is what blew the limit. Fail fast, never silently.
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?} config=$CONFIG run=$RUN"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| arch', torch.cuda.get_arch_list())"

CHAIN="${CHAIN:-0}"
# Resubmit THIS launcher. The original chained "$REPO/run_tri_train_sm120.sbatch" unconditionally,
# so a derived launcher's successors silently came back as the tri BASELINE script -- wrong job name
# and wrong TORCH_CUDA_ARCH_LIST -- and only looked healthy because CONFIG/RUN ride through the
# environment and --partition/--gres are passed explicitly.
# Named literally, not via $0/BASH_SOURCE: SLURM executes a spool copy of the script, so those
# resolve to a path that is deleted when the job ends.
CHAIN_SELF="$REPO/run_tri_ctrl_h200.sbatch"
# ⛔⛔ Submit a successor AT MOST ONCE per job id. A preempted job is requeued under the SAME
# SLURM_JOB_ID and re-runs this script from the top, so without this guard every preemption queues
# ANOTHER successor -- and they all carry --dependency=afterany on the same id, so they would all
# become eligible together and train the same RUN into the same store dir concurrently. Two
# processes writing one last.ckpt is how this project produced a 0-byte checkpoint before.
# Observed live on 2026-09-04: 21944570 was preempted once and had queued both 21950636 and
# 21969672 against itself.
CHAIN_MARK="$REPO/.chain_submitted.${SLURM_JOB_ID:-none}"
if [ "$CHAIN" -gt 0 ] && [ -e "$CHAIN_MARK" ]; then
    echo "[$(date)] successor already queued for job $SLURM_JOB_ID (requeue detected); not queueing another"
    CHAIN=0
fi
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
               --job-name="$SLURM_JOB_NAME" \
               "$CHAIN_SELF" \
        && touch "$CHAIN_MARK" \
        && echo "[$(date)] queued next chain segment on $SLURM_JOB_PARTITION $CHAIN_GRES $CHAIN_TIME (CHAIN=$((CHAIN - 1)) remaining)"
fi

# unquoted on purpose: OVERRIDES is a space-separated list of hydra key=value pairs
exec torchrun --standalone --nproc_per_node="$NGPU" proteinfoundation/train.py \
    --config_name "$CONFIG" \
    --ngpus_per_node "$NGPU" \
    --nnodes 1 \
    run_name_="$RUN" \
    ${OVERRIDES:-}
