#!/bin/bash
#SBATCH --job-name=tri_cb8synth
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_tri/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_tri/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@60
#
# tri retrain: CB-8 contacts + synthetic references + vocab 44 + aux heads, 4x H200 on mit_preemptable
# (user directive 2026-09-09; h200:4 is the whole per-user preemptable GPU cap). Clone of
# run_tri_ctrl_h200.sbatch with: NGPU 4, --cpus-per-task 16 (4 dataloader workers per rank; the 2-GPU
# run averaged ~2 busy cores), --mem 512G (user floor; measured need is ~21 GB RSS for the 2-GPU run
# plus a ~3 GB mmap'd index; H200 nodes carry 2 TB), the per-job-name lock, and the three env
# values the yamls read (SYNTH_INDEX_DIR, ALIGN_W, MLM_W -- REQUIRED, no defaults: the weights are
# set from the recorded sweep, never from this file).
#
# --ntasks-per-node stays 1 + torchrun --nproc_per_node=4: Lightning probes TorchElastic before
# SLURM, so this is the working DDP pattern in this repo (reference_sbatch_gres_needs_ntasks_per_node).
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SYNTH_INDEX_DIR="${SYNTH_INDEX_DIR:-/orcd/scratch/orcd/011/chenxiou/synth_index}"
export ALIGN_W="${ALIGN_W:?ALIGN_W (alignment loss weight) must be set from the recorded sweep}"
export MLM_W="${MLM_W:?MLM_W (masked-token loss weight) must be set from the recorded sweep}"
CONFIG="${CONFIG:-training_contact_tri_full384_cb8synth_v1}"
RUN="${RUN:-tri_cb8synth}"
NGPU="${NGPU:-4}"

module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"

export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_sm120
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_sm120
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs"
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?} config=$CONFIG run=$RUN ngpu=$NGPU align_w=$ALIGN_W mlm_w=$MLM_W index=$SYNTH_INDEX_DIR"
git log --oneline -1
[ -f "$SYNTH_INDEX_DIR/topology_index_cb8_synth.pt" ] || { echo "FATAL: no synthetic index at $SYNTH_INDEX_DIR"; exit 77; }
[ -f "$SYNTH_INDEX_DIR/eligible_ids_tm0.5-0.9.txt" ] || { echo "FATAL: no eligible-ids list at $SYNTH_INDEX_DIR"; exit 77; }
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| arch', torch.cuda.get_arch_list())"

# ── single-writer lock, keyed by job name (from scratchpad/train_c2c.sbatch) ────────────────────
# A requeued (preempted) job carries the SAME job id and may continue; any other holder that is
# still queued means this copy must exit; a holder that vanished left a stale lock to take over.
LOCK=/orcd/scratch/orcd/011/chenxiou/c2c_store/.run.lock.${SLURM_JOB_NAME}
if ! mkdir "$LOCK" 2>/dev/null; then
    HOLDER=$(cat "$LOCK/jobid" 2>/dev/null)
    if [ "$HOLDER" != "$SLURM_JOB_ID" ]; then
        ST=$(squeue -h -j "$HOLDER" -o %T 2>/dev/null); RC=$?
        if [ "$RC" -ne 0 ]; then
            echo "[lock] squeue failed (rc=$RC); refusing to judge whether $HOLDER is alive. Exiting."; exit 0
        fi
        if [ -n "$ST" ]; then
            echo "[lock] job $HOLDER is $ST and holds the run; this duplicate exits cleanly."; exit 0
        fi
        echo "[lock] stale lock from $HOLDER (no longer queued); taking it."
    fi
fi
echo "$SLURM_JOB_ID" > "$LOCK/jobid"
LOCK_PID=$BASHPID
trap '[ "$BASHPID" = "$LOCK_PID" ] && rm -rf "$LOCK"' EXIT
echo "[lock] held by $SLURM_JOB_ID on $(hostname)"

CHAIN="${CHAIN:-0}"
CHAIN_SELF="$REPO/run_tri_cb8synth_h200x4.sbatch"
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
    SYNTH_INDEX_DIR="$SYNTH_INDEX_DIR" ALIGN_W="$ALIGN_W" MLM_W="$MLM_W" \
        sbatch --dependency=afterany:${SLURM_JOB_ID} \
               --partition="$SLURM_JOB_PARTITION" \
               --gres="$CHAIN_GRES" \
               --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
               --mem="${SLURM_MEM_PER_NODE}M" \
               --time="$CHAIN_TIME" \
               --job-name="$SLURM_JOB_NAME" \
               "$CHAIN_SELF" \
        && touch "$CHAIN_MARK" \
        && echo "[$(date)] queued next chain segment on $SLURM_JOB_PARTITION $CHAIN_GRES $CHAIN_TIME (CHAIN=$((CHAIN - 1)) remaining)"
fi

# not exec'd: the EXIT trap above must still run to release the lock when torchrun ends
# unquoted on purpose: OVERRIDES is a space-separated list of hydra key=value pairs
torchrun --standalone --nproc_per_node="$NGPU" proteinfoundation/train.py \
    --config_name "$CONFIG" \
    --ngpus_per_node "$NGPU" \
    --nnodes 1 \
    run_name_="$RUN" \
    ${OVERRIDES:-}
rc=$?
echo "TRI_EXIT=$rc"
exit $rc
