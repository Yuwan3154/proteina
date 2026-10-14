#!/bin/bash
#SBATCH --job-name=tri_t8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_t8/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_t8/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@60
# node5101: DDP dies in the first NCCL broadcast; node5204: under-allocates GRES; node3006: CUDA init failure (10-13).
#SBATCH --exclude=node5101,node5204,node3006
#
# T8 ConFind tri v2 (user 2026-10-09..14). Clone of run_tri_cb8synth_h200x4.sbatch with the T8 config, index, pack and
# compile caches. Partition / --gres / --time are given at submit. GPU TYPE IS PINNED for the run (user 2026-10-14 "(a) ...
# Then stick to that GPU type"): chain successors re-request this job's exact TresPerNode, and --requeue keeps it.
# Per-GPU batch and accumulation are REQUIRED env (BATCH, ACCUM; set from the measured peak memory of the pinned type) and
# must give NGPU x BATCH x ACCUM = 32 (effective batch, user D4). torchrun + --ntasks-per-node 1 (repo DDP pattern).
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

REPO="${REPO:-/orcd/scratch/orcd/011/chenxiou/proteina_t8}"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SYNTH_INDEX_DIR="${SYNTH_INDEX_DIR:-/orcd/compute/so3/002/chenxi/t8_index}"
export PACK_PATH="${PACK_PATH:-/orcd/compute/so3/002/chenxi/proteina_pack/tri_t8_512.pack}"
# 0.3 / 0.3 from the 1500-step sweep (user 2026-09-10, job 22483230); T8 keeps the recipe (Q10)
export ALIGN_W="${ALIGN_W:-0.3}"
export MLM_W="${MLM_W:-0.3}"
CONFIG="${CONFIG:-training_contact_tri_t8_confind_v1}"
RUN="${RUN:-tri_t8_confind_v1}"
BATCH="${BATCH:?BATCH (per-GPU batch, from measured memory) is required}"
ACCUM="${ACCUM:?ACCUM (gradient accumulation) is required}"
[ -f "$SYNTH_INDEX_DIR/topology_index_confind_synth_t8.pt" ] && [ -f "$SYNTH_INDEX_DIR/eligible_ids_tm0.5-0.9.txt" ] \
    || { echo "FATAL: no T8 index under $SYNTH_INDEX_DIR"; exit 76; }
[ -f "$PACK_PATH" ] && [ -f "$PACK_PATH.idx.npz" ] || { echo "FATAL: no pack at $PACK_PATH"; exit 77; }

module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
[ -n "$CC" ] && [ "$(echo "$CC" | wc -l)" -eq 1 ] || { echo "FATAL: need exactly one GPU arch, got: '$CC'"; exit 79; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
[ "$(echo "$GPU_NAME" | wc -l)" -eq 1 ] || { echo "FATAL: mixed GPU models: $GPU_NAME"; exit 79; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$NGPU" -eq "${SLURM_GPUS_ON_NODE:-$NGPU}" ] || { echo "FATAL: nvidia-smi sees $NGPU GPUs, SLURM granted ${SLURM_GPUS_ON_NODE}"; exit 80; }
[[ $BATCH =~ ^[1-9][0-9]*$ && $ACCUM =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: BATCH='$BATCH' ACCUM='$ACCUM' must be positive integers"; exit 81; }
[ $((NGPU * BATCH * ACCUM)) -eq 32 ] || { echo "FATAL: $NGPU x $BATCH x $ACCUM != 32 (effective batch, user D4)"; exit 81; }
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$CC"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"

export SKIP_CONFIND_PRECOMPUTE=1 DATALOADER_TIMEOUT_S=1800
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# ⛔ compile caches on /orcd/compute/so3/002, never scratch (user 2026-10-11: tens of thousands of files filled the
# scratch file quota); per arch, since compiled kernels are arch-specific
CACHE=/orcd/compute/so3/002/chenxi/t8_compile_cache/sm${CC/./}
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID:-nojob}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs" "$TMPDIR"
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO" || exit 3
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?} gpu=$GPU_NAME x$NGPU batch=$BATCH accum=$ACCUM config=$CONFIG run=$RUN index=$SYNTH_INDEX_DIR"
git log --oneline -1

# one live segment per run name (same lock as the CB-8 launcher: a LIVE holder state, never "squeue printed something")
LOCK=/orcd/scratch/orcd/011/chenxiou/c2c_store/.run.lock.${SLURM_JOB_NAME}
if ! mkdir "$LOCK" 2>/dev/null; then
    HOLDER=$(cat "$LOCK/jobid" 2>/dev/null)
    if [ "$HOLDER" != "$SLURM_JOB_ID" ]; then
        # a job slurmctld has purged makes squeue exit 1 with "Invalid job id" (measured 2026-10-14): that is GONE, not a
        # failed query; any other failure still refuses to judge
        ST=""; RC=0; ERR=""
        if [ -n "$HOLDER" ]; then
            # ONE call: state, error text and exit code from the same query
            ST=$(squeue -h -j "$HOLDER" -o %T 2>"$TMPDIR/lock_sq.err"); RC=$?; ERR=$(cat "$TMPDIR/lock_sq.err")
            if [ "$RC" -ne 0 ] && printf '%s' "$ERR" | grep -q "Invalid job id"; then ST=""; RC=0; fi
        fi
        [ "$RC" -eq 0 ] || { echo "[lock] squeue failed (rc=$RC: $ERR); refusing to judge whether $HOLDER is alive. Exiting."; exit 0; }
        # take the lock only from a holder that cannot be training: gone, PENDING (requeued after a SIGKILL'd segment
        # that never ran its EXIT trap) or terminal; any other state (RUNNING, SIGNALING, REQUEUED, ...) may have ranks
        case "$ST" in
            ""|PENDING|COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)
                echo "[lock] stale lock from $HOLDER (state '${ST:-gone}'); taking it." ;;
            *)
                echo "[lock] job $HOLDER is $ST and may hold the run; this duplicate exits cleanly."; exit 0 ;;
        esac
    fi
fi
mkdir -p "$LOCK"
echo "$SLURM_JOB_ID" > "$LOCK/jobid"
LOCK_PID=$BASHPID
trap '[ "$BASHPID" = "$LOCK_PID" ] && rm -rf "$LOCK"' EXIT
echo "[lock] held by $SLURM_JOB_ID on $(hostname)"

# the GPU type stays pinned for the whole run (user 2026-10-14): every segment, not only chaining ones, needs a typed gres
JOBINFO=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n')
CHAIN_GRES=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TresPerNode=gres/||p')
CHAIN_TIME=$(printf '%s\n' "$JOBINFO" | sed -n 's|^TimeLimit=||p')
case "$CHAIN_GRES" in gpu:*:*) ;; *) echo "FATAL: gres '$CHAIN_GRES' has no GPU type; the type must stay pinned"; exit 82;; esac

CHAIN="${CHAIN:-0}"
CHAIN_MARK="$REPO/.chain_submitted.${SLURM_JOB_ID:-none}"
if [ "$CHAIN" -gt 0 ] && [ -e "$CHAIN_MARK" ]; then
    echo "[$(date)] successor already queued for job $SLURM_JOB_ID (requeue detected); not queueing another"
    CHAIN=0
fi
if [ "$CHAIN" -gt 0 ]; then
    CHAIN=$((CHAIN - 1)) CONFIG="$CONFIG" RUN="$RUN" BATCH="$BATCH" ACCUM="$ACCUM" REPO="$REPO" OVERRIDES="${OVERRIDES:-}" \
    SYNTH_INDEX_DIR="$SYNTH_INDEX_DIR" PACK_PATH="$PACK_PATH" ALIGN_W="$ALIGN_W" MLM_W="$MLM_W" \
        sbatch --dependency=afterany:${SLURM_JOB_ID} --partition="$SLURM_JOB_PARTITION" --gres="$CHAIN_GRES" \
               --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --mem="${SLURM_MEM_PER_NODE}M" --time="$CHAIN_TIME" \
               --job-name="$SLURM_JOB_NAME" "$REPO/run_tri_t8.sbatch" \
        && touch "$CHAIN_MARK" \
        && echo "[$(date)] queued next chain segment on $SLURM_JOB_PARTITION $CHAIN_GRES $CHAIN_TIME (CHAIN=$((CHAIN - 1)) remaining)"
fi

# not exec'd: the EXIT trap must still release the lock; OVERRIDES unquoted on purpose (space-separated hydra pairs)
torchrun --standalone --nproc_per_node="$NGPU" proteinfoundation/train.py \
    --config_name "$CONFIG" --ngpus_per_node "$NGPU" --nnodes 1 \
    run_name_="$RUN" +opt.batch_size="$BATCH" opt.accumulate_grad_batches="$ACCUM" \
    ${OVERRIDES:-}
rc=$?
echo "T8_EXIT=$rc"
exit $rc
