#!/bin/bash
#SBATCH --job-name=c2c
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
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

# ── self-chaining, mit_normal_gpu caps wall-clock at 6 h ───────────────────────────────────────
# Carries every fix earned on the tri launcher this session:
#  - resubmit THIS script via CHAIN_SELF, named literally (SLURM runs a spool COPY, so $0 and
#    BASH_SOURCE point at a path that dies with the job);
#  - pass --job-name through, or successors come back under the wrong name and become invisible to
#    any check that matches by name;
#  - submit AT MOST ONCE per job id. A preempted job is requeued under the SAME SLURM_JOB_ID and
#    re-runs this script from the top, so without the marker every preemption queues ANOTHER
#    successor -- and they all carry --dependency=afterany on the same id, so they would start
#    CONCURRENTLY and train one run into one checkpoint directory from several processes.
# ── single-writer lock ─────────────────────────────────────────────────────────────────────────
# We queue the SAME run against several partitions and card types at once (rtx_pro_6000 / h200 /
# h100) and take whichever is scheduled first. Without a lock, two of them starting would train two
# processes into ONE checkpoint directory. mkdir is atomic on POSIX, so it is the lock.
LOCK=/orcd/scratch/orcd/011/chenxiou/c2c_store/.run.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    HOLDER=$(cat "$LOCK/jobid" 2>/dev/null)
    ST=$(squeue -h -j "$HOLDER" -o %T 2>/dev/null); RC=$?
    # ⛔ Gate on the EXIT CODE. A failed squeue returns an empty string, which reads exactly like
    # "that job is gone" and would have both jobs take the lock and train into the same directory.
    if [ "$RC" -ne 0 ]; then
        echo "[lock] squeue failed (rc=$RC); refusing to judge whether $HOLDER is alive. Exiting."
        exit 0
    fi
    if [ -n "$ST" ]; then
        echo "[lock] job $HOLDER is $ST and holds the run; this duplicate exits cleanly."
        exit 0
    fi
    echo "[lock] stale lock from $HOLDER (no longer queued); taking it."
fi
echo "$SLURM_JOB_ID" > "$LOCK/jobid"
# ⛔ Guard on BASHPID, not a bare trap. bash runs an EXIT trap when a COMMAND SUBSTITUTION
# subshell exits, so a plain `trap 'rm -rf $LOCK' EXIT` is destroyed by the very next $(...) --
# observed: the $(hostname) on the line below deleted the lock ~0.1 s after it was taken, while
# the job kept running. $$ does not help: it stays the parent PID inside subshells. BASHPID does.
LOCK_PID=$BASHPID
trap '[ "$BASHPID" = "$LOCK_PID" ] && rm -rf "$LOCK"' EXIT
HOSTN=$(hostname)
echo "[lock] held by $SLURM_JOB_ID on $HOSTN"

# The successor must ask for the SAME card this job actually landed on -- we queue against
# rtx_pro_6000 / h200 / h100 at once, so a hardcoded gres would migrate the chain to another card.
GRES="${GRES:-gpu:rtx_pro_6000:2}"
CHAIN="${CHAIN:-0}"
CHAIN_SELF="$REPO/scratchpad/train_c2c.sbatch"
CHAIN_MARK="/orcd/scratch/orcd/011/chenxiou/c2c_store/.chain_submitted.${SLURM_JOB_ID:-none}"
if [ "$CHAIN" -gt 0 ] && [ -e "$CHAIN_MARK" ]; then
    echo "[$(date)] successor already queued for $SLURM_JOB_ID (requeue); not queueing another"
    CHAIN=0
fi
if [ "$CHAIN" -gt 0 ]; then
    CHAIN=$((CHAIN - 1)) DEVICES="${DEVICES:-2}" ACCUM="${ACCUM:-8}" NDIFF="${NDIFF:-8}" LR="${LR:-0.0006}" GRES="$GRES" \
        sbatch --dependency=afterany:${SLURM_JOB_ID} \
               --partition="$SLURM_JOB_PARTITION" \
               --gres="$GRES" \
               --time=06:00:00 \
               --job-name="$SLURM_JOB_NAME" \
               "$CHAIN_SELF" \
        && touch "$CHAIN_MARK" \
        && echo "[$(date)] queued next c2c segment (CHAIN=$((CHAIN - 1)) remaining)"
fi

# ⛔ torchrun, not bare python, and --ntasks-per-node stays 1. Lightning's SLURMEnvironment
# validates devices against SLURM_NTASKS_PER_NODE and aborts with
#   "You set devices=2 ... but --ntasks-per-node=1 does not match"
# Raising ntasks to 2 is the wrong fix: sbatch would still run this script once, and Lightning would
# then expect srun to have spawned the peers. torchrun sets the torchelastic variables Lightning
# prefers, and it is the pattern the working tri launcher already uses
# (run_tri_train_sm120.sbatch:74 with ntasks-per-node=1).
torchrun --standalone --nproc_per_node="${DEVICES:-2}" \
    scratchpad/train_c2c.py --devices "${DEVICES:-2}" --accum "${ACCUM:-8}" \
    --n_diff "${NDIFF:-8}" --lr "${LR:-0.0006}" \
    ${INIT_FROM:+--init_from "$INIT_FROM"} ${EXTRA:-}
rc=$?
echo "C2C_EXIT=$rc"
exit $rc
