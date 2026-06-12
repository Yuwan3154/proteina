#!/bin/bash
#SBATCH --job-name=cb_domaincrop_s1
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
# 200G: the host-RAM dataloader-worker CoW leak that pushed this past 200G near the 6h
# mark is fixed (persistent_workers + gc.freeze + FrozenStrMap refcount-free dicts +
# ncpus_per_task_train_ 4); real-run peak is now ~25G, so 200G has ample margin.
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.err

# CATH-balanced domain-crop Stage-1, combined PDB+AFDB, on mit_normal_gpu (2-GPU
# per-user cap, non-preemptable, 6h job limit). Effective batch = 2 GPU * 2 micro
# * 32 accum = 128 (batch 4 OOMs on 2-GPU L40S; batch 2 fits). config_name carries
# the training_ prefix: the file is training_<run>.yaml and train.py passes
# config_name verbatim to hydra. Config path resolves via the pip -e
# proteinfoundation __file__ at /orcd/pool, independent of REPO/CWD.
#
# 6h chain: --resume_option=allow resumes from last.ckpt. On a non-zero exit we
# resubmit: if a checkpoint advanced this run (time-limit SIGTERM, or a crash after
# progress) we reset state and resubmit on all nodes; if NO checkpoint advanced (a
# startup fault -- bad GPU/ECC, NCCL, node) we exclude the faulting node and resubmit,
# capping consecutive no-progress retries at MAX_NOPROGRESS (=3) so a deterministic bug
# can't infinite-loop. A clean finish / max_steps exits 0 and the chain self-terminates.
# No --requeue (mit_normal_gpu is non-preemptable; the resubmit chain covers the limit).

set -euo pipefail
RUN=dssp_contact_20M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
REPO=/home/chenxiou/proteina  # NOT the pip -e source root (avoids srun+hydra relative-config-path failure); imports still resolve to pip -e merged-main
# torchrun runs in the BACKGROUND below so this trap can fire: a bash trap is DEFERRED
# while waiting on a FOREGROUND child, so the previous foreground torchrun made SIGUSR1
# (sent 120s before the 6h limit by --signal=B:USR1@120) never run -> SLURM's SIGKILL
# bypassed the resubmit logic -> the chain died at the first clean 6h timeout. Here we
# SIGTERM torchrun for a graceful stop so the resubmit logic below runs within the 120s.
trap 'echo "[$(date)] SIGUSR1: 120s to time limit; SIGTERM torchrun so the resubmit fires."; kill -TERM ${TORCH_PID:-0} 2>/dev/null || true' USR1

mkdir -p "$REPO/store/$RUN/slurm"

module load miniforge cuda/13.0.1
source "$(conda info --base 2>/dev/null || echo /opt/conda)/etc/profile.d/conda.sh"
conda activate cue_openfold

export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/orcd/compute/so3/001/chenxi/torchinductor_cache
export TRITON_CACHE_DIR=/orcd/compute/so3/001/chenxi/triton_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO"
echo "[$(date)] Launching $RUN on $(hostname) GPUs $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed s/,$//) jobid=${SLURM_JOB_ID:-?} restart=${SLURM_RESTART_COUNT:-0}"

CKPT=$REPO/store/$RUN/checkpoints/last.ckpt
START_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)

# torchrun spawns the 2 GPU workers as plain python processes and sets
# RANK/WORLD_SIZE/LOCAL_RANK, which train.py reads for the manual NCCL init.
torchrun --standalone --nnodes=1 --nproc_per_node=2 proteinfoundation/train.py \
    --config_name "training_$RUN" \
    --ngpus_per_node 2 \
    --nnodes 1 \
    --batch_size 2 \
    --accumulate_grad_batches 32 \
    --resume_option allow \
    af2_ipa_weights_path=/orcd/pool/006/chenxiou/params/params_model_1_ptm.npz &
TORCH_PID=$!
# Wait for torchrun, RE-WAITING if the SIGUSR1 trap interrupts the wait, until it truly
# exits. RC = torchrun's real exit code (set -e safe via the || guard).
RC=0
while kill -0 "$TORCH_PID" 2>/dev/null; do
  wait "$TORCH_PID" && RC=0 || RC=$?
done
echo "[$(date)] torchrun exited rc=$RC"

END_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
STREAK_FILE="$REPO/store/$RUN/.noprogress_streak"   # consecutive no-progress crashes
EXCLUDE_FILE="$REPO/store/$RUN/.exclude_nodes"       # nodes that crashed with no progress
MAX_NOPROGRESS=3                                      # give up after this many consecutive no-progress crashes
NODE="${SLURMD_NODENAME:-$(hostname -s)}"; NODE="${NODE%%.*}"

if [ "$RC" -eq 0 ]; then
  echo "[$(date)] clean exit (max_steps reached / fit complete); chain terminates."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
elif [ "$END_MTIME" -gt "$START_MTIME" ]; then
  # Progress this run (time-limit SIGTERM, or a crash after a checkpoint): node is fine
  # and last.ckpt is fresh -> reset the no-progress streak/excludes, resubmit on all nodes.
  echo "[$(date)] non-zero exit AFTER checkpoint progress; resubmitting fresh (reset streak/excludes)."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
  cd "$REPO" && sbatch run_stage1_catbalanced_domaincrop_sbatch.sh
else
  # NO progress this run: a startup fault (bad GPU/ECC, NCCL, node) or a deterministic
  # bug. Exclude this node and resubmit, but cap consecutive no-progress retries so a
  # real bug can't infinite-loop the chain.
  STREAK=$(cat "$STREAK_FILE" 2>/dev/null || echo 0); STREAK=$((STREAK + 1)); echo "$STREAK" > "$STREAK_FILE"
  echo "$NODE" >> "$EXCLUDE_FILE"
  EXCLUDES=$(sort -u "$EXCLUDE_FILE" | paste -sd, -)
  if [ "$STREAK" -ge "$MAX_NOPROGRESS" ]; then
    echo "[$(date)] NO-progress crash #$STREAK on $NODE (>= $MAX_NOPROGRESS consecutive) -> likely a deterministic bug, not transient node faults. STOPPING. Crashed nodes: $EXCLUDES. Investigate, then resubmit by hand."
  else
    echo "[$(date)] NO-progress crash #$STREAK on $NODE; resubmitting with --exclude=$EXCLUDES."
    cd "$REPO" && sbatch --exclude="$EXCLUDES" run_stage1_catbalanced_domaincrop_sbatch.sh
  fi
fi
exit $RC
