#!/bin/bash
#SBATCH --job-name=cb_dc_s1_48m
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48  # 1 task (torchrun) spawns 4 GPU procs; 4 ranks x (4 dataloader workers + main) ~= 24 CPUs, 48 leaves prefetch headroom
#SBATCH --mem=400G          # in_memory=False (per-batch disk read); 4 ranks of dataloader workers peak ~50G, 400G ample on the TB-RAM preemptable nodes
#SBATCH --time=1-00:00:00   # TEMP 24h: mapo-2026 maintenance (Jun 15 05:00 - Jun 18 21:00) blocks 2-day jobs now; restore 2-00:00:00 after Jun 18
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.err

# CATH-balanced domain-crop Stage-1 (48M), combined PDB+AFDB, on mit_preemptable
# (4-GPU/user cap, PREEMPTABLE, 2-day limit). Effective batch = 4 GPU * 2 micro
# * 16 accum = 128. H100 80G fits 48M at L=320 batch 2 (~50G); L40S 44G would OOM
# at batch 2, so H100 is requested. config_name carries the training_ prefix; the
# config path resolves via the pip -e proteinfoundation __file__ at /orcd/pool,
# independent of REPO/CWD.
#
# FOUR death-mode protections (runs to max_steps=150000 unattended):
#  1. Preemption / node-failure -> SBATCH --requeue (same job id; --resume_option=allow
#     resumes from last.ckpt). The common case on mit_preemptable.
#  2. App crash (NCCL timeout, OOM, any non-zero torchrun exit) -> self-resubmit chain:
#     resubmit fresh if a checkpoint advanced this run; else exclude the node + cap
#     consecutive no-progress retries at MAX_NOPROGRESS so a deterministic bug can't loop.
#  3. Wall-clock time limit -> SIGTERM trap. --requeue does NOT cover the time limit (a
#     job that reaches --time just TIMEOUTs and is gone). SLURM sends SIGTERM at the limit,
#     so on_term resubmits fresh if near the limit, else re-raises for --requeue (preemption).
#  4. Clean finish at max_steps -> exit 0; neither chain nor trap fires.
# torchrun runs in the BACKGROUND so the trap is not deferred behind a foreground child
# (a foreground child made the previous SIGUSR1 trap miss -> the 6h chain once died clean).

set -euo pipefail
RUN=dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
REPO=/home/chenxiou/proteina  # NOT the pip -e source root; imports resolve to pip -e at /orcd/pool
TIME_LIMIT_SECONDS=86400      # keep in sync with --time=1-00:00:00 (TEMP 24h; restore to 172800 / 2-00:00:00 after the Jun 15-18 mapo-2026 maintenance)
RESUBMITTED=0
TORCH_PID=0

resubmit_fresh() {
  if [ "$RESUBMITTED" -eq 0 ]; then
    RESUBMITTED=1
    echo "[$(date)] resubmitting fresh job"
    (cd "$REPO" && sbatch run_stage1_catbalanced_domaincrop_sbatch.sh)
  fi
}
# USR1 (sent 60s before the limit) is unreliable on Engaging -> no-op. The SIGTERM
# trap (delivered at the limit AND on preemption) does the real time-limit work.
trap 'echo "[$(date)] USR1 (no-op)"' USR1
on_term() {
  echo "[$(date)] SIGTERM at SECONDS=$SECONDS; SIGTERM torchrun so it stops cleanly."
  kill -TERM "${TORCH_PID:-0}" 2>/dev/null || true
  if [ "$SECONDS" -ge "$((TIME_LIMIT_SECONDS - 300))" ]; then
    echo "[$(date)] near time limit -> resubmitting fresh and exiting 0."
    resubmit_fresh
    exit 0
  fi
  echo "[$(date)] early SIGTERM (preemption) -> re-raising for --requeue."
  trap - TERM
  kill -TERM "$$"
}
trap on_term TERM

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

# torchrun spawns the 4 GPU workers as plain python processes and sets
# RANK/WORLD_SIZE/LOCAL_RANK, which train.py reads for the manual NCCL init.
torchrun --standalone --nnodes=1 --nproc_per_node=4 proteinfoundation/train.py \
    --config_name "training_$RUN" \
    --ngpus_per_node 4 \
    --nnodes 1 \
    --batch_size 2 \
    --accumulate_grad_batches 16 \
    --resume_option allow \
    af2_ipa_weights_path=/orcd/pool/006/chenxiou/params/params_model_1_ptm.npz &
TORCH_PID=$!
# Wait for torchrun, RE-WAITING if a trap interrupts the wait, until it truly exits.
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

if [ "$RESUBMITTED" -ne 0 ]; then
  echo "[$(date)] already resubmitted by the SIGTERM trap; not resubmitting again."
elif [ "$RC" -eq 0 ]; then
  echo "[$(date)] clean exit (max_steps reached / fit complete); chain terminates."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
elif [ "$END_MTIME" -gt "$START_MTIME" ]; then
  # Progress this run (a crash after a checkpoint): node is fine and last.ckpt is fresh
  # -> reset the no-progress streak/excludes, resubmit on all nodes.
  echo "[$(date)] non-zero exit AFTER checkpoint progress; resubmitting fresh (reset streak/excludes)."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
  cd "$REPO" && sbatch run_stage1_catbalanced_domaincrop_sbatch.sh
else
  # NO progress this run: a startup fault (bad GPU/ECC, NCCL, node) or a deterministic
  # bug. Exclude this node and resubmit, capping consecutive no-progress retries.
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
