#!/bin/bash
#SBATCH --job-name=cb_dc_s1_48m_h200   # DEFAULT = H200 tier; the per-tier submits OVERRIDE --job-name/--gres on the CLI
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4              # DEFAULT = H200; overridden per-tier (h100:4 / l40s:4) on the sbatch CLI
#SBATCH --cpus-per-task=20             # 4 ranks x 4 workers + 4 main; lean (training reads pre-processed .pt, compute-bound)
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00              # 2-day MAX (mit_preemptable cap=48h): maximize walltime -> minimize resubmits/lost progress
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.err

# TIERED GPU-escalation Stage-1 (48M), combined PDB+AFDB, on mit_preemptable.
# THREE tier-jobs (H200/H100/L40S) are queued, each requesting 4 GPU. The per-USER
# QOS cap is gres/gpu=4 TOTAL (untyped, verified) -> at most ONE runs at a time ->
# NO checkpoint contention (the cap is a free mutex). Escalation is SLURM-native via
# --begin (H200 now, H100 +8h, L40S +24h): later tiers sit in BeginTime until their
# time, then compete; whichever lands first trains from last.ckpt. Effective batch is
# always 128 (4 GPU x batch x accum): H200 4x8, H100 2x16, L40S 1x32. Changing
# batch/accum across a resume is safe (ckpt stores weights/opt/step/epoch, not batch).
#
# TIER is read from the env (--export=ALL,TIER=...). DEFAULT h200 so a bare
# `sbatch run_..._sbatch.sh` is the H200 tier (lets the old single-tier chain adopt
# this launcher on its next resubmit with no edit).
#
# Death-mode protections (per tier; runs to max_steps=150000 unattended):
#  1. Preemption/node-failure -> --requeue (same job id, resumes last.ckpt). Common on mit_preemptable.
#  2. App crash -> self-resubmit chain: resubmit THIS tier if a ckpt advanced; else exclude the node,
#     cap consecutive no-progress retries at MAX_NOPROGRESS (deterministic-bug guard). Per-tier files.
#  3. Wall-clock limit -> SIGTERM trap resubmits THIS tier (--requeue does NOT cover the time limit).
#  4. Clean finish at max_steps -> exit 0 + CANCEL the other two tiers (training done; no orphans).
# Resubmits are anti-duplicate-guarded (skip if a job of this tier is already queued) and use
# --begin=now (escalation offsets apply only to the initial submit). torchrun runs in the BACKGROUND
# so the SIGTERM trap is not deferred behind a foreground child.

set -euo pipefail
RUN=dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
REPO=/home/chenxiou/proteina       # NOT the pip -e source; imports resolve to pip -e at /orcd/pool
LAUNCHER=run_stage1_catbalanced_domaincrop_sbatch.sh
TIME_LIMIT_SECONDS=172800          # keep in sync with --time=2-00:00:00

# --- TIER -> GPU type + per-GPU batch/accum (eff batch = 4 GPU * batch * accum = 128) ---
TIER="${TIER:-h200}"
case "$TIER" in
  h200) GRES=gpu:h200:4; BATCH=4; ACCUM=8  ;;  # 141 GB: batch 4 ~90 GB
  h100) GRES=gpu:h100:4; BATCH=2; ACCUM=16 ;;  # 80 GB:  batch 2 ~70 GB
  l40s) GRES=gpu:l40s:4; BATCH=1; ACCUM=32 ;;  # 44 GB:  batch 1 ~32 GB
  *) echo "[$(date)] FATAL: unknown TIER=$TIER (expected h200|h100|l40s)"; exit 1 ;;
esac
JOBNAME="cb_dc_s1_48m_${TIER}"
ALL_TIERS="h200 h100 l40s"
STREAK_FILE="$REPO/store/$RUN/.noprogress_streak_$TIER"  # per-tier (a tier's bad node doesn't exclude another tier)
EXCLUDE_FILE="$REPO/store/$RUN/.exclude_nodes_$TIER"
MAX_NOPROGRESS=3
RESUBMITTED=0
TORCH_PID=0

# Resubmit THIS tier (begin=now), ONLY if no other job of this tier is already queued (anti-duplicate).
resubmit_tier() {
  local extra="${1:-}"
  local existing
  existing=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v n="$JOBNAME" -v self="${SLURM_JOB_ID:-0}" '$2==n && $1!=self {print $1}')
  if [ -n "$existing" ]; then
    echo "[$(date)] anti-dup: a $JOBNAME job is already queued ($existing); NOT resubmitting tier $TIER."
    return
  fi
  echo "[$(date)] resubmitting tier $TIER (begin=now) ${extra}"
  (cd "$REPO" && sbatch --job-name="$JOBNAME" --gres="$GRES" --begin=now --export=ALL,TIER="$TIER" ${extra} "$LAUNCHER")
}

trap 'echo "[$(date)] USR1 (no-op; SIGTERM does the time-limit work)"' USR1
on_term() {
  echo "[$(date)] SIGTERM at SECONDS=$SECONDS; SIGTERM torchrun so it stops cleanly."
  kill -TERM "${TORCH_PID:-0}" 2>/dev/null || true
  if [ "$SECONDS" -ge "$((TIME_LIMIT_SECONDS - 300))" ]; then
    echo "[$(date)] near time limit -> resubmit tier $TIER and exit 0."
    if [ "$RESUBMITTED" -eq 0 ]; then RESUBMITTED=1; resubmit_tier; fi
    exit 0
  fi
  echo "[$(date)] early SIGTERM (preemption) -> re-raise for --requeue."
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
export DIAG_DATALOADER=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/compute/so3/001/chenxi/torchinductor_cache
export TRITON_CACHE_DIR=/orcd/compute/so3/001/chenxi/triton_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO"
echo "[$(date)] Launching $RUN TIER=$TIER (batch $BATCH accum $ACCUM, eff 128) on $(hostname) GPUs $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) jobid=${SLURM_JOB_ID:-?} restart=${SLURM_RESTART_COUNT:-0}"

CKPT=$REPO/store/$RUN/checkpoints/last.ckpt
START_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)

# Random rendezvous port (avoid torchrun --standalone fixed-port collision with co-tenants on shared nodes).
RDZV_PORT=$(( 20000 + (RANDOM % 40000) ))
echo "[$(date)] torchrun rendezvous 127.0.0.1:$RDZV_PORT"
torchrun --nnodes=1 --nproc_per_node=4 --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --rdzv-id="${SLURM_JOB_ID:-$$}" proteinfoundation/train.py \
    --config_name "training_$RUN" \
    --ngpus_per_node 4 \
    --nnodes 1 \
    --batch_size "$BATCH" \
    --accumulate_grad_batches "$ACCUM" \
    --resume_option allow \
    af2_ipa_weights_path=/orcd/pool/006/chenxiou/params/params_model_1_ptm.npz &
TORCH_PID=$!
RC=0
while kill -0 "$TORCH_PID" 2>/dev/null; do
  wait "$TORCH_PID" && RC=0 || RC=$?
done
echo "[$(date)] torchrun exited rc=$RC (tier $TIER)"

END_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
NODE="${SLURMD_NODENAME:-$(hostname -s)}"; NODE="${NODE%%.*}"

if [ "$RESUBMITTED" -ne 0 ]; then
  echo "[$(date)] already resubmitted by the SIGTERM trap; not resubmitting again."
elif [ "$RC" -eq 0 ]; then
  echo "[$(date)] clean exit (max_steps / fit complete) on tier $TIER; CANCEL the other tiers + terminate."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
  for t in $ALL_TIERS; do
    [ "$t" = "$TIER" ] && continue
    scancel -u "$USER" --partition=mit_preemptable --name="cb_dc_s1_48m_$t" 2>/dev/null && echo "[$(date)] cancelled tier $t (training complete)"
  done
elif [ "$END_MTIME" -gt "$START_MTIME" ]; then
  echo "[$(date)] non-zero exit AFTER checkpoint progress; resubmit tier $TIER (reset streak/excludes)."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
  resubmit_tier
else
  STREAK=$(cat "$STREAK_FILE" 2>/dev/null || echo 0); STREAK=$((STREAK + 1)); echo "$STREAK" > "$STREAK_FILE"
  echo "$NODE" >> "$EXCLUDE_FILE"
  EXCLUDES=$(sort -u "$EXCLUDE_FILE" | paste -sd, -)
  if [ "$STREAK" -ge "$MAX_NOPROGRESS" ]; then
    echo "[$(date)] NO-progress crash #$STREAK on $NODE (>= $MAX_NOPROGRESS) for tier $TIER -> STOPPING this tier (other tiers continue). Crashed nodes: $EXCLUDES."
  else
    echo "[$(date)] NO-progress crash #$STREAK on $NODE; resubmit tier $TIER with --exclude=$EXCLUDES."
    resubmit_tier "--exclude=$EXCLUDES"
  fi
fi
exit $RC
