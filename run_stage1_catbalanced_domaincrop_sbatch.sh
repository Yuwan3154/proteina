#!/bin/bash
#SBATCH --job-name=cb_dc_s1_48m_h200   # DEFAULT = H200 tier; per-tier submits OVERRIDE --job-name/--gres on the CLI
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4              # DEFAULT = H200; overridden per-tier on the sbatch CLI
#SBATCH --cpus-per-task=20             # 4 ranks x 4 workers + 4 main; lean (training reads pre-processed .pt)
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00              # 2-day MAX (mit_preemptable cap=48h)
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.err
# NO --requeue: a preemption is treated as a round-end -> RE-ESCALATE (per design), not requeue.

# PER-ROUND GPU-escalation Stage-1 (48M), combined PDB+AFDB, on mit_preemptable.
# The escalating queue is rebuilt EACH ROUND (a round begins when the previous job ends):
#   - On job END (preemption / time-limit / crash-with-progress): ESCALATE = submit a fresh set
#       {H200 begin=now, H100 begin=+8h, L40S begin=+24h}. SLURM --begin does the timed broadening
#       (try H200 hardest; add H100 after 8h unassigned; add L40S after 24h) with NO daemon.
#   - On job START (this job got a node): PURGE all OTHER pending cb_dc_s1_48m* jobs (the escalation
#       losers / any duplicate) -> "we got one, drop the rest."
#   - On CLEAN finish (max_steps): cancel ALL tiers; done.
# The per-user QOS cap gres/gpu=4 TOTAL (verified, untyped) is a free MUTEX -> at most ONE tier ever
# RUNS -> no checkpoint contention even during the brief escalation window. Effective batch always 128
# (4 GPU x batch x accum): H200 4x8, H100 2x16, L40S 1x32; batch/accum are picked from the ACTUAL GPU
# at runtime (OOM-safe). Tier is read from the SLURM job name (cb_dc_s1_48m_<tier>). A no-progress streak
# (cap MAX_NOPROGRESS) stops a deterministic crash loop. torchrun runs in the BACKGROUND so the SIGTERM
# trap is not deferred behind a foreground child.

set -euo pipefail
RUN=dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
REPO=/home/chenxiou/proteina       # NOT the pip -e source; imports resolve to pip -e at /orcd/pool
LAUNCHER=run_stage1_catbalanced_domaincrop_sbatch.sh
TIME_LIMIT_SECONDS=172800          # keep in sync with --time=2-00:00:00
STREAK_FILE="$REPO/store/$RUN/.noprogress_streak"
EXCLUDE_FILE="$REPO/store/$RUN/.exclude_nodes"  # nodes that crashed this job with NO progress (bad GPU / CUDA invalid device ordinal); escalate AVOIDS them. Cleared on any real progress.
MAX_NOPROGRESS=3

# --- tier from the SLURM-set job name (cb_dc_s1_48m_<tier>) -> gres for resubmit targeting ---
JOBNAME="${SLURM_JOB_NAME:-cb_dc_s1_48m_h200}"
TIER="${JOBNAME##*_}"
case "$TIER" in
  h200) GRES=gpu:h200:4 ;;
  h100) GRES=gpu:h100:4 ;;
  l40s) GRES=gpu:l40s:4 ;;
  *) echo "[$(date)] WARN: job-name '$JOBNAME' has no known tier suffix -> default h200"; TIER=h200; JOBNAME=cb_dc_s1_48m_h200; GRES=gpu:h200:4 ;;
esac

# submit ONE tier with a begin offset, anti-duplicate (skip if a job of that tier is already queued)
_submit_tier() {  # $1=tier  $2=gres  $3=begin ; anti-dup: skip if ANOTHER job of this tier is queued (exclude self, the ending job)
  local existing
  existing=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v n="cb_dc_s1_48m_$1" -v self="${SLURM_JOB_ID:-0}" '$2==n && $1!=self {print $1}' || true)
  if [ -n "$existing" ]; then
    echo "[$(date)] escalate: tier $1 already queued ($existing) -> skip"; return 0
  fi
  local exarg=""
  [ -s "$EXCLUDE_FILE" ] && exarg="--exclude=$(sort -u "$EXCLUDE_FILE" | paste -sd, -)"
  echo "[$(date)] escalate: submit tier $1 (gres $2 begin $3) ${exarg}"
  (cd "$REPO" && sbatch --job-name="cb_dc_s1_48m_$1" --gres="$2" --begin="$3" ${exarg} --export=ALL "$LAUNCHER") || true
}
# ESCALATE = start a fresh round: H200 now, H100 +8h, L40S +24h (--begin gates the broadening)
escalate() {
  _submit_tier h200 gpu:h200:4 now
  _submit_tier h100 gpu:h100:4 now+8hours
  _submit_tier l40s gpu:l40s:4 now+24hours
}

ESCALATED=0
TORCH_PID=0
on_term() {  # SIGTERM = preemption OR time-limit -> treat BOTH as a round-end: escalate + exit
  echo "[$(date)] SIGTERM (preemption or time-limit) -> round-end: escalate fresh set + exit 0."
  kill -TERM "${TORCH_PID:-0}" 2>/dev/null || true
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"   # the job RAN (preemption/time-limit, not a startup fault) -> reset no-progress state
  if [ "$ESCALATED" -eq 0 ]; then ESCALATED=1; escalate; fi
  exit 0
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
# batch/accum from the ACTUAL allocated GPU (OOM-safe; eff batch always 4*batch*accum = 128)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
case "$GPU_NAME" in
  *H200*) BATCH=4; ACCUM=8  ;;
  *H100*) BATCH=2; ACCUM=16 ;;
  *L40S*|*L40*) BATCH=1; ACCUM=32 ;;
  *) echo "[$(date)] WARN: unrecognized GPU '$GPU_NAME' -> safe default batch 1/accum 32"; BATCH=1; ACCUM=32 ;;
esac
echo "[$(date)] Launching $RUN TIER=$TIER GPU='$GPU_NAME' -> batch $BATCH accum $ACCUM (eff 128) on $(hostname) jobid=${SLURM_JOB_ID:-?} restart=${SLURM_RESTART_COUNT:-0}"

# START-HOOK (we got a node = end of this round's escalation): purge ALL OTHER pending cb_dc_s1_48m* jobs
# (the escalation losers + any duplicate). -t PENDING so a running job (incl. self) is never touched.
purge_pending_others=$(squeue -u "$USER" -h -t PENDING -o "%i %j" 2>/dev/null | awk -v self="${SLURM_JOB_ID:-0}" '$2 ~ /^cb_dc_s1_48m/ && $1 != self {print $1}' || true)
if [ -n "$purge_pending_others" ]; then
  echo "[$(date)] assigned a node -> purge other pending tiers: $(echo $purge_pending_others | tr '\n' ' ')"
  scancel $purge_pending_others 2>/dev/null || true
fi

CKPT=$REPO/store/$RUN/checkpoints/last.ckpt
START_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)

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

if [ "$ESCALATED" -ne 0 ]; then
  echo "[$(date)] already escalated by the SIGTERM trap."
elif [ "$RC" -eq 0 ]; then
  echo "[$(date)] clean finish (max_steps) on tier $TIER -> cancel ALL pending tiers; chain done."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
  done_ids=$(squeue -u "$USER" -h -t PENDING -o "%i %j" 2>/dev/null | awk '$2 ~ /^cb_dc_s1_48m/ {print $1}' || true)
  [ -n "$done_ids" ] && scancel $done_ids 2>/dev/null || true
elif [ "$END_MTIME" -gt "$START_MTIME" ]; then
  echo "[$(date)] non-zero exit AFTER checkpoint progress -> escalate (reset streak + excludes)."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
  escalate
else
  STREAK=$(cat "$STREAK_FILE" 2>/dev/null || echo 0); STREAK=$((STREAK + 1)); echo "$STREAK" > "$STREAK_FILE"
  echo "$NODE" >> "$EXCLUDE_FILE"   # startup fault, NO progress -> AVOID this node on the next escalate (bad GPU / CUDA invalid device ordinal, e.g. node3101)
  if [ "$STREAK" -ge "$MAX_NOPROGRESS" ]; then
    echo "[$(date)] NO-progress crash #$STREAK (>= $MAX_NOPROGRESS) on $NODE -> likely persistent. STOPPING. Bad nodes: $(sort -u "$EXCLUDE_FILE" | paste -sd, -). Investigate, then resubmit by hand (bash submit_tiered.sh)."
  else
    echo "[$(date)] NO-progress crash #$STREAK on $NODE -> escalate (excluding crashed nodes)."
    escalate
  fi
fi
exit $RC
