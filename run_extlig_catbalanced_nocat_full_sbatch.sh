#!/bin/bash
#SBATCH --job-name=elft_nocat
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32             # 2 ranks x 16 workers/rank (matches hardware.ncpus_per_task_train_)
#SBATCH --mem=300G                     # in_memory=False (disk-read dataset), generous headroom
#SBATCH --time=06:00:00                # mit_normal_gpu MaxTime; non-preemptable (PreemptMode=OFF)
#SBATCH --output=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/%x-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/%x-%j.err
#SBATCH --no-requeue                   # defensive: avoid the duplicate-job class of bug (partition auto-requeue +
                                        # this script's own resubmit-on-end both firing) found earlier this project.

# ext_lig + full-nocat-coverage continuation finetune, on mit_normal_gpu (6h wall, non-preemptable, 2x L40S).
# Single-tier (L40S only -- this partition has no other GPU type to escalate to). Resubmits itself on
# TIMEOUT (6h wall) or crash-with-progress; tracks a no-progress streak and stops after MAX_NOPROGRESS
# consecutive no-progress crashes (same pattern as the 48M project's launcher). Job-name `elft_nocat` is
# a NEW, distinct SLURM safety-scope prefix for this project -- do not confuse with `cb_dc_s1_48m*`.

set -euo pipefail
RUN=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont
REPO=/home/chenxiou/proteina       # NOT the pip -e source; imports resolve to pip -e at /orcd/pool
LAUNCHER=run_extlig_catbalanced_nocat_full_sbatch.sh
STREAK_FILE="$REPO/store/$RUN/.noprogress_streak"
EXCLUDE_FILE="$REPO/store/$RUN/.exclude_nodes"
MAX_NOPROGRESS=3

resubmit() {  # anti-dup: skip if another instance of this job name is already queued (exclude self)
  local existing
  existing=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v n="elft_nocat" -v self="${SLURM_JOB_ID:-0}" '$2==n && $1!=self {print $1}' || true)
  if [ -n "$existing" ]; then
    echo "[$(date)] resubmit: elft_nocat already queued ($existing) -> skip"; return 0
  fi
  local exarg=""
  [ -s "$EXCLUDE_FILE" ] && exarg="--exclude=$(sort -u "$EXCLUDE_FILE" | paste -sd, -)"
  echo "[$(date)] resubmit: submitting fresh $LAUNCHER ${exarg}"
  (cd "$REPO" && sbatch ${exarg} --export=ALL "$LAUNCHER") || true
}

TORCH_PID=0
on_term() {  # SIGTERM = 6h wall hit -> treat as a normal chain link: resubmit + exit
  echo "[$(date)] SIGTERM (time-limit) -> resubmit + exit 0."
  kill -TERM "${TORCH_PID:-0}" 2>/dev/null || true
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"   # the job RAN (not a startup fault) -> reset no-progress state
  resubmit
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
export TORCHINDUCTOR_CACHE_DIR=/orcd/compute/so3/001/chenxi/torchinductor_cache_extlig
export TRITON_CACHE_DIR=/orcd/compute/so3/001/chenxi/triton_cache_extlig
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO"
echo "[$(date)] Launching $RUN on $(hostname) jobid=${SLURM_JOB_ID:-?} restart=${SLURM_RESTART_COUNT:-0}"

CKPT=$REPO/store/$RUN/checkpoints/last.ckpt
START_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)

RDZV_PORT=$(( 20000 + (RANDOM % 40000) ))
echo "[$(date)] torchrun rendezvous 127.0.0.1:$RDZV_PORT"
torchrun --nnodes=1 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --rdzv-id="${SLURM_JOB_ID:-$$}" proteinfoundation/train.py \
    --config_name "training_$RUN" \
    --ngpus_per_node 2 \
    --nnodes 1 \
    --batch_size 8 \
    --accumulate_grad_batches 1 \
    --resume_option allow \
    af2_ipa_weights_path=/orcd/pool/006/chenxiou/params/params_model_1_ptm.npz &
TORCH_PID=$!
RC=0
while kill -0 "$TORCH_PID" 2>/dev/null; do
  wait "$TORCH_PID" && RC=0 || RC=$?
done
echo "[$(date)] torchrun exited rc=$RC"

END_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
NODE="${SLURMD_NODENAME:-$(hostname -s)}"; NODE="${NODE%%.*}"

if [ "$RC" -eq 0 ]; then
  echo "[$(date)] clean finish (max_steps) -> done, no resubmit."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
elif [ "$END_MTIME" -gt "$START_MTIME" ]; then
  echo "[$(date)] non-zero exit AFTER checkpoint progress -> resubmit (reset streak + excludes)."
  rm -f "$STREAK_FILE" "$EXCLUDE_FILE"
  resubmit
else
  STREAK=$(cat "$STREAK_FILE" 2>/dev/null || echo 0); STREAK=$((STREAK + 1)); echo "$STREAK" > "$STREAK_FILE"
  echo "$NODE" >> "$EXCLUDE_FILE"
  if [ "$STREAK" -ge "$MAX_NOPROGRESS" ]; then
    echo "[$(date)] NO-progress crash #$STREAK (>= $MAX_NOPROGRESS) on $NODE -> likely persistent. STOPPING. Bad nodes: $(sort -u "$EXCLUDE_FILE" | paste -sd, -). Investigate, then resubmit by hand."
  else
    echo "[$(date)] NO-progress crash #$STREAK on $NODE -> resubmit (excluding crashed nodes)."
    resubmit
  fi
fi
exit $RC
