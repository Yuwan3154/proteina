#!/bin/bash
#SBATCH --job-name=cb_dc_ckpt_mirror
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue                   # mit_preemptable auto-requeues preempted jobs by default; without this flag
                                        # SLURM's auto-requeue AND our own on_term trap both resubmit -> duplicate job
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/ckpt-mirror-%j.out
# Mirrors last.ckpt + last-EMA.ckpt from scratch -> pool hourly.
# Self-resubmits on SIGTERM (48h limit or preemption). Stops when training reaches max_steps.

set -uo pipefail
SCR=/orcd/scratch/orcd/011/chenxiou
POOL=/orcd/pool/006/chenxiou
RUN=dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
REPO=/home/chenxiou/proteina  # symlink -> scratch
INTERVAL=3600
RESUBMITTED=0
MAX_STEPS=150000

on_term() {
    # Check if training is done before resubmitting
    step=$(grep -oP 'step=\K[0-9]+' "$REPO/store/$RUN/monitor/STATUS" 2>/dev/null | head -1 || echo 0)
    if [ "$step" -ge "$MAX_STEPS" ]; then
        echo "[$(date)] training at max_steps=$step -> mirror done, no resubmit"
    elif [ "$RESUBMITTED" -eq 0 ]; then
        RESUBMITTED=1
        echo "[$(date)] SIGTERM -> resubmit mirror"
        (cd "$REPO" && sbatch run_ckpt_mirror.sbatch) || true
    fi
    exit 0
}
trap on_term TERM

mirror_once() {
    local src="$SCR/proteina/store/$RUN/checkpoints"
    local dst="$POOL/proteina/store/$RUN/checkpoints"
    rsync -a --quiet "$src/last.ckpt" "$dst/last.ckpt" 2>/dev/null && \
        echo "[$(date)] mirrored last.ckpt"
    rsync -a --quiet "$src/last-EMA.ckpt" "$dst/last-EMA.ckpt" 2>/dev/null && \
        echo "[$(date)] mirrored last-EMA.ckpt"
}

echo "[$(date)] ckpt-mirror START on $(hostname) jobid=${SLURM_JOB_ID:-?}"
while true; do
    sleep $INTERVAL
    mirror_once
done
