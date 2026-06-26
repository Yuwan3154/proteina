#!/bin/bash
#SBATCH --job-name=cb_dc_poller
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/monitor/poller-slurm-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/monitor/poller-slurm-%j.out
#
# DURABLE poller: runs the poll loop (script_utils/run_health_poller.sh) inside a TINY CPU job (1 CPU / 1G)
# so it SURVIVES login-node reaping -- a login setsid/nohup poller was reaped after ~2.6 days on login008.
# Partition = mit_preemptable: CPU-only so it does NOT touch the 4-GPU cap, and mit_preemptable has huge CPU
# capacity that starts promptly. (mit_normal was tried first but got priority-blocked behind ~1600 queued
# jobs -- a ~3-day wait -- on 2026-06-26.) Self-resubmits on the 48h wall-clock limit OR on preemption
# (SIGTERM trap), so it persists across the whole training run.
#   Launch:  cd ~/proteina && sbatch run_health_poller.sbatch
#   Check:   squeue -u chenxiou | grep cb_dc_poller   (RUNNING = alive)  +  cat <run>/monitor/STATUS
#   Stop:    touch <run>/monitor/STOP   (poll loop exits AND this wrapper won't resubmit), then scancel the cb_dc_poller job.
set -u
REPO=/home/chenxiou/proteina
RUN=dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
MON=$REPO/store/$RUN/monitor
SELF=run_health_poller.sbatch
mkdir -p "$MON"
RESUBMITTED=0; CHILD=0
on_term() {  # 48h wall-clock limit OR preemption/node failure -> resubmit a fresh poller job + exit (unless done/STOP)
  echo "[$(date)] SIGTERM -> resubmit poller job + exit"
  kill -TERM "${CHILD:-0}" 2>/dev/null || true
  if [ "$RESUBMITTED" -eq 0 ] && [ ! -f "$MON/STOP" ]; then RESUBMITTED=1; ( cd "$REPO" && sbatch "$SELF" ); fi
  exit 0
}
trap on_term TERM
echo "[$(date)] poller SLURM job ${SLURM_JOB_ID:-?} START on $(hostname)"
# run the poll loop in the BACKGROUND so the SIGTERM trap fires promptly (the loop's sleep would otherwise defer it)
bash "$REPO/script_utils/run_health_poller.sh" &
CHILD=$!
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD" || true; done
# reached ONLY if the poll loop exited on its own (training complete / monitor/STOP present) -> do NOT resubmit
echo "[$(date)] poll loop exited on its own (training done or STOP); poller job terminating without resubmit."
