#!/bin/bash
#SBATCH --job-name=elft_watchdog_r2
#SBATCH --partition=download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/gridsan/cou/proteina/store/elft_watchdog_r2-%j.out
#SBATCH --error=/home/gridsan/cou/proteina/store/elft_watchdog_r2-%j.err
#
# Watchdog for ext_lig SuperCloud round 2 (elft_sc_r2) -- same proven pattern as round 1's
# watchdog (run_elft_watchdog.sbatch), scoped to round 2's job name/run/max_steps, WITH the
# sacct-based completion fix already applied (round 1's watchdog originally used only the last
# logged "step=" line, which lagged the true final step and caused an unwanted redundant
# resubmit right at round 1's real finish -- fixed there and built correctly here from the
# start). Runs as a tiny SLURM job on 'download' (separate node-cap pool from xeon-g6-volta's
# node=4 limit -- can never contend with the 4-node production job). Self-resubmits on
# SIGTERM (wall-limit/preemption); monitor/STOP is the manual kill-switch.
set -uo pipefail
REPO=/home/gridsan/cou/proteina
RUN=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_21-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_combined_3to1nocat_supercloud
MON="$REPO/store/$RUN/monitor"
STOP="$MON/STOP"
LOG="$MON/WATCHDOG_LOG"
JOBNAME=elft_sc_r2
MAX_STEPS=642000
INTERVAL=600
SELF=run_elft_watchdog_r2.sbatch
mkdir -p "$MON"

RESUBMITTED=0
on_term() {
  echo "[$(date)] SIGTERM -> resubmit watchdog job + exit" >> "$LOG"
  if [ "$RESUBMITTED" -eq 0 ] && [ ! -f "$STOP" ]; then
    RESUBMITTED=1
    (cd "$REPO" && sbatch "$SELF")
  fi
  exit 0
}
trap on_term TERM

echo "[$(date)] watchdog SLURM job ${SLURM_JOB_ID:-?} START on $(hostname)" >> "$LOG"
last_seen_jobid=""
while true; do
  if [ -f "$STOP" ]; then
    echo "[$(date)] STOP present -> exiting" >> "$LOG"
    exit 0
  fi
  cur_jobid=$(squeue -u cou -n "$JOBNAME" -h -o '%i' 2>/dev/null | head -1)
  latest_out=$(ls -t "$REPO"/store/"$JOBNAME"-*.out 2>/dev/null | head -1)
  step=$(grep -oE 'step=[0-9]+' "$latest_out" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo 0)
  step=${step:-0}
  if [ -z "$cur_jobid" ]; then
    sacct_state=""
    if [ -n "$last_seen_jobid" ]; then
      sacct_state=$(sacct -j "$last_seen_jobid" -o State -X -n 2>/dev/null | tr -d ' ')
    fi
    if [ "$sacct_state" = "COMPLETED" ]; then
      echo "[$(date)] job $last_seen_jobid COMPLETED (sacct-confirmed), last logged step=$step -> training done, no resubmit" >> "$LOG"
      exit 0
    else
      echo "[$(date)] job absent (last=$last_seen_jobid, sacct State=$sacct_state), step=$step < $MAX_STEPS -> RESUBMITTING" >> "$LOG"
      (cd "$REPO" && sbatch run_extlig_3to1nocat_supercloud_full.sbatch) >> "$LOG" 2>&1
    fi
  else
    last_seen_jobid="$cur_jobid"
    echo "[$(date)] job present (id=$cur_jobid), step=$step, OK" >> "$LOG"
  fi
  sleep "$INTERVAL" &
  wait $!
done
