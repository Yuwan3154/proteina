#!/bin/bash
# Detached on-box health poller for the per-round 48M stage-1 run on Engaging.
# DETECTION ONLY: writes STATUS + ALERTS files. No cron, no scancel, no sbatch/resubmit.
# Survives SSH/ControlMaster drops and terminal close (launch under setsid+nohup).
#
# Launch (from a login node):
#   ssh -n Engaging 'cd ~/proteina && setsid nohup bash script_utils/run_health_poller.sh \
#       >/dev/null 2>&1 </dev/null & echo started'
#
# Every INTERVAL it inspects squeue (any cb_dc_s1_48m* tier) + last.ckpt + the newest slurm logs and writes
#   $RUNDIR/monitor/STATUS    (latest one-liner, overwritten)
#   $RUNDIR/monitor/poller.log (full history, append)
#   $RUNDIR/monitor/ALERTS    (append-only; ONLY chain-down / stall / OOM-or-CUDA-error / stuck-pending)
# Exits cleanly when step >= MAX_STEPS. Stop it any time: touch $RUNDIR/monitor/STOP (or kill the pid).
set -u
RUN="${1:-dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined}"
JOBPREFIX="${2:-cb_dc_s1_48m}"      # matches cb_dc_s1_48m and cb_dc_s1_48m_{h200,h100,l40s}
INTERVAL="${POLL_INTERVAL:-600}"    # 10 min between polls (a full stop surfaces in ~20 min)
STALL_SECS="${STALL_SECS:-5400}"    # RUNNING + last.ckpt this stale => stall (covers compile + val)
PENDING_POLLS="${PENDING_POLLS:-6}" # PENDING this many polls (~1h) => alert (pending is NORMAL in per-round escalation)
MAX_STEPS="${MAX_STEPS:-150000}"
RUNDIR="/home/chenxiou/proteina/store/$RUN"
MON="$RUNDIR/monitor"; CKPT="$RUNDIR/checkpoints/last.ckpt"
mkdir -p "$MON"
echo $$ > "$MON/poller.pid"
nojob=0; pending=0
echo "[$(date '+%F %T')] poller START pid=$$ run=$RUN prefix=$JOBPREFIX interval=${INTERVAL}s (detection-only)" >> "$MON/poller.log"
while true; do
  [ -f "$MON/STOP" ] && { echo "[$(date '+%F %T')] STOP file present; poller exiting." >> "$MON/poller.log"; rm -f "$MON/STOP"; break; }
  ts=$(date '+%F %T'); now=$(date +%s)
  # any of my tier jobs (prefer a RUNNING one for the state line)
  line=$(squeue -u chenxiou -h -o '%i %j %T %M %R' 2>/dev/null | grep -F "$JOBPREFIX" | grep -m1 RUNNING || squeue -u chenxiou -h -o '%i %j %T %M %R' 2>/dev/null | grep -F "$JOBPREFIX" | head -1)
  jid=$(awk '{print $1}' <<<"$line"); state=$(awk '{print $3}' <<<"$line"); rtime=$(awk '{print $4}' <<<"$line")
  ckmt=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0); age=$(( now - ${ckmt:-0} ))
  cknice=$(date -d "@${ckmt:-0}" '+%m-%d_%H:%M' 2>/dev/null || echo none)
  # latest training step + error markers from the newest 3 slurm .err logs (diag step= lives in stderr)
  errs=$(ls -t "$RUNDIR"/slurm/*.err 2>/dev/null | head -3)
  step=0; oom=0
  if [ -n "$errs" ]; then
    s=$(grep -hoE 'step=[0-9]+' $errs 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1); step="${s:-0}"
    oom=$(grep -icE 'out of memory|cuda error|invalid device ordinal|Traceback' $errs 2>/dev/null | paste -sd+ - | bc 2>/dev/null); oom="${oom:-0}"
  fi
  flag=OK; alert=""
  if [ -z "$jid" ]; then
    nojob=$((nojob+1)); pending=0
    if [ "$step" -ge "$MAX_STEPS" ]; then flag=DONE
    elif [ "$nojob" -ge 2 ]; then flag=ALERT; alert="CHAIN-DOWN: NO cb_dc_s1_48m* job for ${nojob} polls and step=$step < $MAX_STEPS -> training STOPPED (resubmit: bash submit_tiered.sh)"
    else flag="transient-no-job(${nojob})"; fi
  else
    nojob=0
    if [ "$state" = RUNNING ]; then
      pending=0
      [ "$age" -gt "$STALL_SECS" ] && { flag=ALERT; alert="STALL: $jid RUNNING (rt=$rtime) but last.ckpt ${age}s stale, step=$step"; }
    else
      pending=$((pending+1))
      [ "$pending" -ge "$PENDING_POLLS" ] && { flag=ALERT; alert="STUCK-PENDING: $jid state=$state for ${pending} polls (~$((pending*INTERVAL/60))min, no GPU); step=$step < $MAX_STEPS"; }
    fi
  fi
  [ "${oom:-0}" -gt 0 ] 2>/dev/null && { flag=ALERT; alert="OOM/CUDA-error markers in newest slurm .err (count=$oom)"; }
  status="[$ts] job=${jid:-NONE} state=${state:-NONE} step=$step/$MAX_STEPS ckpt=${cknice}(${age}s) oom=$oom -> $flag"
  echo "$status" > "$MON/STATUS"; echo "$status" >> "$MON/poller.log"
  [ -n "$alert" ] && echo "[$ts] $alert" >> "$MON/ALERTS"
  [ "$flag" = DONE ] && { echo "[$ts] run complete (step>=$MAX_STEPS); poller exiting." >> "$MON/poller.log"; break; }
  sleep "$INTERVAL"
done
