#!/bin/bash
# Detached on-box health poller for a stage-1 autoresume run on Engaging.
# Survives SSH/ControlMaster drops and terminal close (launch under setsid+nohup).
#
# Launch (from a login node):
#   ssh -n Engaging 'cd ~/proteina && setsid nohup bash script_utils/run_health_poller.sh \
#       >/dev/null 2>&1 </dev/null & echo started'
#
# Polls squeue + last.ckpt + the newest slurm log every POLL_INTERVAL and writes under
# $RUNDIR/monitor/:  STATUS (latest one-liner, overwritten),  poller.log (full history),
# ALERTS (append-only; only stall / chain-down / OOM). Exits cleanly when step >= MAX_STEPS.
set -u
RUN="${1:-dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined}"
JOBNAME="${2:-cb_dc_s1_48m}"
INTERVAL="${POLL_INTERVAL:-1800}"   # 30 min between polls
STALL_SECS="${STALL_SECS:-5400}"    # RUNNING + last.ckpt this stale => stall (covers compile+short pend)
MAX_STEPS="${MAX_STEPS:-150000}"
RUNDIR="/home/chenxiou/proteina/store/$RUN"
MON="$RUNDIR/monitor"
CKPT="$RUNDIR/checkpoints/last.ckpt"
mkdir -p "$MON"
nojob=0
echo "[$(date '+%F %T')] poller START pid=$$ run=$RUN job=$JOBNAME interval=${INTERVAL}s" >> "$MON/poller.log"
while true; do
  ts=$(date '+%F %T'); now=$(date +%s)
  line=$(squeue -u chenxiou -h -o '%i %j %T %M %R' 2>/dev/null | grep -F "$JOBNAME" | head -1)
  jid=$(awk '{print $1}' <<<"$line"); state=$(awk '{print $3}' <<<"$line"); rtime=$(awk '{print $4}' <<<"$line")
  ckmt=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0); age=$(( now - ${ckmt:-0} ))
  cknice=$(date -d "@${ckmt:-0}" '+%m-%d_%H:%M' 2>/dev/null || echo none)
  log=$(ls -t "$RUNDIR"/slurm/*.out 2>/dev/null | head -1)
  step=0; oom=0
  if [ -n "$log" ]; then
    s=$(grep -oE 'step=[0-9]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+'); step="${s:-0}"
    oom=$(grep -icE 'out of memory|outofmemory|cuda error|traceback' "$log" 2>/dev/null | head -1); oom="${oom:-0}"
  fi
  flag=OK; alert=""
  if [ -z "$jid" ]; then
    nojob=$((nojob+1))
    if [ "$step" -ge "$MAX_STEPS" ]; then flag=DONE
    elif [ "$nojob" -ge 2 ]; then flag=ALERT; alert="CHAIN-DOWN: no '$JOBNAME' job for ${nojob} polls and step=$step < $MAX_STEPS (autoresume chain may have stopped)"
    else flag="transient-no-job(${nojob})"; fi
  else
    nojob=0
    if [ "$state" = RUNNING ] && [ "$age" -gt "$STALL_SECS" ]; then flag=ALERT; alert="STALL: $jid RUNNING (rt=$rtime) but last.ckpt ${age}s stale, step=$step"; fi
  fi
  if [ "$oom" -gt 0 ] 2>/dev/null; then flag=ALERT; alert="OOM/ERROR markers in $(basename "$log")"; fi
  status="[$ts] job=${jid:-none} state=${state:-NONE} step=$step/$MAX_STEPS ckpt=${cknice}(${age}s) oom=$oom -> $flag"
  echo "$status" > "$MON/STATUS"; echo "$status" >> "$MON/poller.log"
  [ -n "$alert" ] && echo "[$ts] $alert" >> "$MON/ALERTS"
  [ "$flag" = DONE ] && { echo "[$ts] run complete (step>=$MAX_STEPS); poller exiting." >> "$MON/poller.log"; break; }
  sleep "$INTERVAL"
done
