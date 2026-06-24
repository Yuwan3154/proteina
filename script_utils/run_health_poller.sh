#!/bin/bash
# Detached on-box health poller for the per-round 48M stage-1 run on Engaging.
# Writes STATUS + ALERTS AND auto-resubmits the chain on a GENUINE chain-down (bash submit_tiered.sh).
# sbatch ONLY -- NEVER scancel. Defers to the launcher's no-progress cap: if the launcher DELIBERATELY
# stopped (.noprogress_streak >= MAX_NOPROGRESS = deterministic bug / 3 distinct bad nodes), it does NOT
# resubmit, only ALERTs (human needed). No agent cron.
# This is the POLL LOOP. The DURABLE way to run it is the SLURM wrapper run_health_poller.sbatch
# (a tiny mit_normal CPU job that self-resubmits at its 12h limit) -- a login setsid/nohup process gets
# REAPED after a few days on the shared login node. Durable launch:  cd ~/proteina && sbatch run_health_poller.sbatch
# Quick/temporary login launch (NOT durable):
#   ssh -n Engaging 'cd ~/proteina && setsid nohup bash script_utils/run_health_poller.sh >/dev/null 2>&1 </dev/null & echo started'
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
REPO="${REPO:-/home/chenxiou/proteina}"
MAX_NOPROGRESS="${MAX_NOPROGRESS:-3}"   # = launcher cap; if .noprogress_streak >= this, the stop was DELIBERATE -> do NOT resubmit
MAX_PRESUB="${MAX_PRESUB:-3}"           # cap consecutive poller auto-resubmits that fail to produce a lasting job -> then give up + alert
RUNDIR="/home/chenxiou/proteina/store/$RUN"
MON="$RUNDIR/monitor"; CKPT="$RUNDIR/checkpoints/last.ckpt"
mkdir -p "$MON"
echo $$ > "$MON/poller.pid"
nojob=0; pending=0; presub=0
echo "[$(date '+%F %T')] poller START pid=$$ run=$RUN prefix=$JOBPREFIX interval=${INTERVAL}s (detection-only)" >> "$MON/poller.log"
rt2s() {  # slurm elapsed D-HH:MM:SS / HH:MM:SS / MM:SS -> seconds
  local t="$1" d=0
  case "$t" in *-*) d="${t%%-*}"; t="${t#*-}";; esac
  local IFS=:; set -- $t
  case $# in
    3) echo $(( d*86400 + 10#$1*3600 + 10#$2*60 + 10#$3 )) ;;
    2) echo $(( d*86400 + 10#$1*60 + 10#$2 )) ;;
    *) echo $(( d*86400 + 10#${1:-0} )) ;;
  esac
}
while true; do
  [ -f "$MON/STOP" ] && { echo "[$(date '+%F %T')] STOP file present; poller exiting." >> "$MON/poller.log"; rm -f "$MON/STOP"; break; }
  ts=$(date '+%F %T'); now=$(date +%s)
  # any of my tier jobs (prefer a RUNNING one for the state line)
  line=$(squeue -u chenxiou -h -o '%i %j %T %M %R' 2>/dev/null | grep -F "$JOBPREFIX" | grep -m1 RUNNING || squeue -u chenxiou -h -o '%i %j %T %M %R' 2>/dev/null | grep -F "$JOBPREFIX" | head -1)
  jid=$(awk '{print $1}' <<<"$line"); state=$(awk '{print $3}' <<<"$line"); rtime=$(awk '{print $4}' <<<"$line")
  ckmt=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0); age=$(( now - ${ckmt:-0} ))
  cknice=$(date -d "@${ckmt:-0}" '+%m-%d_%H:%M' 2>/dev/null || echo none)
  # step + GPU-error markers from the CURRENT job's .err ONLY. (Grepping the newest-3 caused FALSE oom alerts:
  # old logs legitimately hold a timed-out job's SIGTERM traceback + historical node3101 CUDA errors.)
  # No running/pending job -> step from the newest log (for the done-check) and oom=0 (ignore historical errors).
  if [ -n "$jid" ]; then cerr=$(ls -t "$RUNDIR"/slurm/*-"$jid".err 2>/dev/null | head -1)
  else cerr=$(ls -t "$RUNDIR"/slurm/*.err 2>/dev/null | head -1); fi
  step=0; oom=0
  if [ -n "$cerr" ]; then
    s=$(grep -hoE 'step=[0-9]+' "$cerr" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1); step="${s:-0}"
    [ -n "$jid" ] && oom=$(grep -ihE 'out of memory|cuda error|invalid device ordinal' "$cerr" 2>/dev/null | wc -l)
    oom="${oom:-0}"
  fi
  flag=OK; alert=""
  if [ -z "$jid" ]; then
    nojob=$((nojob+1)); pending=0
    streak=$(cat "$RUNDIR/.noprogress_streak" 2>/dev/null || echo 0)
    if [ "$step" -ge "$MAX_STEPS" ]; then flag=DONE
    elif [ "$nojob" -lt 2 ]; then flag="transient-no-job(${nojob})"
    elif [ "${streak:-0}" -ge "$MAX_NOPROGRESS" ]; then
      flag=ALERT; alert="CHAIN-DOWN + launcher hit no-progress cap (streak=$streak): DELIBERATE stop (deterministic bug / bad nodes) -> NOT auto-resubmitting; HUMAN needed (read newest slurm .err)."
    elif [ "$presub" -ge "$MAX_PRESUB" ]; then
      flag=ALERT; alert="CHAIN-DOWN: poller already auto-resubmitted $presub times with no lasting job -> giving up; HUMAN needed."
    else
      presub=$((presub+1))
      echo "[$ts] AUTO-RESUBMIT #$presub via submit_tiered.sh:" >> "$MON/poller.log"
      ( bash "$REPO/submit_tiered.sh" ) >> "$MON/poller.log" 2>&1
      nojob=0
      flag=ALERT; alert="CHAIN-DOWN -> AUTO-RESUBMITTED via submit_tiered.sh (poller resubmit #$presub/$MAX_PRESUB, sbatch only, no scancel); re-checking next poll."
    fi
  else
    nojob=0; presub=0   # a job exists again -> chain alive -> reset the poller-resubmit counter
    if [ "$state" = RUNNING ]; then
      pending=0
      # gate on the job's OWN runtime: a fresh resume inherits an OLD last.ckpt + needs ~30min compile before its
      # first ckpt -> only flag STALL once the job itself has run > STALL_SECS yet STILL hasn't checkpointed.
      rt_secs=$(rt2s "$rtime")
      { [ "$rt_secs" -gt "$STALL_SECS" ] && [ "$age" -gt "$STALL_SECS" ]; } && { flag=ALERT; alert="STALL: $jid RUNNING (rt=$rtime/${rt_secs}s) but last.ckpt ${age}s stale, step=$step"; }
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
