#!/bin/bash
#SBATCH --job-name=cb_dc_free_node_monitor
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue                   # mit_preemptable defaults to auto-requeue on preemption; without this, SLURM's
                                        # auto-requeue AND our own on_term trap BOTH resubmit -> duplicate running jobs
                                        # (hit 2026-07-01: 16964905 auto-requeued to node2902 while its trap spawned 16970787)
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/free-node-monitor-%j.out
# Polls every 30min for an H200/H100 node with >=4 GPUs of the SAME type idle (our --nodes=1 --gres=gpu:<type>:4
# requirement). Gates on GPU count only (not mem) -- every node seen with >=4 free GPUs has had far more than the
# 300-400G we need, so GPU count is the binding constraint (verified 2026-07-01: max free-on-one-node was 2).
# CPU-only (doesn't touch the 4-GPU cap); self-resubmits on SIGTERM (48h limit/preemption); stops on training
# max_steps or a manual STOP marker.

set -uo pipefail
REPO=/home/chenxiou/proteina
RUN=dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
MON="$REPO/store/$RUN/monitor"
LOG="$MON/FREE_NODE_LOG"
FOUND="$MON/FREE_NODE_FOUND"
STOP="$MON/FREE_NODE_MONITOR_STOP"
PART=mit_preemptable
NEED=4
INTERVAL=1800
MAX_STEPS=150000
RESUBMITTED=0

on_term() {
    step=$(grep -oP 'step=\K[0-9]+' "$MON/STATUS" 2>/dev/null | head -1 || echo 0)
    if [ "$step" -ge "$MAX_STEPS" ] || [ -f "$STOP" ]; then
        echo "[$(date)] training done or STOP present -> no resubmit" >> "$LOG"
    elif [ "$RESUBMITTED" -eq 0 ]; then
        RESUBMITTED=1
        echo "[$(date)] SIGTERM -> resubmit free-node monitor" >> "$LOG"
        (cd "$REPO" && sbatch run_free_node_monitor.sbatch) || true
    fi
    exit 0
}
trap on_term TERM

check_type() {  # $1=gpu type (h200|h100); prints "node type free" for nodes with free >= NEED
  local gtype="$1"
  sinfo -p "$PART" -N -h -o '%N %t' 2>/dev/null | awk '$2 !~ /drain|resv|down|maint/ {print $1}' | while read -r n; do
    local show cfgline allocline cfg alloc free
    show=$(scontrol show node="$n" 2>/dev/null)
    cfgline=$(echo "$show" | grep 'CfgTRES=')
    echo "$cfgline" | grep -q "gres/gpu:${gtype}=" || continue
    cfg=$(echo "$cfgline" | grep -oP "gres/gpu:${gtype}=\K[0-9]+")
    allocline=$(echo "$show" | grep 'AllocTRES=')
    alloc=$(echo "$allocline" | grep -oP "gres/gpu:${gtype}=\K[0-9]+")
    alloc=${alloc:-0}
    free=$((cfg - alloc))
    [ "$free" -ge "$NEED" ] && echo "$n $gtype $free"
  done
}

poll_once() {
  local hits
  hits=$( { check_type h200; check_type h100; } )
  if [ -n "$hits" ]; then
    { echo "[$(date)] FOUND:"; echo "$hits"; } >> "$LOG"
    { echo "[$(date)]"; echo "$hits"; } > "$FOUND"
  else
    echo "[$(date)] no qualifying node (need >=$NEED free h200 or h100 on one node)" >> "$LOG"
    rm -f "$FOUND"
  fi
}

echo "[$(date)] free-node-monitor START on $(hostname) jobid=${SLURM_JOB_ID:-?}" >> "$LOG"
poll_once
while true; do
  sleep $INTERVAL &
  wait $!
  if [ -f "$STOP" ]; then
    echo "[$(date)] STOP marker present -> exiting" >> "$LOG"
    exit 0
  fi
  poll_once
done
