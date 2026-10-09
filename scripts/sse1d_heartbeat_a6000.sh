#!/bin/bash
# On-box heartbeat for the CA template-model runs (independent of any agent): every 10 min, one line per run with
# alive/latest top-k ckpt step/log size+mtime/error count, plus GPU memory+util. Never kills or relaunches anything.
#   setsid nohup bash scripts/sse1d_heartbeat_a6000.sh ca_tc1d_pair_row_xattn_v1 ... > ~/sse1d_runs/heartbeat.log 2>&1 &
while true; do
  ts=$(date "+%F %T")
  for run in "$@"; do
    f=$HOME/sse1d_runs/$run.log
    up=$(pgrep -f "run_name_=$run" >/dev/null && echo up || echo DOWN)
    step=$(ls "$HOME/proteina_sse1d/store/$run/checkpoints" 2>/dev/null | sed -n 's/^chk_epoch=[0-9]*_step=0*\([0-9][0-9]*\)\.ckpt$/\1/p' | sort -n | tail -1)
    err=$(grep -cE "Traceback|out of memory|transform failed|nan loss|NaN" "$f")
    echo "$ts $run $up ckpt_step=${step:-none} err=$err log=$(stat -c %s "$f")@$(date -r "$f" +%T)"
  done
  echo "$ts gpus $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  sleep 600
done
