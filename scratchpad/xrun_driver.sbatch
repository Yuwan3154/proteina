#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=pi_so3
#SBATCH --time=2-00:00:00
#SBATCH --exclude=node5101,node5204,node1800,node1801,node1305,node3006,node2700
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/toporef_xrun/logs/%x-%j.out
# Cross-run TopoRef comparison, stage C driver (CPU, sbatch/squeue/sacct only). Submits the remaining (model, arm, chunk)
# pass + c2c pairs of $X/todo.txt through xrun_submit_c.sh as the mit_normal_gpu per-user submit limit (64 jobs,
# QOSMaxSubmitJobPerUserLimit hit 10-29) frees; t8b items only once T8B_CKPT exists. STOPS submitting and writes
# $X/ALARM on any failed / timed-out / cancelled / preempted xp_ or xb_ job, or a pass RUNNING past ALARM_H hours.
#   T8B_CKPT=<captured ckpt>  (todo.txt lines: "<model> <arm> <chunk>", t8b lines last; the capture job writes the
#   ckpt through tmp + os.replace, so its existence = complete). Relaunch by hand after a TIMEOUT: todo.txt is the state.
# Exits 0 only when todo is empty AND every recorded job COMPLETED.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
X=$S/toporef_xrun
LIMIT=64                        # mit_normal_gpu MaxSubmitPU (SUMMARY do/don't, 10-13)
ALARM_H=5.5                     # passes run under a 6 h time limit; T7 chunks of 25 chains x 8 finished inside it
POLL=300
cd "$S/proteina_xrun" || exit 3
[ -f "$X/todo.txt" ] || { echo "FATAL: no $X/todo.txt"; exit 3; }
sed -i '/^[[:space:]]*$/d; s/\r$//' "$X/todo.txt"
if grep -q '^t8b ' "$X/todo.txt"; then [ -n "${T8B_CKPT:-}" ] || { echo "FATAL: t8b in todo but T8B_CKPT unset"; exit 3; }; fi
while true; do
  Q=$(squeue -u chenxiou -h -o "%P") || { echo "$(date +%T) squeue failed; retry"; sleep $POLL; continue; }
  n=$(echo "$Q" | grep -c mit_normal_gpu)
  ids=$(grep -ho "PASS [0-9]* C2C [0-9A-Z]*" "$X"/logs/submit_*.txt | awk '{print $2; if ($4 != "NONE") print $4}' | paste -sd, -)
  if [ -n "$ids" ]; then
    A=$(sacct -j "$ids" -X -n -P -o JobID,JobName,State,Elapsed) || { echo "$(date +%T) sacct failed; retry"; sleep $POLL; continue; }
    bad=$(echo "$A" | awk -F'|' '$3 ~ /FAILED|TIMEOUT|CANCELLED|PREEMPTED|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|REVOKED/')
    long=$(echo "$A" | awk -F'|' -v h=$ALARM_H '$2 ~ /^xp_/ && $3=="RUNNING" {split($4,t,"[-:]"); s=(length(t)==4? t[1]*86400+t[2]*3600+t[3]*60+t[4] : t[1]*3600+t[2]*60+t[3]); if (s > h*3600) print}')
    if [ -n "$bad$long" ]; then
      printf "%s\n%s\n%s\n" "$(date)" "$bad" "$long" > "$X/ALARM"
      echo "$(date +%T) ALARM:"; cat "$X/ALARM"; exit 1
    fi
    echo "$(date +%T) slots used $n/$LIMIT; done $(echo "$A" | grep -c COMPLETED)/$(echo "$A" | grep -c .)"
  fi
  left=$(grep -c . "$X/todo.txt")
  if [ "$left" -eq 0 ] && [ -n "$ids" ] && [ "$(echo "$A" | grep -vc '|COMPLETED|')" -eq 0 ] \
     && [ "$(echo "$A" | grep -c .)" -eq "$(echo "$ids" | tr , '\n' | grep -c .)" ]; then
    echo "$(date +%T) all submitted jobs COMPLETED"; exit 0
  fi
  while [ $(( LIMIT - n )) -ge 2 ] && [ -s "$X/todo.txt" ]; do
    read -r m arm k < <(head -1 "$X/todo.txt")
    if [ "$m" = t8b ] && [ ! -f "${T8B_CKPT:-}" ]; then break; fi
    MODELS=$m ARMS=$arm CHUNKS=$k T8B_CKPT="${T8B_CKPT:-}" bash scratchpad/xrun_submit_c.sh >> "$X/logs/submit_driver.txt" 2>&1
    rc=$?
    if [ $rc -eq 4 ]; then echo "$(date +%T) pass sbatch refused for $m $arm $k; retry next poll"; break
    elif [ $rc -ne 0 ]; then
      { date; echo "submit_c rc=$rc for $m $arm $k"; tail -3 "$X/logs/submit_driver.txt"; } > "$X/ALARM"; cat "$X/ALARM"; exit 1
    fi
    sed -i 1d "$X/todo.txt"
    n=$(( n + 2 ))
    echo "$(date +%T) submitted $m $arm $k"
  done
  sleep $POLL
done
