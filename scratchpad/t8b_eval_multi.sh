#!/bin/bash
# T8b eval (user 2026-11-08): submit ONE job per partition for a stage, each with that partition's own max wall clock and
# GPU type, all named t8e_<STAGE> and run through t8b_eval_claim.sh: the first to start claims the stage and cancels the
# others by job id (claims/<STAGE>.jobs), so exactly one does the work. Login node: sbatch only. Pass every stage
# variable (MODE, ARM, MODEL, TDSSP, CHUNK, ...) explicitly on the command line; they reach the job via --export=ALL.
#   STAGE=<unique name> KIND=cpu|gpu PARTS="<partitions>" [CPUS=8 MEM=64G] bash scratchpad/t8b_eval_multi.sh <script> [args]
# GPU types: user 2026-10-13 (L40S, H100, H200, RTX PRO 6000 only; never A100); mit_preemptable only for a stage that is
# safe to rerun (user 2026-10-25) -- the caller decides via PARTS.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
E=$S/t8b_eval
: "${STAGE:?}" "${KIND:?}" "${PARTS:?}"
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
cd "$S/proteina_xrun" || exit 3
mkdir -p "$E/logs" "$E/claims" "$E/submitted"
mkdir "$E/submitted/$STAGE" 2>/dev/null || { echo "FATAL: stage name $STAGE already submitted ($E/submitted/$STAGE)"; exit 2; }
[ -f "$1" ] || { echo "FATAL: no script $1 in $PWD"; exit 2; }
export T8E_KIND=$KIND
for p in $PARTS; do
  case "$p" in
    pi_so3)          T=2-00:00:00; G=gpu:1 ;;          # pi_so3 GPUs: h200 + rtx_pro_6000 only, both approved
    mit_preemptable) T=2-00:00:00; G=gpu:l40s:1 ;;
    mit_normal_gpu)  T=06:00:00;   G=gpu:l40s:1 ;;
    mit_normal)      T=12:00:00;   G= ;;
    *) echo "FATAL: unknown partition $p"; exit 2 ;;
  esac
  GRES=()
  if [ "$KIND" = gpu ]; then
    [ -n "$G" ] || { echo "FATAL: $p has no GPUs"; exit 2; }
    GRES=(--gres="$G")
  fi
  j=$(sbatch --parsable --export=ALL -J "t8e_$STAGE" --partition="$p" --time="$T" "${GRES[@]}" \
      --nodes=1 --ntasks-per-node=1 --cpus-per-task="$CPUS" --mem="$MEM" --no-requeue \
      --exclude=node5101,node5204,node4400 \
      --output="$E/logs/%x-%j-$p.out" --error="$E/logs/%x-%j-$p.err" \
      scratchpad/t8b_eval_claim.sh "$STAGE" "$@") || { echo "FATAL: sbatch on $p"; exit 3; }
  echo "$j" >> "$E/claims/$STAGE.jobs"
  echo "$STAGE $p $j"
done
