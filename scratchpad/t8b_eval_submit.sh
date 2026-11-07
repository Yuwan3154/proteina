#!/bin/bash
# T8b eval on the T7-NEWEST pairs (login node: sbatch only). After xrun_index.sbatch t7prep / t7f2c / t7build x3:
#   PHASE=passes  per chunk k in 0..3: one L40S pass per template SSE classifier (xrun_pass.sbatch ARM=t7_<TDSSP>),
#                 then afterok all three the ContactEBM job (t8b_eval_cebm.sbatch). No c2c here: stageB control (b)
#                 decides which chains c2c can score (T8b re-processed natives may differ from the c2c dataset's).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
E=$S/t8b_eval
CKPT=${T8B_CKPT:?T8B_CKPT = the frozen checkpoint in $E/ckpt}
G="--gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=06:00:00 --no-requeue --exclude=node5101,node5204,node4400"
mkdir -p "$E/logs"
[ -f "$CKPT" ] || { echo "FATAL: $CKPT"; exit 2; }
cd "$S/proteina_xrun" || exit 3
case "${PHASE:?}" in
  passes)
    for k in 0 1 2 3; do
      J=()
      for td in h_bond tmalign tmalign_smooth; do
        L=t8b_t7_${td}_c$k
        j=$(env MODEL=t8b ARM=t7_$td CHUNK=$k SEED=0 T8B_CKPT="$CKPT" sbatch --parsable --export=ALL -J xp_$L \
            --partition=mit_normal_gpu $G --output=$E/logs/%x-%j.out --error=$E/logs/%x-%j.err \
            scratchpad/xrun_pass.sbatch) || exit 4
        echo "$L PASS $j"; J+=("$j")
      done
      c=$(env CHUNK=$k sbatch --parsable --export=ALL -J xe_t8b_t7_cebm_c$k --dependency=afterok:$(IFS=:; echo "${J[*]}") \
          --partition=mit_normal_gpu $G scratchpad/t8b_eval_cebm.sbatch) || exit 5
      echo "t8b_t7_cebm_c$k CEBM $c"
    done
    ;;
  *) echo "FATAL: PHASE=$PHASE"; exit 2;;
esac
