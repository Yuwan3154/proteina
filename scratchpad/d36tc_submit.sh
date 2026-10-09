#!/bin/bash
# D36tc submitter (run on the Engaging LOGIN node; sbatch only). Every GPU job is named d36tc_gpu with
# --dependency=singleton, so at most one runs at a time. STAGE picks what to submit:
#   controlB | controlC | prod (tri + c2c + native-map, groups A/B/C) | af2rank | score
# Seed pairing: every arm of a group gets the SAME chain list (tri: group<G>_x8.txt; c2c: group<G>.txt).
set -euo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
I=$P/inputs
R=$S/proteina_d36tc/scratchpad
TRI=$S/stageA/ckpt/cfft_tri_last-EMA_step60505.ckpt;      TRI_MD5=f8328020bde6267c632d5c8345f8eca2
C2C=$S/c2c_store/c2c_confind_tbeta/final_twin_s29000.ckpt; C2C_MD5=1f5693e773b05ac643a33b9a25ebec9f
GPU="--job-name=d36tc_gpu --partition=mit_normal_gpu --gres=gpu:l40s:1 --time=06:00:00 --exclude=node5101"
STAGE="${STAGE:?controlB|controlC|prod|af2rank|score}"
cd "$P/logs"
for f in d36tc_tri d36tc_c2c d36tc_controlB d36tc_controlC d36tc_af2rank d36tc_score; do bash -n "$R/$f.sbatch"; done

case "$STAGE" in
  controlB) echo "controlB $(sbatch --parsable $GPU --dependency=singleton $R/d36tc_controlB.sbatch)" ;;
  controlC) echo "controlC $(sbatch --parsable $GPU --dependency=singleton $R/d36tc_controlC.sbatch)" ;;
  prod)
    for G in A B C; do
      if [ "$G" = C ]; then TD=d36tc_tri_max493; CD=d36tc_c2c_max493; else TD=d36tc_tri_max384; CD=d36tc_c2c_max384; fi
      ARMS="self tmpl"; [ "$G" = B ] && ARMS="self"
      for ARM in $ARMS; do
        L=d36tc_${G}_${ARM}
        MAPV=""; [ "$ARM" = tmpl ] && MAPV=$I/d36tc_template_map.json
        T=$(env CKPT=$TRI EXPECT_MD5=$TRI_MD5 CHAINS=$I/group${G}_x8.txt N=8 LABEL=$L MAP=$MAPV DATASET=$TD \
            sbatch --parsable --export=ALL $GPU --dependency=singleton $R/d36tc_tri.sbatch)
        C=$(env CKPT=$C2C EXPECT_MD5=$C2C_MD5 CHAINS=$I/group${G}.txt MAPS_DIR=$P/tri/$L LABEL=$L DATASET=$CD \
            sbatch --parsable --export=ALL $GPU --dependency=afterok:$T,singleton $R/d36tc_c2c.sbatch)
        echo "$L tri $T c2c $C"
      done
      L=d36tc_${G}_native
      N=$(env CKPT=$C2C EXPECT_MD5=$C2C_MD5 CHAINS=$I/group${G}.txt NATIVE=1 N_SEEDS=8 LABEL=$L DATASET=$CD \
          sbatch --parsable --export=ALL $GPU --dependency=singleton $R/d36tc_c2c.sbatch)
      echo "$L c2c $N"
    done ;;
  af2rank)
    for M in model_1_ptm model_2_ptm; do
      echo "$M $(env MODEL=$M LABELS=d36tc_A_self,d36tc_A_tmpl,d36tc_B_self,d36tc_C_self,d36tc_C_tmpl \
          sbatch --parsable --export=ALL $GPU --dependency=singleton $R/d36tc_af2rank.sbatch)"
    done ;;
  score) echo "score $(sbatch --parsable $R/d36tc_score.sbatch)" ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
