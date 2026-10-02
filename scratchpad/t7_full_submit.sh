#!/bin/bash
# T7-FULL: tri sampling (16 chunk jobs) -> c2c scoring afterok (16) + native ceilings for the 96 rest queries (2).
# Checkpoints = the (3) set (user 2026-10-02). Run on the login node (sbatch only).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
F=$S/t7/full; O=$S/t7/full_out; mkdir -p $O
CK=$S/stageA/ckpt
TB=$S/c2c_store/c2c_cb8_tbeta/final_tbeta_s29620.ckpt
TW=$S/t7/twin_t7q_s18800.ckpt
DCB=pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828
DCF=pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828
G="--gres=gpu:l40s:1 --cpus-per-task=8 --mem=32G --time=06:00:00 --exclude=node5101"
cd $S/proteina_t7
for i in 0 1 2 3 4 5 6 7; do c=c0$i
  for m in cb8 cf; do
    if [ $m = cb8 ]; then MODEL=new; TCK=$CK/new_tri_last-EMA_step92700.ckpt; C2C=$TB; D=$DCB; BL=t7fl_cb8_$c
    else MODEL=cfft; TCK=$CK/cfft_tri_last-EMA_step24872.ckpt; C2C=$TW; D=$DCF; BL=t7fl_cf18800_$c; fi
    J=$(env MODEL=$MODEL ARM=nonself REGIME=single ZERO_CA=0 SEED=0 REPO_OVERRIDE=$S/proteina_t7 CHAINS=$F/chunk0${i}.txt \
        CKPT=$TCK LABEL=t7f_${m}_$c sbatch --parsable --export=ALL --job-name=t7f_${m}_$c --partition=mit_normal_gpu $G \
        scratchpad/stageA_pass.sbatch) || exit 4
    B=$(env REPO=$S/proteina_t7 CKPT=$C2C DATASET=$D CHAINS=$F/chunk0${i}_q.txt MAPS_DIR=$S/stageA/t7f_${m}_$c LABEL=$BL \
        OUT=$O/B_$BL.jsonl sbatch --parsable --export=ALL --dependency=afterok:$J --job-name=t7fB_${m}_$c \
        --partition=mit_normal_gpu,mit_preemptable $G scratchpad/stageB_c2c_from_maps.sbatch) || exit 5
    echo "$m $c TRI $J B $B"
  done
done
for m in cb8 cf; do
  if [ $m = cb8 ]; then C2C=$TB; D=$DCB; BL=t7fl_cb8_native; else C2C=$TW; D=$DCF; BL=t7fl_cf18800_native; fi
  N=$(env REPO=$S/proteina_t7 CKPT=$C2C DATASET=$D CHAINS=$F/rest_q.txt NATIVE=1 LABEL=$BL OUT=$O/B_$BL.jsonl \
      sbatch --parsable --export=ALL --job-name=t7fN_$m --partition=mit_normal_gpu,mit_preemptable $G \
      scratchpad/stageB_c2c_from_maps.sbatch) || exit 6
  echo "$m NATIVE $N"
done
