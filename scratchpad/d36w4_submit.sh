#!/bin/bash
# D36 W4 submission chain (Engaging login node; sbatch only): precheck -> stage -> control (gate) -> prod array.
# Every step runs on the same GPU partitions/GPU type: openfold's deepspeed import needs a GPU driver.
set -uo pipefail
R=/orcd/scratch/orcd/011/chenxiou/d36_af2rank_mask/w4_20261121
S=$R/repo/scratchpad/d36w4.sbatch
X=node5101,node5204
G="-p mit_preemptable,mit_normal_gpu --gres=gpu:l40s:1 --exclude=$X"
j0=$(sbatch --parsable -J d36w4_pre $G --cpus-per-task=4 --mem=16G --time=00:10:00 --requeue \
  --export=ALL,MODE=precheck "$S") || exit 1
j1=$(sbatch --parsable -J d36w4_stage $G --cpus-per-task=4 --mem=16G --time=00:45:00 --requeue \
  --dependency=afterok:$j0 --export=ALL,MODE=stage "$S") || exit 1
j2=$(sbatch --parsable -J d36w4_ctrl $G --mem=96G --time=02:00:00 --requeue \
  --dependency=afterok:$j1 --export=ALL,MODE=control "$S") || exit 1
j3=$(sbatch --parsable -J d36w4_prod $G --mem=96G --time=02:30:00 --requeue --array=0-19 \
  --dependency=afterok:$j2 --export=ALL,MODE=prod "$S") || exit 1
echo "W4_SUBMITTED precheck $j0 stage $j1 control $j2 prod $j3"
