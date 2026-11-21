#!/bin/bash
# D36 W4b submission chain (Engaging login node; sbatch only): precheck -> control (gate) -> prod array, KERNELS=plain.
# Reuses the staged templates + manifest of W4 stage 29939971 (no stage step). Plain kernels: control ii/iii and
# production write to ctrl_plain/ and out_plain/ (the cuEq W4 control/ dir is left untouched).
set -uo pipefail
R=/orcd/scratch/orcd/011/chenxiou/d36_af2rank_mask/w4_20261121
S=$R/repo/scratchpad/d36w4.sbatch
X=node5101,node5204
G="-p mit_preemptable,mit_normal_gpu --gres=gpu:l40s:1 --exclude=$X"
K=plain
j0=$(sbatch --parsable -J d36w4b_pre $G --cpus-per-task=4 --mem=16G --time=00:10:00 --requeue \
  --export=ALL,MODE=precheck "$S") || exit 1
j2=$(sbatch --parsable -J d36w4b_ctrl $G --mem=96G --time=02:00:00 --requeue \
  --dependency=afterok:$j0 --export=ALL,MODE=control,KERNELS=$K "$S") || exit 1
j3=$(sbatch --parsable -J d36w4b_prod $G --mem=96G --time=04:00:00 --requeue --array=0-19 \
  --dependency=afterok:$j2 --export=ALL,MODE=prod,KERNELS=$K "$S") || exit 1
echo "W4B_SUBMITTED kernels $K precheck $j0 control $j2 prod $j3"
