#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=40
#SBATCH --output=/home/gridsan/cou/d36/w8/logs/%x-%A_%a.out
# D36 W8 (B): re-score the published 'ours' AF2Rank step (af2rank_uniform_rc6) on SuperCloud V100 with d36w8_ours.py.
# One node, 2 V100: model_1_ptm on GPU 0 and model_2_ptm on GPU 1 in parallel, for each kernel set in KERNEL_SETS.
#   MODE=control : PIDS from the env (2 targets), KERNEL_SETS="cueq plain"
#   MODE=prod    : array; task i = line i+1 of $W/ours_tasks.txt (comma list of pids), KERNEL_SETS="plain cueq"
# Env = sc_badafdb_rc6.sbatch's (cathfold, cuda/12.6, USalign on PATH). Time/name/array on the sbatch command line.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate cathfold
set -uo pipefail
export OMP_NUM_THREADS=20
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
W=/home/gridsan/cou/d36/w8
REPO=/home/gridsan/cou/proteina
OUT=$W/ours
TARS=/home/gridsan/cou/proteina/inference/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_200-epoch_pdb_purge-test_warmup_maxlen-256_cutoff-190828_unified/seq_cond_segment
CSVS=/home/gridsan/cou/data/bad_afdb/eval_bad_afdb_disorder_smooth06.csv,/home/gridsan/cou/d36/eval_bad_afdb_new16.csv
CIF=/home/gridsan/cou/data/bad_afdb/pdb
export PYTHONPATH=$REPO
: "${MODE:?}"
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for pm in "params_model_1_ptm.npz dfca7cd972028fc6adcc0c8995857a2a" "params_model_2_ptm.npz edb0729d11e83734b820c348c5ffe42a"; do
  set -- $pm; h=$(md5sum < "$HOME/openfold/openfold/resources/params/$1" | cut -d' ' -f1); echo "[params] $1 $h (want $2)"
  [ "$h" = "$2" ] || { echo "FATAL: $1 differs from the production params"; exit 72; }
done
if [ "$MODE" = control ]; then
  : "${PIDS:?}"
  KERNEL_SETS="cueq plain"
  TAG=control
elif [ "$MODE" = prod ]; then
  i=${SLURM_ARRAY_TASK_ID:?}
  PIDS=$(sed -n "$((i + 1))p" "$W/ours_tasks.txt")
  [ -n "$PIDS" ] || { echo "FATAL: no task line $((i + 1)) in $W/ours_tasks.txt"; exit 3; }
  KERNEL_SETS="plain cueq"
  TAG=prod$i
else
  echo "unknown MODE=$MODE"; exit 2
fi
echo "[w8] MODE=$MODE pids $PIDS kernels $KERNEL_SETS"
python -m scratchpad.d36w8_ours stage --pids "$PIDS" --out "$OUT" --tar-dir "$TARS"
rc=$?; [ $rc -eq 0 ] || { echo "STAGE_RC=$rc"; echo "W8_OURS_RC=$rc"; exit $rc; }
for K in $KERNEL_SETS; do
  CUDA_VISIBLE_DEVICES=0 python -m scratchpad.d36w8_ours score --pids "$PIDS" --out "$OUT" --tar-dir "$TARS" \
    --model model_1_ptm --kernels $K --tag $TAG --dataset-csvs "$CSVS" --cif-dir "$CIF" > "$W/logs/${SLURM_JOB_ID}_${TAG}_${K}_m1.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 python -m scratchpad.d36w8_ours score --pids "$PIDS" --out "$OUT" --tar-dir "$TARS" \
    --model model_2_ptm --kernels $K --tag $TAG --dataset-csvs "$CSVS" --cif-dir "$CIF" > "$W/logs/${SLURM_JOB_ID}_${TAG}_${K}_m2.log" 2>&1 &
  p2=$!
  wait $p1; r1=$?
  wait $p2; r2=$?
  echo "[w8] kernels $K: model_1_ptm rc $r1, model_2_ptm rc $r2"
  grep -h -E "^\[score\]|D36W8_OURS_DONE" "$W/logs/${SLURM_JOB_ID}_${TAG}_${K}_m1.log" "$W/logs/${SLURM_JOB_ID}_${TAG}_${K}_m2.log"
  [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || { echo "W8_OURS_RC=1"; exit 1; }
done
echo "W8_OURS_RC=0"
exit 0
