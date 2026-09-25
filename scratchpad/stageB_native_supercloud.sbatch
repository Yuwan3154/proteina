#!/bin/bash
#SBATCH --job-name=c2cbench_sc
#SBATCH --partition=debug-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/c2cbench/logs/%x-%j.out
# T5 (user 2026-09-25): tbeta native-map benchmark on SuperCloud debug-gpu, two checkpoints at once (one per V100):
# EMA 21,228 (before the step-23,515 structure-metric jump) vs the last.ckpt snapshot (after). Same harness, chain list
# and arm as the 23 Sep EMA-vs-raw benchmark (stageB_c2c_from_maps.py --native, EMA weights, 1 seed). fp32 (the harness
# has no autocast), so both arms share one precision; V100 numerics are compared only with each other.
set -uo pipefail
D=/home/gridsan/cou/c2cbench
REPO=$D/proteina
# /etc/profile first (as run_full_pipeline_afdb_slurm.sh does), else `module`/`source activate` silently no-op
set +u   # /etc/profile.d/Z97-byobu.sh reads an unset LC_BYOBU and kills a `set -u` shell at 0 s
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate cathfold
set -u
PY=/home/gridsan/cou/.conda/envs/cathfold/bin/python   # absolute: a failed activation must not fall back to /usr/bin/python
export DATA_PATH=$D/data SKIP_CONFIND_PRECOMPUTE=1 SIGMA_DATA=16.0 PYTHONPATH=$REPO TMPDIR=$D/tmp
mkdir -p $D/logs $D/out $D/tmp
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
git -C $REPO log -1 --format="[repo] %h"
CH=$REPO/scratchpad/stageA/stageA_run_union.txt
DS=pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828
run() {  # $1 gpu, $2 ckpt, $3 label
  CUDA_VISIBLE_DEVICES=$1 $PY $REPO/scratchpad/stageB_c2c_from_maps.py --ckpt "$2" --weights ema --dataset $DS \
      --chains $CH --n_seeds 1 --native --out $D/out/$3.jsonl --label $3 --usalign /home/gridsan/cou/.local/bin/USalign \
      > $D/logs/$3.log 2>&1
}
for f in $D/out/B_native_tbeta_ema21228_sc.jsonl $D/out/B_native_tbeta_last_sc.jsonl; do
  [ -s "$f" ] && { echo "FATAL: $f exists -- one fresh file per arm"; exit 78; }
done
run 0 $D/ckpt/tbeta_ema-epoch=12-step=21228.ckpt B_native_tbeta_ema21228_sc & P0=$!
run 1 $D/ckpt/tbeta_last.ckpt B_native_tbeta_last_sc & P1=$!
wait $P0; R0=$?; wait $P1; R1=$?
echo "RC ema21228=$R0 last=$R1"
for f in $D/out/B_native_tbeta_ema21228_sc.jsonl $D/out/B_native_tbeta_last_sc.jsonl; do echo "$(wc -l < $f 2>/dev/null) rows $f"; done
[ $R0 -eq 0 ] && [ $R1 -eq 0 ]
exit $?
