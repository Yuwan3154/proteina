#!/bin/bash
#SBATCH --job-name=t7_fpdist
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/chenxiou/qlogs/t7_fpdist_%j.out
# T7 (b): native CB-CB distance of each tri's false contacts (CPU), T7-quick dumps (best-by-sampling tri ckpts).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
REPO="${REPO:-$S/proteina_t7}"
cd "$REPO"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=
git log -1 --format="[repo] %h"
A=$S/stageA
python scratchpad/t7_fp_distance.py $S/t7/quick_out/fp_distance_best.jsonl \
  CB8=$A/t7q_cb8_c00,$A/t7q_cb8_c01,$A/t7q_cb8_c02,$A/t7q_cb8_c03 \
  CONFIND=$A/t7q_cf_c00,$A/t7q_cf_c01,$A/t7q_cf_c02,$A/t7q_cf_c03
rc=$?; echo "FPDIST_EXIT=$rc"; exit $rc
