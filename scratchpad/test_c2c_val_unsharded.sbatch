#!/bin/bash
#SBATCH --job-name=test_c2c_valids
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/chenxiou/qlogs/test_c2c_valids_%j.out
# Real-data check: c2c val dataloader gives every DDP rank the single-process val ids (both c2c datasets).
set -uo pipefail
REPO="${REPO:-/orcd/scratch/orcd/011/chenxiou/proteina_dev}"
cd "$REPO/scratchpad"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES= DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1
git log -1 --format="[repo] %h"
rc_all=0
for DS in pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828 pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828; do
  python test_c2c_val_unsharded.py "$DS" 2>&1 | grep -E "^epoch|ALL PASS|Error|error|assert|Traceback" ; rc=${PIPESTATUS[0]}
  echo "DS=$DS rc=$rc"; [ "$rc" -eq 0 ] || rc_all=$rc
done
echo "TEST_EXIT=$rc_all"
exit $rc_all
