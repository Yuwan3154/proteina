#!/bin/bash
#SBATCH --job-name=t7_pairs
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/chenxiou/qlogs/t7_pairs_%j.out
# T7 pre-flight: same (query, template) pairs for the CB-8 and ConFind tris on the 195-chain union (CPU).
set -uo pipefail
REPO="${REPO:-/orcd/scratch/orcd/011/chenxiou/proteina_dev}"
cd "$REPO/scratchpad"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export PYTHONPATH=$REPO DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1 CUDA_VISIBLE_DEVICES=
git log -1 --format="[repo] %h"
python t7_pair_identity.py stageA/stageA_run_union.txt /home/chenxiou/qlogs/t7_pairs_union195.tsv 0
rc=$?
echo "T7_PAIRS_EXIT=$rc"
exit $rc
