#!/bin/bash
#SBATCH --job-name=distotest
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_ot/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_ot/logs/%x-%j.err
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_ot
cd "$REPO"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PY=/orcd/home/002/chenxiou/cue-openfold-env-sm120/.venv/bin/python3
$PY scratchpad/test_distogram_head.py
rc=$?
echo "DISTOTEST_EXIT=$rc"
exit $rc
