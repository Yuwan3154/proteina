#!/bin/bash
#SBATCH --job-name=test_ot
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/ot_test/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/ot_test/logs/%x-%j.out

source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/ot_test
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
source "$ENV_DIR/.venv/bin/activate"
cd "$REPO"

echo "=== unit: ot_alignment ==="
python scratchpad/test_ot_alignment.py; U=$?
echo
echo "=== integration: ContactMapTriSiT + OT head ==="
python scratchpad/test_ot_integration.py; I=$?
echo
echo "UNIT_RC=$U INTEGRATION_RC=$I"
