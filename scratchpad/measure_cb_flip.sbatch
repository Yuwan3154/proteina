#!/bin/bash
#SBATCH --job-name=cbflip
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Pair-level delta of the pseudo-CB switch. CPU-only on purpose: pure coordinate arithmetic, and it
# must not compete with training for a GPU. mit_quicktest caps at 15 min and has idle CPU capacity,
# so this lands immediately instead of queueing behind the GPU backlog.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export CUE_HOST=engaging
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"

python scratchpad/measure_cb_flip.py --n "${NCHAINS:-200}"
echo "CBFLIP_EXIT=$?"
