#!/bin/bash
#SBATCH --job-name=armcmp
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Full-range overfit arm comparison. CPU-only: it reads dumped PDBs and does superpositions.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
# activate.sh hard-requires this on Engaging and there is no nvidia-smi on a CPU node. Inert here.
export TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export CUE_HOST=engaging
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"

python scratchpad/arm_compare_full.py c2c_of8_baseL c2c_of8_w03L
echo "ARMCMP_EXIT=$?"
