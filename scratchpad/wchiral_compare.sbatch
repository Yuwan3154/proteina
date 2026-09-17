#!/bin/bash
#SBATCH --job-name=wchcmp
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# CPU-only: reads dumped validation PDBs and superposes. No GPU, must not queue behind training.
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

python scratchpad/wchiral_compare.py c2c_cb8_tbeta c2c_cb8_wch03 c2c_cb8_wch10
echo "WCHCMP_EXIT=$?"
