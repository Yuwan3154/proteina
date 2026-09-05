#!/bin/bash
#SBATCH --job-name=gradvar
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_sh
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# ⛔ Copy the checkpoint first: the live run rewrites last.ckpt every 500 steps, and reading it
# mid-write gives a truncated file (or a silently stale one).
SRC=/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_v1/last.ckpt
CK="$TMPDIR/gradvar-$SLURM_JOB_ID.ckpt"
cp "$SRC" "$CK" || { echo "FATAL: could not copy $SRC"; exit 77; }
python scratchpad/grad_variance.py --ckpt "$CK" \
    --structures "${NSTRUCT:-4}" --copies "${NCOPY:-8}"
rc=$?
rm -f "$CK"
echo "GRADVAR_EXIT=$rc"
exit $rc
