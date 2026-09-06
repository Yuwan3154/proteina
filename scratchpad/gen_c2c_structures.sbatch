#!/bin/bash
#SBATCH --job-name=c2cgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_sh
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ⛔ Copy the checkpoint first: the live run rewrites last.ckpt every 100 steps and reading it
# mid-write gives a truncated file.
SRC="${CKPT:-/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_v1/last.ckpt}"
CK="$TMPDIR/gen-$SLURM_JOB_ID.ckpt"
cp "$SRC" "$CK" || { echo "FATAL: cannot copy $SRC"; exit 77; }

US=$(command -v USalign || echo "$HOME/bin/USalign")
echo "USalign: $US"
python scratchpad/gen_c2c_structures.py --ckpt "$CK" --n "${NGEN:-6}" \
    --steps "${STEPS:-200}" --out "${OUT:-/orcd/scratch/orcd/011/chenxiou/c2c_gen}" \
    --usalign "$US"
rc=$?
rm -f "$CK"
echo "GEN_EXIT=$rc"
exit $rc
