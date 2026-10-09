#!/bin/bash
#SBATCH --job-name=sse1d_trace
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/%x-%j.out
# Lists every file the CA template model data path opens (for the A6000 copy).
set -uo pipefail
: "${EXPECT_COMMIT:?}"
REPO=$(mktemp -d)/proteina_sse1d
git clone -q -b sse-template-1d git@github.com:Yuwan3154/proteina.git "$REPO" && git -C "$REPO" checkout -q "$EXPECT_COMMIT" || exit 2
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SYNTH_INDEX_DIR=/orcd/scratch/orcd/011/chenxiou/synth_index_v4
export PACK_PATH=/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack
export CUE_HOST=engaging SKIP_CONFIND_PRECOMPUTE=1 CUDA_VISIBLE_DEVICES=
export TORCH_CUDA_ARCH_LIST=9.0  # activate.sh requires it; only steers JIT builds, none happen on CPU
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID:-nojob}; mkdir -p "$TMPDIR"
source "$ENV_DIR/.venv/bin/activate"; source "$ENV_DIR/activate.sh"
cd "$REPO"; export PYTHONPATH=$REPO
echo "commit $(git rev-parse --short HEAD) host $(hostname)"

python scripts/sse1d_trace_data_files.py --n_batches 4 --out /orcd/scratch/orcd/011/chenxiou/sse1d_trace_${SLURM_JOB_ID}.json --sizes model.nn.decompress=basis_pool model.nn.token_dim=64 model.nn.pair_repr_dim=32 model.nn.dim_cond=64 model.nn.nheads=4 model.nn.n_pre=1 model.nn.n_mid=1 model.nn.n_post=1 model.nn.mid_dim=32 model.nn.spectral_heads=2 model.nn.spectral_r=4 model.nn.spectral_hidden=16
rc=$?; echo "rc=$rc"; exit $rc
