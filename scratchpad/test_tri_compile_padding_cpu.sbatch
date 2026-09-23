#!/bin/bash
#SBATCH --job-name=tri_cpad_cpu
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
# CPU run of the tri compile/padding checks in the real training env. REPO defaults to the dev checkout.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
REPO=${REPO:-$S/proteina_dev}
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
# activate.sh requires an arch value; on a CPU node it steers nothing
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export TORCHINDUCTOR_CACHE_DIR=$S/torchinductor_cache_cpu TMPDIR=$S/.tmp
cd "$REPO" && git log -1 --format="[repo] %h"
PYTHONPATH="$REPO" python scratchpad/test_tri_compile_padding.py --device cpu
exit $?
