#!/bin/bash
#SBATCH --job-name=sm120fix
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# Repairs the two packages still broken in the sm_120 env. Every value below is taken from
# cue-openfold-env-sm120/postsync.sh or the recorded env manual -- none are guessed.
#
# pyg_lib: chase_pyg failed with "pyg-lib was not found in the package registry" because the
#   SOURCE-BUILD FALLBACK omits the PyG flat index. pyg-lib is not on PyPI at all, so
#   `uv pip install pyg-lib` can never resolve. The index must be passed to the fallback too.
#   glibc here is 2.28 and the cu128 prebuilt wheel needs GLIBC_2.32, hence --no-binary.
# flash-attn: same glibc story; postsync's recipe is FLASH_ATTENTION_FORCE_BUILD=TRUE at 2.8.3.
#   ~60-90 min to compile. MAX_JOBS bounded to keep peak RAM in check.
# CPU partition is correct: neither build needs a GPU, only nvcc.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
PYG_INDEX=https://data.pyg.org/whl/torch-2.7.1+cu128.html

module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export MAX_JOBS=6
# nvtx3 headers moved in newer CUDA; pyg_lib's CMake looks for the old libnvToolsExt path.
export CPATH="${CUDA_HOME:-/usr/local/cuda}/include${CPATH:+:$CPATH}"

TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "[before] torch=$TORCH_BEFORE glibc=$(ldd --version | head -1) nvcc=$(command -v nvcc || echo MISSING)"
uv pip install "cmake<4"

echo "=============== pyg_lib (source build, WITH the PyG index) ==============="
python -c 'import pyg_lib' 2>/dev/null && echo "  already OK" || \
  uv pip install --no-cache --no-binary :all: --no-build-isolation --no-deps pyg_lib -f "$PYG_INDEX"

echo "=============== flash-attn 2.8.3 (forced source build) ==============="
python -c 'import flash_attn' 2>/dev/null && echo "  already OK" || {
  uv pip uninstall flash-attn >/dev/null 2>&1 || true
  FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --no-build-isolation --no-deps flash-attn==2.8.3
}

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || echo "FATAL: torch CHANGED $TORCH_BEFORE -> $TORCH_AFTER"

echo "=============== VERIFY BY IMPORT (exit status proves nothing) ==============="
for p in torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib torch_geometric flash_attn; do
  python - "$p" <<'PY'
import importlib, sys
m = sys.argv[1]
try:
    importlib.import_module(m)
    print(f"  {m:20s} OK")
except Exception as e:
    print(f"  {m:20s} FAIL {type(e).__name__}: {str(e)[:110]}")
PY
done
echo "[after ] torch=$TORCH_AFTER"
