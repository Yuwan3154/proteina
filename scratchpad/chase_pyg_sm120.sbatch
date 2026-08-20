#!/bin/bash
#SBATCH --job-name=chase_pyg
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# The three sm_120-env packages still broken after the first rebuild pass. None block training
# (torch_geometric and pdb_data both import), so this is cleanup, not a blocker:
#   torch_cluster      prebuilt wheel wants GLIBC_2.32, host is 2.28
#   torch_spline_conv  same
#   pyg_lib            source build failed, so the package is now absent entirely
# pyg_lib is pinned to 0.8.0 because the env manual records it as the last version buildable with
# cmake<4 plus the nvtx3 patch on modern CUDA -- that pin is quoted, not guessed. cmake is held
# below 4 for the same reason.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"

TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "[torch before] $TORCH_BEFORE   glibc: $(ldd --version | head -1)   nvcc: $(command -v nvcc || echo MISSING)"
echo "[cmake] $(cmake --version 2>/dev/null | head -1 || echo absent)"

# nvtx3 headers moved in newer CUDA; pyg_lib's CMake looks for the old path.
export CPATH="${CUDA_HOME:-/usr/local/cuda}/include${CPATH:+:$CPATH}"

echo "=============== torch_cluster / torch_spline_conv ==============="
for pkg in torch_cluster torch_spline_conv; do
  if python -c "import $pkg" >/dev/null 2>&1; then echo "$pkg already OK"; continue; fi
  uv pip uninstall "${pkg//_/-}" || true
  uv pip install --no-cache --no-binary :all: --no-build-isolation --no-deps "${pkg//_/-}" \
    && python -c "import $pkg; print('  $pkg OK')" || echo "  $pkg FAILED"
done

echo "=============== pyg_lib 0.8.0 ==============="
uv pip install "cmake<4"
uv pip install --no-cache --no-binary :all: --no-build-isolation --no-deps "pyg_lib==0.8.0" \
  && python -c "import pyg_lib; print('  pyg_lib OK')" || echo "  pyg_lib FAILED"

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
echo "[torch after ] $TORCH_AFTER"
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || echo "FATAL: torch changed ($TORCH_BEFORE -> $TORCH_AFTER)"

echo "=============== final state ==============="
python - <<'PY'
import torch
print("torch", torch.__version__, "| sm_120 present:", "sm_120" in torch.cuda.get_arch_list())
for name in ("torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "pyg_lib",
             "torch_geometric", "flash_attn"):
    try:
        __import__(name)
        print(f"  {name:20s} OK")
    except Exception as e:
        print(f"  {name:20s} FAIL {type(e).__name__}: {str(e)[:95]}")
PY
