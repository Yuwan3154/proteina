#!/bin/bash
#SBATCH --job-name=pyg_sm120
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# torch_sparse and pyg_lib still carry PREBUILT wheels that need GLIBC_2.29+ (libpyg.so) while
# Engaging is glibc 2.28, so they fail at dlopen -- long after the install "succeeded".
# postsync.sh already has this exact probe+rebuild fallback, so the wheels here are stale state,
# not a missing recipe. --no-cache is the addition: without it uv can hand back a previously built
# wheel and the "source build" is a silent no-op.
# CPU partition on purpose: these need nvcc, not a GPU (recorded in the env manual), and the login
# node's memory cgroup OOM-kills the compile.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"

echo "[$(date)] glibc: $(ldd --version | head -1)"
echo "[$(date)] nvcc:  $(command -v nvcc || echo MISSING)"
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "[torch before] $TORCH_BEFORE"

for pkg in torch_sparse pyg_lib; do
  echo "=============== $pkg ==============="
  if python -c "import $pkg" >/dev/null 2>&1; then
    echo "$pkg already importable; leaving it alone."
    continue
  fi
  uv pip uninstall "${pkg//_/-}" || true
  uv pip install --no-cache --no-binary :all: --no-build-isolation --no-deps "${pkg//_/-}"
  if python -c "import $pkg; print('  $pkg import OK')"; then
    echo "$pkg rebuilt and importable."
  else
    echo "$pkg STILL not importable after a from-source build -- needs its own investigation."
  fi
done

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
echo "[torch after ] $TORCH_AFTER"
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || echo "FATAL: torch changed ($TORCH_BEFORE -> $TORCH_AFTER)"

echo "=== final probe of all five PyG companions + the training import ==="
python - <<'PY'
for name in ("torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "pyg_lib", "torch_geometric"):
    try:
        __import__(name)
        print(f"  {name:20s} OK")
    except Exception as e:
        print(f"  {name:20s} FAIL {type(e).__name__}: {str(e)[:110]}")
PY
export PYTHONPATH=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier${PYTHONPATH:+:$PYTHONPATH}
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
python -c "import proteinfoundation.datasets.pdb_data as m; print('  pdb_data import OK ->', m.__file__)" \
  || echo "  pdb_data STILL failing"
