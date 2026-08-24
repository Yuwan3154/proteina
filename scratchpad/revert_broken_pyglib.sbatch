#!/bin/bash
#SBATCH --job-name=pyglibrv
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# I installed pyg-lib 0.5.0+pt27cu128, which CANNOT import (libpyg.so needs GLIBC_2.29,
# Engaging has 2.28). Installed-but-unimportable is WORSE than absent: torch_geometric
# degrades gracefully when pyg_lib is missing, but a present-yet-broken .so risks a runtime
# failure instead. Restore the prior state and prove torch_geometric still works.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
module load cuda/12.8 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
set +u
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
source /home/chenxiou/cue-openfold-env-sm120/activate.sh
set -u

echo "--- before ---"
uv pip list 2>/dev/null | grep -iE '^pyg-lib|^torch-geometric|^torch ' || true
uv pip uninstall pyg-lib
echo "--- after: verify the WHOLE PyG stack by import ---"
for p in torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric flash_attn; do
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
python -c "import pyg_lib" 2>/dev/null && echo "  pyg_lib STILL PRESENT (unexpected)" || echo "  pyg_lib absent (intended: graceful degradation)"
python -c "import torch; print('[torch]', torch.__version__)"
