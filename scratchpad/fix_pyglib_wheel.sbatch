#!/bin/bash
#SBATCH --job-name=pyglibwhl
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# Corrected pyg_lib approach. Evidence, in order:
#   1. `--no-binary :all:` is IMPOSSIBLE for pyg-lib: uv says "all versions of pyg-lib have no
#      source distribution". There is no sdist anywhere, so no pip/uv source build can exist.
#   2. The WORKING cu126 env has pyg-lib 0.8.0 installed as a WHEEL on this same glibc 2.28 host.
#      So a wheel is the intended and only mechanism.
# Therefore: install the wheel from the PyG flat index (no --no-binary), then PROVE it imports.
# Also re-checks the cu126 env as a control, since its wheel is known-good.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

PYG_INDEX=https://data.pyg.org/whl/torch-2.7.1+cu128.html
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
# activate.sh (line 40) HARD-REQUIRES this and aborts without it. 12.0 = sm_120 Blackwell,
# the same value the earlier working sm_120 job used. Only needs to be SET for an import
# test, but it must be set or activate.sh refuses.
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0

echo "=============== CONTROL: cu126 env (known-good wheel) ==============="
( set +u
  source /home/chenxiou/cue-openfold-env/.venv/bin/activate
  source /home/chenxiou/cue-openfold-env/activate.sh >/dev/null 2>&1 || true
  python -c "import pyg_lib; print('  cu126 pyg_lib', pyg_lib.__version__, 'IMPORTS OK')" 2>&1 | tail -2 )

echo "=============== TARGET: sm_120 env ==============="
set +u
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
source /home/chenxiou/cue-openfold-env-sm120/activate.sh
set -u
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "[before] torch=$TORCH_BEFORE"

# WHEEL install (the only possible route), pinned to the version the working env runs.
uv pip install --no-deps pyg-lib==0.8.0 -f "$PYG_INDEX"

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || echo "FATAL: torch CHANGED $TORCH_BEFORE -> $TORCH_AFTER"

echo "=============== VERIFY BY IMPORT ==============="
for p in torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib torch_geometric flash_attn; do
  python - "$p" <<'PY'
import importlib, sys
m = sys.argv[1]
try:
    importlib.import_module(m)
    print(f"  {m:20s} OK")
except Exception as e:
    print(f"  {m:20s} FAIL {type(e).__name__}: {str(e)[:120]}")
PY
done
echo "[after ] torch=$TORCH_AFTER"
