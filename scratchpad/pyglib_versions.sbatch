#!/bin/bash
#SBATCH --job-name=pyglibver
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# The 0.8.0 pin came from the env manual, but that pin was recorded for the cu126 env.
# The cu128 index reports "no version of pyg-lib==0.8.0", so enumerate what it ACTUALLY hosts
# instead of guessing another version, then install whatever is available and prove it imports.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
PYG_INDEX=https://data.pyg.org/whl/torch-2.7.1+cu128.html
module load cuda/12.8 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
set +u
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
source /home/chenxiou/cue-openfold-env-sm120/activate.sh
set -u

echo "=============== what pyg_lib wheels does the cu128 index host? ==============="
python - "$PYG_INDEX" <<'PY'
import re, sys, urllib.request
url = sys.argv[1]
html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
hits = sorted(set(re.findall(r"pyg_lib-([0-9][^-]*)-([^\"'>]*?)\.whl", html)))
if not hits:
    print("  NO pyg_lib wheels on this index at all")
else:
    for ver, tag in hits:
        print(f"  pyg_lib {ver}  ({tag})")
PY

echo "=============== install whatever is available (unpinned) ==============="
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
uv pip install --no-deps pyg-lib -f "$PYG_INDEX" || echo "  unpinned install also failed"
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || echo "FATAL: torch CHANGED $TORCH_BEFORE -> $TORCH_AFTER"

echo "=============== VERIFY BY IMPORT ==============="
python -c "import pyg_lib; print('  pyg_lib', pyg_lib.__version__, 'OK')" 2>&1 | tail -2
python -c "import torch_geometric; print('  torch_geometric OK')" 2>&1 | tail -1
echo "[torch] $TORCH_AFTER"
