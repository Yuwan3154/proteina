#!/bin/bash
#SBATCH --job-name=repair_sm120
#SBATCH --partition=pi_so3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# The cu128 rebuild re-resolved the WHOLE dependency graph, not just torch. Two packages regressed
# against the known-good cu126 env and both break the training path:
#   einops  0.6.1 -> 0.6.0   (0.6.0 has no `allow_ops_in_compiled_graph`; this is the actual crash)
#   biotite 1.5.0 -> 0.41.0  (four major versions back; structure parsing would break later)
# Versions below are copied from the working cue_openfold env, not chosen.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"

TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "[torch before] $TORCH_BEFORE"

uv pip install "einops==0.6.1" "biotite==1.5.0"

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
echo "[torch after ] $TORCH_AFTER"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "FATAL: the install changed torch ($TORCH_BEFORE -> $TORCH_AFTER). This is the exact"
    echo "failure mode already recorded for cuequivariance; refusing to continue."
    exit 1
fi

python - <<'PY'
import torch
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
import einops, biotite
from einops._torch_specific import allow_ops_in_compiled_graph
print("einops", einops.__version__, "| biotite", biotite.__version__)
print("arch list still carries sm_120:", torch.cuda.get_arch_list())
PY

echo "=== real training smoke on sm_120 ==="
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_sm120
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_sm120
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"

timeout 3000 torchrun --standalone --nproc_per_node=1 proteinfoundation/train.py \
    --config_name training_contact_tri_overfit_v1 \
    --ngpus_per_node 1 --nnodes 1 run_name_=sm120_smoke
echo "[exit] $?  (124 = timeout, which means it was still training happily)"
