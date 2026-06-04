#!/bin/bash
# Env wrapper for running proteina FoldBench inference/scoring on Engaging INSIDE a
# salloc allocation. Loads the full-stack inference env, sets the pool DATA_PATH and
# sequence-conditioned mode, then execs whatever command is passed as arguments.
#
# Usage (inside a salloc on a GPU node, hosted in tmux to survive SSH drops):
#   salloc --partition=mit_normal_gpu --gres=gpu:1 --time=6:00:00 --cpus-per-task=8 \
#     bash proteinfoundation/prediction_pipeline/run_foldbench_engaging.sh \
#       python proteinfoundation/inference.py --pt <id> --config_name <cfg> \
#         --conditioning_mode seq --nsamples_per_protein 16 --max_nsamples 16
set -euo pipefail

module load miniforge/25.11.0-0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cue_openfold

export PROTEINA_BASE_DIR="$HOME/proteina"
cd "$PROTEINA_BASE_DIR"
export DATA_PATH="${DATA_PATH:-/orcd/pool/006/chenxiou/proteina/data}"
export PROTEINA_CONDITIONING_MODE=seq   # sequence-conditioned (NOT seq+CATH)
export TOKENIZERS_PARALLELISM=false

echo "[wrapper] host=$(hostname) cuda_devices=${CUDA_VISIBLE_DEVICES:-unset} DATA_PATH=$DATA_PATH"
nvidia-smi -L 2>/dev/null | head || echo "[wrapper] nvidia-smi -L unavailable"
echo "[wrapper] exec: $*"
exec "$@"
