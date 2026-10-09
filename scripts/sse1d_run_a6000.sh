#!/bin/bash
# One CA template-model run on ONE A6000 GPU (user 2026-10-09: the 4 decompressors on the 4 A6000s).
#   scripts/sse1d_run_a6000.sh <decompress> <gpu> <run_name> [extra hydra overrides...]
# Data = the Engaging files mirrored under ~/sse1d_data/<abs path> (scripts/sse1d_xfer_to_hub.sbatch). Effective batch
# 32 = the tri's 4 GPU x bs 1 x accum 8, here 1 GPU x bs 1 x accum 32. Resumes from store/<run_name>/checkpoints.
set -uo pipefail
DEC=$1; GPU=$2; RUN=$3; shift 3
ROOT=$HOME/sse1d_data
export DATA_PATH=$ROOT/orcd/pool/006/chenxiou/proteina/data
export SYNTH_INDEX_DIR=$ROOT/orcd/scratch/orcd/011/chenxiou/synth_index_v4
export PACK_PATH=$ROOT/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU  # same numbering as nvidia-smi -i
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)")
if [ "$busy" -ne 0 ]; then echo "GPU $GPU busy ($busy processes)"; exit 3; fi
cd "$HOME/proteina_sse1d" || exit 2
# the env has ~/proteina's proteinfoundation installed; without this, train.py's imports come from that checkout
export PYTHONPATH=$HOME/proteina_sse1d
PF=$("$HOME/miniconda3/envs/proteina_sse1d/bin/python" -c "import proteinfoundation; print(proteinfoundation.__file__)")
[ "$PF" = "$HOME/proteina_sse1d/proteinfoundation/__init__.py" ] || { echo "wrong proteinfoundation: $PF"; exit 4; }
echo "commit $(git rev-parse --short HEAD) dec=$DEC gpu=$GPU run=$RUN start $(date) pkg=$PF"
exec "$HOME/miniconda3/bin/conda" run --no-capture-output -p "$HOME/miniconda3/envs/proteina_sse1d" \
  python proteinfoundation/train.py --config_name training_ca_template_compress_v1 \
  --ngpus_per_node 1 --nnodes 1 --accumulate_grad_batches 32 \
  "model.nn.decompress=$DEC" "run_name_=$RUN" \
  "validation_sampling.fixed_chain_list=$ROOT/orcd/scratch/orcd/011/chenxiou/valset_analysis/val_fixed32_max256.txt" "$@"
