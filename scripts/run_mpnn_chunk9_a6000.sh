#!/bin/bash
# 4x RTX A6000 training launch script for ConFind TopoRef with fixed 9-residue ProteinMPNN featurization
# (reference_mode="mpnn_chunk9"), template sequence input with 25% i.i.d. sequence identity masking
# (coordinate features unmasked), 10% chunk token masking without MLM loss, and tmalign_smooth DSSP targets.
#
# Usage:
#   bash scripts/run_mpnn_chunk9_a6000.sh [run_name] [extra hydra overrides...]

set -euo pipefail

RUN="${1:-tri_v2_confind_mpnn_chunk9_seqmask25_tmsmooth}"
if [ $# -ge 1 ]; then shift 1; fi

REPO_DIR="${REPO_DIR:-$HOME/proteina_mpnn_chunk9}"
ROOT="${DATA_ROOT:-$HOME/sse1d_data}"
ENV_DIR="${ENV_DIR:-$HOME/miniconda3/envs/proteina_sse1d}"

export DATA_PATH="${DATA_PATH:-$ROOT/orcd/pool/006/chenxiou/proteina/data}"
export PACK_PATH="${PACK_PATH:-$ROOT/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack}"
export SYNTH_INDEX_DIR="${SYNTH_INDEX_DIR:-$HOME/sse1d_idx/mpnn_chunk9}"
export TEMPLATE_POOL_DIR="${TEMPLATE_POOL_DIR:-$HOME/sse1d_idx/t2_pool_auth}"
export CHAIN_ALIAS_PATH="${CHAIN_ALIAS_PATH:-$HOME/sse1d_idx/in/orcd/scratch/orcd/011/chenxiou/synth_alias/auth_key.tsv}"
export MPNN_CKPT_PATH="${MPNN_CKPT_PATH:-$HOME/ProteinMPNN/vanilla_model_weights/v_48_020.pt}"
export VAL_FIXED_LIST="${VAL_FIXED_LIST:-$ROOT/orcd/scratch/orcd/011/chenxiou/valset_analysis/val_fixed32_max256.txt}"
export ALIGN_W="${ALIGN_W:-0.3}"
export MLM_W="${MLM_W:-0.0}"
export SKIP_CONFIND_PRECOMPUTE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$REPO_DIR" || exit 2
export PYTHONPATH="$REPO_DIR"

PF=$("$ENV_DIR/bin/python" -c "import proteinfoundation; print(proteinfoundation.__file__)")
if [ "$PF" != "$REPO_DIR/proteinfoundation/__init__.py" ]; then
  echo "FATAL: wrong proteinfoundation imported: $PF (expected $REPO_DIR/proteinfoundation/__init__.py)"
  exit 4
fi

for req in \
  "$PACK_PATH" \
  "$SYNTH_INDEX_DIR/topology_index_confind_synth_mpnn_chunk9.pt" \
  "$SYNTH_INDEX_DIR/eligible_ids_tm0.5-0.9.txt" \
  "$TEMPLATE_POOL_DIR" \
  "$CHAIN_ALIAS_PATH" \
  "$MPNN_CKPT_PATH" \
  "$VAL_FIXED_LIST"; do
  if [ ! -e "$req" ]; then
    echo "FATAL: missing required path: $req"
    exit 5
  fi
done

echo "=== Launching 4x A6000 ConFind TopoRef MPNN-Chunk9 Training ==="
echo "Commit: $(git rev-parse --short HEAD) ($(git branch --show-current))"
echo "Run Name: $RUN"
echo "Repo: $REPO_DIR"
echo "Index: $SYNTH_INDEX_DIR/topology_index_confind_synth_mpnn_chunk9.pt"
echo "Pack: $PACK_PATH"
echo "Template Pool: $TEMPLATE_POOL_DIR"
echo "MPNN Weights: $MPNN_CKPT_PATH"
echo "Start Time: $(date)"

exec "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=4 \
  proteinfoundation/train.py \
  --config_name training_contact_tri_v2_confind_mpnn_chunk9 \
  --ngpus_per_node 4 \
  --nnodes 1 \
  --accumulate_grad_batches 8 \
  "run_name_=$RUN" \
  "$@"
