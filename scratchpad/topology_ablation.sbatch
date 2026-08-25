#!/bin/bash
#SBATCH --job-name=topoabl
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
cd "$REPO"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sample count must be matched across arms, and the arms differ in batch size: tri's dataset
# config pins batch_size 1 (its N^3 einsum), la's is 4, and the val loader inherits the train
# batch size. So 32 tri batches and 8 la batches both yield 32 samples. The first run used 8 for
# both, which gave tri only 8 samples -- half of them MASK dropout -- and no low-t sample with a
# real reference at all.
NB_TRI="${NB_TRI:-32}"
NB_LA="${NB_LA:-8}"
for ARM in ${ARMS:-tri la}; do
  if [ "$ARM" = "tri" ]; then
    CONFIG=training_contact_tri_full384_v1;            SRC=tri_full384;         NB=$NB_TRI
  else
    CONFIG=training_contact_hier_topology_full384_v1;  SRC=localattn_full384;   NB=$NB_LA
  fi
  SNAP=/tmp/${ARM}_abl_EMA_$$.ckpt
  cp "$REPO/store/$SRC/checkpoints/last-EMA.ckpt" "$SNAP"
  echo "================ $ARM  ($SRC) ================"
  python scratchpad/topology_ablation.py \
      --config_name "$CONFIG" --ema_ckpt "$SNAP" --tag "$ARM" --n_batches "$NB"
  echo "  exit=$?"
  rm -f "$SNAP"
done
