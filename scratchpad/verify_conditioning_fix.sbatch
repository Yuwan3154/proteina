#!/bin/bash
#SBATCH --job-name=fixcheck
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=6:00:00
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
# Needed for the topo_present / topo_n_valid record that proves the reference reached the sampler.
export SAMPLETRACE=1

# Both arms in ONE job, sequentially: the mit_normal_gpu per-user cap is 2 GPUs and the pending
# training segment wants both, so holding a single GPU once beats queueing twice.
for ARM in tri la; do
  if [ "$ARM" = "tri" ]; then
    CONFIG=training_contact_tri_full384_v1;            SRC=tri_full384
  else
    CONFIG=training_contact_hier_topology_full384_v1;  SRC=localattn_full384
  fi
  SNAP=/tmp/${ARM}_fixcheck_EMA_$$.ckpt
  cp "$REPO/store/$SRC/checkpoints/last-EMA.ckpt" "$SNAP"
  echo "================ $ARM  ($SRC)  $(stat -c %s "$SNAP") bytes ================"
  python scratchpad/verify_conditioning_fix.py \
      --config_name "$CONFIG" --ema_ckpt "$SNAP" --tag "$ARM" --steps 50 --n_dump 4
  echo "  exit=$?"
  rm -f "$SNAP"
done
