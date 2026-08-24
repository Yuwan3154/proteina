#!/bin/bash
#SBATCH --job-name=dtsweep
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# Sweep the number of sampling steps for BOTH arms on the SAME fixed 32 val chains.
# Steps = ceil(1/dt). 200 was the inherited default; it was cut to 50 purely for cost, while
# local_attn was the arm being tuned -- tri's step count has never been validated.
#
# ⛔ SAFETY: each variant runs under its OWN probe run_name_ with its own store, seeded by a
# COPY of the live checkpoint. The live stores are never opened for writing.
# Validation fires via the sanity check (num_sanity_val_steps=1) at startup, so each variant
# needs only model load + one validation pass; we kill it as soon as that pass completes.
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

# steps -> dt (ceil(1/dt) = steps)
declare -A DT=( [50]=0.02 [100]=0.01 [150]=0.0066667 [200]=0.005 [400]=0.0025 )
ARM="${ARM:?set ARM=tri|la}"
if [ "$ARM" = "tri" ]; then
  SRC=tri_full384; CONFIG=training_contact_tri_full384_v1
else
  SRC=localattn_full384; CONFIG=training_contact_hier_topology_full384_v1
fi

for STEPS in 50 100 150 200 400; do
  RUN="dtprobe_${ARM}_${STEPS}"
  echo "================ $ARM  steps=$STEPS  dt=${DT[$STEPS]}  run=$RUN ================"
  rm -rf "$REPO/store/$RUN"
  mkdir -p "$REPO/store/$RUN/checkpoints"
  cp "$REPO/store/$SRC/checkpoints/last.ckpt" "$REPO/store/$RUN/checkpoints/last.ckpt" || { echo "  no source ckpt"; continue; }
  # Full output to a per-variant file FIRST. Piping straight into grep discards the exception
  # and leaves an empty .err, which is exactly how the previous attempt hid its own failure.
  VLOG="$REPO/logs/dtsweep_${RUN}.log"
  timeout 600 torchrun --standalone --nproc_per_node=1 proteinfoundation/train.py \
      --config_name "$CONFIG" --ngpus_per_node 1 --nnodes 1 \
      run_name_="$RUN" \
      validation_sampling.dt="${DT[$STEPS]}" \
      opt.max_epochs=1 \
      log.checkpoint=False > "$VLOG" 2>&1
  echo "  exit=$?  full log: $VLOG"
  grep -E "fixed chain set loaded|traj: exit|val_end: gather" "$VLOG" | head -6
  grep -E "^[A-Za-z_.]*Error|MisconfigurationException|^ValueError|^KeyError" "$VLOG" | head -4
  echo "  ---- variant $STEPS done ----"
done
echo "=== sweep complete ==="
