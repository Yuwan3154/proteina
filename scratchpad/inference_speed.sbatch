#!/bin/bash
#SBATCH --job-name=infspeed
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

OUT=$REPO/curves/inference_speed.tsv
mkdir -p "$(dirname "$OUT")"

# L grid spans the dataset's own bounds (dataselector min_length 50, max_length 384). T grid spans
# the observed per-chain element counts (~10-35 from the fixed-32 traces) up to the config cap
# max_topology_he_len 64. Neither grid is a tuned value.
LENGTHS="${LENGTHS:-64 128 192 256 320 384}"
TOPO_LENS="${TOPO_LENS:-16 32 64}"

for ARM in ${ARMS:-tri la}; do
  if [ "$ARM" = "tri" ]; then
    CONFIG=training_contact_tri_full384_v1
  else
    CONFIG=training_contact_hier_topology_full384_v1
  fi
  for L in $LENGTHS; do
    # One process per (arm, L): an OOM costs this row only. On failure, retry once at a quarter of
    # the predicted batch -- fragmentation, not a wrong model of the scaling, is what makes the
    # measured prediction unreachable.
    python scratchpad/inference_speed.py \
        --config_name "$CONFIG" --tag "$ARM" --length "$L" \
        --topo_lens $TOPO_LENS --out_tsv "$OUT"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "!! $ARM L=$L failed (rc=$rc); retrying with a forced small batch"
      python scratchpad/inference_speed.py \
          --config_name "$CONFIG" --tag "$ARM" --length "$L" \
          --topo_lens $TOPO_LENS --out_tsv "$OUT" --force_batch 4
    fi
  done
done

echo "================ RESULTS ================"
column -t "$OUT"
