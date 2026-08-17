#!/bin/bash
#SBATCH --job-name=topo_index
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Rebuilds the topology index with the element-pair featurization. Not restartable -- it assembles
# one flat tensor from a full pass over the tree -- so it runs on a non-preemptable partition.
# Memory is sized for the assembly: the feature block is roughly six bytes per element pair on top
# of the one byte the contact map already costs.
#
#   PROTEINA_REPO=<repo dir>  DATA_PATH=<data dir>  CONDA_ENV=<env>  sbatch run_topology_index.sbatch
set -uo pipefail

: "${PROTEINA_REPO:?set PROTEINA_REPO to the proteina checkout}"
: "${DATA_PATH:?set DATA_PATH to the directory holding pdb_train/}"
: "${CONDA_ENV:=cue_openfold}"
DATASET="${DATASET:-pdb_train_contact-confind_S25_max512_purge-test_cutoff-190828}"
OUT="${OUT:-$DATA_PATH/pdb_train/topology_index.pt}"
WORKERS="${WORKERS:-32}"
SKIP_LOG="${SKIP_LOG:-$DATA_PATH/pdb_train/topology_index.skips.tsv}"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate "$CONDA_ENV"

cd "$PROTEINA_REPO"
export DATA_PATH
echo "[$(date)] host=$(hostname) dataset=$DATASET out=$OUT workers=$WORKERS"

# Written beside the live index and moved into place only on success, so a failed or preempted
# rebuild cannot leave training pointed at a truncated file.
TMP="$OUT.building"
python -u -m proteinfoundation.utils.precompute_topology_index \
    --dataset "$DATASET" --out "$TMP" --workers "$WORKERS" --repo "$PROTEINA_REPO" \
    --skip-log "$SKIP_LOG"
RC=$?
if [ $RC -eq 0 ]; then
    [ -f "$OUT" ] && mv "$OUT" "$OUT.prev"
    mv "$TMP" "$OUT"
    ls -lh "$OUT"
fi
echo "[$(date)] done rc=$RC"
exit $RC
