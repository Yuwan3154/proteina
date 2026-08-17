#!/bin/bash
#SBATCH --job-name=seq_backfill
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

# Backfills graph.sequence into the processed .pt tree (see
# proteinfoundation/utils/backfill_graph_sequence.py). Non-preemptable on purpose even though the
# pass is restartable: it rewrites a tree other training runs read, so fewer interruptions is
# simply less churn. --resume is always on, which is what makes a requeue cheap.
#
# Set both before submitting; neither is guessed here because they differ per checkout:
#   PROTEINA_REPO=<repo dir>  DATA_PATH=<data dir>  CONDA_ENV=<env>  sbatch run_backfill_sequence.sbatch
set -uo pipefail

: "${PROTEINA_REPO:?set PROTEINA_REPO to the proteina checkout}"
: "${DATA_PATH:?set DATA_PATH to the directory holding pdb_train/}"
: "${CONDA_ENV:=cue_openfold}"
PROCESSED="${PROCESSED_DIR:-$DATA_PATH/pdb_train/processed}"
DONE_LOG="${DONE_LOG:-$DATA_PATH/pdb_train/backfill_sequence.done}"
SKIP_LOG="${SKIP_LOG:-$DATA_PATH/pdb_train/backfill_sequence.skips.tsv}"
WORKERS="${WORKERS:-32}"   # matches the topology index build, the only measured point we have

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate "$CONDA_ENV"

cd "$PROTEINA_REPO"
[ -d "$PROCESSED" ] || { echo "no such processed dir: $PROCESSED"; exit 2; }
echo "[$(date)] host=$(hostname) processed=$PROCESSED workers=$WORKERS"

# Dry run first: it reports how many files would change without touching any of them, so a wrong
# --processed-dir shows up as "0 would_write" instead of as a rewritten tree.
python -u -m proteinfoundation.utils.backfill_graph_sequence \
    --processed-dir "$PROCESSED" --workers "$WORKERS" --dry-run --limit 2000
echo "[$(date)] dry run done, starting the real pass"

python -u -m proteinfoundation.utils.backfill_graph_sequence \
    --processed-dir "$PROCESSED" --done-log "$DONE_LOG" --resume \
    --skip-log "$SKIP_LOG" --workers "$WORKERS"
RC=$?
echo "[$(date)] backfill exited rc=$RC, verifying"

python -u -m proteinfoundation.utils.backfill_graph_sequence \
    --processed-dir "$PROCESSED" --verify --skip-log "$SKIP_LOG.verify" --workers "$WORKERS"

# Every skipped file is reopened and described: a count alone does not establish that a
# skip was correct, and the loop now swallows load errors rather than dying on them.
for L in "$SKIP_LOG" "$SKIP_LOG.verify"; do
    [ -s "$L" ] && python -u -m proteinfoundation.utils.backfill_graph_sequence \
        --processed-dir "$PROCESSED" --audit-skips "$L" --workers "$WORKERS"
done
echo "[$(date)] done rc=$RC"
exit $RC
