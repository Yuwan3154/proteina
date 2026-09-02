#!/bin/bash
#SBATCH --job-name=probefinal
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_probe/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_probe/logs/%x-%j.err
#
# Answers the two open questions about the probe head:
#   1. does it overfit?  -> train and test precision@Q are now both reported per fold
#   2. does the self-conditioning double-forward matter? -> --dedupe keeps one forward per
#      (query, reference) pair, so 150 forwards become 107 distinct pairs
# Both arms, both settings, so the four numbers are directly comparable.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_probe
cd "$REPO"
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
export OMP_NUM_THREADS=4
python -c "import torch; print('precheck ok', torch.__version__)" || exit 90

for ARM in tri_ep105 tri_randominit; do
  for MODE in "" "--dedupe"; do
    echo ""
    echo "================ $ARM ${MODE:-(all 150 forwards)} ================"
    python scratchpad/probe_train.py --data "$REPO/probe_data/$ARM" --epochs 8 --folds 5 $MODE
    echo "  exit=$?"
  done
done
echo "PROBEFINAL_DONE"
