#!/bin/bash
#SBATCH --job-name=probectl
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_probe/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_probe/logs/%x-%j.err
#
# THE MISSING CONTROL for the tri alignment probe.
#
# The trained probe scored precision@Q 0.737 against a 0.383 positional floor. But a linear probe on
# RANDOM features can score well on its own, and this repo's own tests note the tri trunk is
# near-identity at initialisation. So without an untrained arm we cannot say the 0.737 reflects
# anything the model LEARNED, as opposed to what the architecture exposes for free.
#
# Same pipeline, same chains, same ground truth, same floor -- only the weights differ.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_probe
cd "$REPO"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export USALIGN=/home/chenxiou/.local/bin/USalign
module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch, hydra; print('precheck ok', torch.__version__)" || exit 90
"$USALIGN" -h >/dev/null 2>&1 || { echo "USalign not runnable"; exit 91; }

OUT=$REPO/probe_data/tri_randominit
python scratchpad/probe_capture.py --random_init --out "$OUT" --n_samples "${N_SAMPLES:-150}"
rc=$?
echo "CAPTURE_EXIT=$rc"
[ "$rc" -ne 0 ] && exit $rc

echo ""
echo "================ CONTROL: RANDOM INIT ================"
python scratchpad/probe_train.py --data "$OUT" --epochs "${EPOCHS:-8}" --folds 5
rc1=$?
echo ""
echo "================ TRAINED: tri epoch 105 ================"
python scratchpad/probe_train.py --data "$REPO/probe_data/tri_ep105" --epochs "${EPOCHS:-8}" --folds 5
rc2=$?
echo "PROBECTL_EXIT rc1=$rc1 rc2=$rc2"
[ "$rc1" -ne 0 ] && exit $rc1
exit $rc2
