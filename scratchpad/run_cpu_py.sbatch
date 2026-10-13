#!/bin/bash
#SBATCH --job-name=cpu_py
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/logs_cpu_py_%j.out
# Run one python script (args after the script path) with the training env, CPU, mit_quicktest (15 min; override -p/-t).
# Usage: sbatch scratchpad/run_cpu_py.sbatch scratchpad/<script>.py [args...]
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
SCRIPT="${1:?usage: sbatch run_cpu_py.sbatch <script.py> [args...]}"
shift
REPO=$SLURM_SUBMIT_DIR  # submit from the checkout under test (sbatch runs a spooled copy of this file)
[ -f "$REPO/proteinfoundation/utils/t2_pack_reader.py" ] || { echo "FATAL: $REPO is not a T8 checkout"; exit 77; }
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SYNTH_INDEX_DIR="${SYNTH_INDEX_DIR:-/orcd/scratch/orcd/011/chenxiou/synth_index_v4}"
export ALIGN_W=0.3 MLM_W=0.3
export PACK_PATH="${PACK_PATH:-/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack}"
export DATALOADER_TIMEOUT_S=1800 SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID:-nojob}
mkdir -p "$TMPDIR"
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=9.0  # as run_tri_cb8synth_h200x4.sbatch; activate.sh requires it
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
git log --oneline -1
python "$SCRIPT" "$@"
rc=$?
echo "PY_EXIT=$rc"
exit $rc
