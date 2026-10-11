#!/bin/bash
#SBATCH --job-name=plan_integ
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/logs_plan_integ_%j.out
# Real-data integration test of the epoch batch plan (scratchpad/test_plan_datamodule_integration.py), CPU only.
# Same data env as run_tri_cb8synth_h200x4.sbatch. Usage: sbatch scratchpad/run_plan_integration.sbatch <config_name> <accum>
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
CONFIG="${1:?usage: sbatch run_plan_integration.sbatch <config_name> <accum>}"
ACCUM="${2:?usage: sbatch run_plan_integration.sbatch <config_name> <accum>}"
REPO=$SLURM_SUBMIT_DIR  # submit from the checkout under test (sbatch runs a spooled copy of this file)
[ -f "$REPO/proteinfoundation/datasets/epoch_batch_plan.py" ] || { echo "FATAL: $REPO has no epoch_batch_plan.py"; exit 77; }
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SYNTH_INDEX_DIR="${SYNTH_INDEX_DIR:-/orcd/scratch/orcd/011/chenxiou/synth_index_v4}"
export ALIGN_W=0.3 MLM_W=0.3
export PACK_PATH="${PACK_PATH:-/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack}"
export DATALOADER_TIMEOUT_S=1800 SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[ -f "$PACK_PATH" ] && [ -f "$PACK_PATH.idx.npz" ] || { echo "FATAL: no pack at $PACK_PATH"; exit 77; }
[ -f "$SYNTH_INDEX_DIR/topology_index_cb8_synth.pt" ] || { echo "FATAL: no synthetic index at $SYNTH_INDEX_DIR"; exit 77; }
[ -f "$SYNTH_INDEX_DIR/eligible_ids_tm0.5-0.9.txt" ] || { echo "FATAL: no eligible-ids list at $SYNTH_INDEX_DIR"; exit 77; }
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID:-nojob}
mkdir -p "$TMPDIR"
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export CUE_HOST=engaging
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
git log --oneline -1
python scratchpad/test_plan_datamodule_integration.py --config_name "$CONFIG" --accum "$ACCUM" \
    --workdir "/orcd/scratch/orcd/011/chenxiou/plan_integ_${SLURM_JOB_ID}" --steps 50 --num_workers 12
rc=$?
echo "INTEG_EXIT=$rc"
exit $rc
