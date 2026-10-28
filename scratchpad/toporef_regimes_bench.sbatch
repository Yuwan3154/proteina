#!/bin/bash
#SBATCH --job-name=toporef_regimes
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/%x-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/%x-%A_%a.err
# TopoRef inference regimes benchmark (Tasks 1.0, 1a, 1b, and Task 2 SSE evaluation) on SuperCloud (2x V100 per node).
# Submit as an array over nodes (e.g. --array=0-4 with N_NODES=5 -> 10 shards = 1 target per GPU).
source /etc/profile
set -eo pipefail

W=/home/gridsan/cou/tri_synth_bench
REPO="${REPO:-/home/gridsan/cou/proteina_toporef_regimes}"
OUT="${OUT:-$W/toporef_regimes}"
MODE="${MODE:-all}"
N_NODES="${N_NODES:-5}"
ARM_OVERRIDE="${ARM_OVERRIDE:-all3}"
T_EVAL="${T_EVAL:-0.25}"

ENV_DIR=/home/gridsan/cou/cue-openfold-env
export CUE_HOST=supercloud
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"

export TMPDIR=/home/gridsan/cou/.tmp_toporef_${SLURM_JOB_ID:-manual$$}
mkdir -p "$TMPDIR" "$OUT" /home/gridsan/cou/logs
trap 'rm -rf "$TMPDIR"' EXIT

export SIGMA_DATA=16.0 ALIGN_W=0.3 MLM_W=0.3 SKIP_CONFIND_PRECOMPUTE=1 PYTHONUNBUFFERED=1
export SYNTH_INDEX_DIR="$W/inputs"
export DATA_PATH="$W/data"
F2C_PARENT="${F2C_PARENT:-$W/pkgs}"
export FRAME2SEQ_HOME="${FRAME2SEQ_HOME:-$W/pkgs/Frame2seq}"
CEBM_REPO="${CEBM_REPO:-/home/gridsan/cou/ContactEBM}"
export PYTHONPATH="$REPO:$F2C_PARENT:$FRAME2SEQ_HOME:$CEBM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=20

TRI_CKPT="${TRI_CKPT:-$W/inputs/cfft_tri_last-EMA_step60505.ckpt}"
C2C_CKPT="${C2C_CKPT:-$W/inputs/final_twin_s29000.ckpt}"
INDEX="${INDEX:-$W/inputs/topology_index_confind_synth.pt}"
F2C_DIR="${F2C_DIR:-/home/gridsan/cou/Frame2ConFind/runs/f2s_v384_smoothap-pearson}"
CEBM_CKPT="${CEBM_CKPT:-/home/gridsan/cou/c2cbench/yard/ckpt/confind_a1_prod100_epoch82.pt}"
USALIGN="${USALIGN:-/home/gridsan/cou/USalign/USalign}"

for f in "$TRI_CKPT" "$C2C_CKPT" "$INDEX" "$F2C_DIR/best.pt" "$CEBM_CKPT" "$USALIGN"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 77; }
done

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS=$(( N_NODES * 2 ))
SHARD0=$(( TASK_ID * 2 ))
SHARD1=$(( TASK_ID * 2 + 1 ))

COMMON=(
  --mode "$MODE"
  --tri-ckpt "$TRI_CKPT"
  --c2c-ckpt "$C2C_CKPT"
  --index "$INDEX"
  --f2c-ckpt-dir "$F2C_DIR"
  --f2c-dtype "${F2C_DTYPE:-fp32}"
  --cebm-repo "$CEBM_REPO"
  --cebm-ckpt "$CEBM_CKPT"
  --usalign "$USALIGN"
  --out "$OUT"
  --arm-override "$ARM_OVERRIDE"
  --k "${K:-8}"
  --n-cycles "${N_CYCLES:-5}"
  --t-eval "$T_EVAL"
  --n-shards "$N_SHARDS"
)
[ -n "${MAX_TARGETS:-}" ] && COMMON+=(--max-targets "$MAX_TARGETS")

echo "[launch] node task=$TASK_ID running shards $SHARD0 and $SHARD1 of $N_SHARDS (arm=$ARM_OVERRIDE t_eval=$T_EVAL)"
CUDA_VISIBLE_DEVICES=0 python scratchpad/toporef_regimes_bench.py "${COMMON[@]}" --shard "$SHARD0" > "$OUT/shard_${SHARD0}.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python scratchpad/toporef_regimes_bench.py "${COMMON[@]}" --shard "$SHARD1" > "$OUT/shard_${SHARD1}.log" 2>&1 &
PID1=$!

RC0=0; RC1=0
wait "$PID0" || RC0=$?
wait "$PID1" || RC1=$?
echo "[done] shard $SHARD0 rc=$RC0 | shard $SHARD1 rc=$RC1"
[ "$RC0" -eq 0 ] && [ "$RC1" -eq 0 ] || exit 1
exit 0
