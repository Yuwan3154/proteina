#!/bin/bash
#SBATCH --job-name=toporef_regimes
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=node5101,node5204,node1800,node1801,node1305,node3006,node2700
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# TopoRef inference regimes benchmark (Tasks 1.0, 1a, 1b, and Task 2 SSE evaluation) on Engaging L40S.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
REPO="${REPO:-$S/proteina_d36tc}"
OUT="${OUT:-$P/toporef_regimes}"
MODE="${MODE:-all}"

ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"

# Ensure Frame2ConFind, Frame2seq, ContactEBM, and pydssp are importable
export FRAME2SEQ_HOME="${FRAME2SEQ_HOME:-$S/Frame2seq}"
CUE_CONDA_SP="/home/chenxiou/miniforge3/envs/cue_openfold/lib/python3.10/site-packages"
[ -d "$CUE_CONDA_SP" ] || CUE_CONDA_SP="/orcd/home/002/chenxiou/miniforge3/envs/cue_openfold/lib/python3.10/site-packages"
export PYTHONPATH="$REPO:$S:$S/Frame2seq:$S/ContactEBM${CUE_CONDA_SP:+:$CUE_CONDA_SP}${PYTHONPATH:+:$PYTHONPATH}"

export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1
export SIGMA_DATA=16.0
export ALIGN_W=0.3
export MLM_W=0.3
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID:-manual$$}
mkdir -p "$TMPDIR" "$OUT" "$P/logs"
trap 'rm -rf "$TMPDIR"' EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h"

TRI_CKPT="${TRI_CKPT:-$S/stageA/ckpt/cfft_tri_last-EMA_step60505.ckpt}"
C2C_CKPT="${C2C_CKPT:-$S/c2c_store/c2c_confind_tbeta/final_twin_s29000.ckpt}"
INDEX="${INDEX:-$P/data/topology_index_d36tc.pt}"
F2C_DIR="${F2C_DIR:-$S/Frame2ConFind/runs/f2s_v384_smoothap-pearson}"
CEBM_REPO="${CEBM_REPO:-$S/ContactEBM}"
CEBM_CKPT="${CEBM_CKPT:-$S/ContactEBM/runs/confind_a1_prod100/checkpoints/model_step_100000.pt}"

for f in "$TRI_CKPT" "$C2C_CKPT" "$INDEX" "$F2C_DIR/best.pt" "$CEBM_CKPT"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 77; }
done

EXTRA=()
[ "${ONLY_BAD_AFDB22:-0}" = "1" ] && EXTRA+=(--only-bad-afdb22)
[ -n "${MAX_TARGETS:-}" ] && EXTRA+=(--max-targets "$MAX_TARGETS")
[ -n "${ARM_OVERRIDE:-}" ] && EXTRA+=(--arm-override "$ARM_OVERRIDE")

python scratchpad/toporef_regimes_bench.py \
  --mode "$MODE" \
  --d36tc-dir "$P" \
  --natives-dir "$S/d36u/natives" \
  --tri-ckpt "$TRI_CKPT" \
  --c2c-ckpt "$C2C_CKPT" \
  --index "$INDEX" \
  --f2c-ckpt-dir "$F2C_DIR" \
  --f2c-dtype "${F2C_DTYPE:-fp32}" \
  --cebm-repo "$CEBM_REPO" \
  --cebm-ckpt "$CEBM_CKPT" \
  --usalign "${USALIGN:-/home/chenxiou/.local/bin/USalign}" \
  --out "$OUT" \
  --k "${K:-8}" \
  --n-cycles "${N_CYCLES:-5}" \
  --t-eval "${T_EVAL:-0.05}" \
  "${EXTRA[@]}"
rc=$?
echo "[done] toporef_regimes_bench mode=$MODE rc=$rc"
exit $rc
