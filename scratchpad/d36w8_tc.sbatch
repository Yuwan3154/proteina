#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/006/chenxiou/d36_w8_plain/logs/%x-%A_%a.out
# D36 W8 (A): plain-kernel re-score of the published d36tc AF2Rank pools (same call as d36tc_af2rank.py production:
# recycles 6, mask_sidechains, whole chain, production template-sequence mask; only the kernel set changes).
#   MODE=control : d36tc_A_tmpl 8TVL_A,6UCK_A x 2 models, then d36w8_tc_ctrl3.py vs control (iii) (exit 1 unless exact)
#   MODE=prod    : array; task i = line i+1 of $W/tc_tasks.tsv (label, model, stems, n_pools); resumable per pool
#   MODE=score   : CPU; d36tc_score.py on the plain pools -> $W/d36tc_per_target_plain.csv
# Partition/gres/time/name/array on the sbatch command line.
set -uo pipefail
W=/orcd/pool/006/chenxiou/d36_w8_plain
REPO=$W/repo
OUT=$W/af2rank
CTRL3=/orcd/scratch/orcd/011/chenxiou/d36_af2rank_mask/w4_20261121/ctrl_plain/prod_plain
: "${MODE:?}"
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
if [ "$MODE" = score ]; then
  module load miniforge
  set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true; conda activate cue_openfold; set -u
  export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
  python scratchpad/d36tc_score.py --af2rank "$OUT" --out "$W/d36tc_per_target_plain.csv"
  rc=$?
  echo "W8_TC_RC=$rc"
  exit $rc
fi
source "$REPO/scratchpad/d36tc_af2.env.sh"
export PYTHONPATH=$REPO
export TMPDIR=/tmp/chenxiou_w8_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for pm in "params_model_1_ptm.npz dfca7cd972028fc6adcc0c8995857a2a" "params_model_2_ptm.npz edb0729d11e83734b820c348c5ffe42a"; do
  set -- $pm; h=$(md5sum < "$AF2RANK_PARAMS_DIR/$1" | cut -d' ' -f1); echo "[params] $1 $h (SC $2)"
  [ "$h" = "$2" ] || { echo "FATAL: $1 differs from the SuperCloud production params"; exit 72; }
done
if [ "$MODE" = control ]; then
  rc=0
  for m in model_1_ptm model_2_ptm; do
    python -m scratchpad.d36tc_af2rank --model $m --labels d36tc_A_tmpl --stems 8TVL_A,6UCK_A --expect-pools 2 \
      --kernels plain --out "$OUT"
    r=$?; [ $r -eq 0 ] || { echo "CONTROL_AF2RANK_RC=$r ($m)"; rc=$r; }
  done
  if [ $rc -eq 0 ]; then
    python -m scratchpad.d36w8_tc_ctrl3 --new "$OUT" --ref "$CTRL3"
    rc=$?
  fi
elif [ "$MODE" = prod ]; then
  i=${SLURM_ARRAY_TASK_ID:?}
  LINE=$(sed -n "$((i + 1))p" "$W/tc_tasks.tsv")
  [ -n "$LINE" ] || { echo "FATAL: no task line $((i + 1)) in $W/tc_tasks.tsv"; exit 3; }
  IFS=$'\t' read -r LABEL MODEL STEMS NPOOLS <<< "$LINE"
  echo "[prod] task $i: $LABEL $MODEL stems $STEMS ($NPOOLS pools)"
  python -m scratchpad.d36tc_af2rank --model "$MODEL" --labels "$LABEL" --stems "$STEMS" --expect-pools "$NPOOLS" \
    --kernels plain --out "$OUT"
  rc=$?
else
  echo "unknown MODE=$MODE"; rc=2
fi
echo "W8_TC_RC=$rc"
exit $rc
