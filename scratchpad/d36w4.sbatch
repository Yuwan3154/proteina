#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/d36_af2rank_mask/w4_20261121/logs/%x-%A_%a.out
# D36 W4 (Q6). MODE=precheck (GPU: params + imports of every W4 module) | stage (GPU node: the scorer module's
# openfold/deepspeed import needs a GPU driver; no GPU compute) | control (GPU: feature unit test, control i, control ii + shakedown, gate) |
# prod (GPU array, 20 tasks: task i -> cell ORDER[i/4], model (i/2)%2, chunk i%2; resumable per target).
# KERNELS (control/prod, required) = run-wide AF2 kernel set: cueq (d36tc production) | plain (D36ab); controls go to
# $R/ctrl_$KERNELS (prod/ = control i, always the default cuEq path; native/ = control ii; prod_$KERNELS/ = control iii,
# only when KERNELS != cueq), production to $R/out_$KERNELS.
# Partition/gres/time/mem/name/array on the sbatch command line.
set -uo pipefail
R=/orcd/scratch/orcd/011/chenxiou/d36_af2rank_mask/w4_20261121
REPO=$R/repo
source "$REPO/scratchpad/d36tc_af2.env.sh"
export PYTHONPATH=$REPO
export TMPDIR=/tmp/chenxiou_w4_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
: "${MODE:?}"
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
python -c "import proteinfoundation, openfold; print('[import]', proteinfoundation.__file__, openfold.__file__)"
M=$R/templates_manifest.csv
if [ "$MODE" = control ] || [ "$MODE" = prod ]; then
  case "${KERNELS:-}" in cueq|plain) ;; *) echo "FATAL: KERNELS must be cueq or plain, got '${KERNELS:-}'"; exit 2;; esac
  CR=$R/ctrl_$KERNELS
  echo "[kernels] $KERNELS ; controls $CR ; production $R/out_$KERNELS"
fi

params_gate() {
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  for pm in "params_model_1_ptm.npz dfca7cd972028fc6adcc0c8995857a2a" "params_model_2_ptm.npz edb0729d11e83734b820c348c5ffe42a"; do
    set -- $pm; h=$(md5sum < "$AF2RANK_PARAMS_DIR/$1" | cut -d' ' -f1); echo "[params] $1 $h (SC $2)"
    [ "$h" = "$2" ] || { echo "FATAL: $1 differs from the SuperCloud production params"; exit 72; }
  done
}

if [ "$MODE" = precheck ]; then
  # openfold -> deepspeed -> triton autotune needs a GPU driver at IMPORT time (stage 29937396 failed on a CPU node)
  params_gate
  python -c "import scratchpad.d36w4_stage, scratchpad.d36w4_af2rank, scratchpad.d36w4_feature_test, scratchpad.d36w4_control_prod, scratchpad.d36w4_gate; print('[precheck] imports OK')"
  rc=$?
elif [ "$MODE" = stage ]; then
  python -m scratchpad.d36w4_stage --selected "$R/templates_selected.csv" --root "$R"
  rc=$?
elif [ "$MODE" = control ]; then
  params_gate
  python -m scratchpad.d36w4_feature_test --manifest "$M"
  rc=$?; echo "FEATURE_TEST_RC=$rc"; [ $rc -eq 0 ] || exit $rc
  STEMS=$(python -m scratchpad.d36w4_gate plan --manifest "$M" --what ctrl_stems) || exit 3
  PIDS=$(python -m scratchpad.d36w4_gate plan --manifest "$M" --what ctrl_pids) || exit 3
  echo "[control] control (i) stems $STEMS ; control (ii)/shakedown targets $PIDS"
  for m in model_1_ptm model_2_ptm; do
    python -m scratchpad.d36w4_control_prod --label d36tc_A_tmpl --stems "$STEMS" --model $m --out "$CR/prod"
    rc=$?; [ $rc -eq 0 ] || { echo "CONTROL_PROD_RC=$rc"; exit $rc; }
  done
  if [ "$KERNELS" != cueq ]; then
    for m in model_1_ptm model_2_ptm; do
      python -m scratchpad.d36w4_control_prod --label d36tc_A_tmpl --stems "$STEMS" --model $m --out "$CR/prod_$KERNELS" \
        --kernels "$KERNELS"
      rc=$?; [ $rc -eq 0 ] || { echo "CONTROL_III_RC=$rc"; exit $rc; }
    done
    python -m scratchpad.d36w4_gate kernel_report --manifest "$M" --ctrl-dir "$CR" --kernels "$KERNELS"
    rc=$?; [ $rc -eq 0 ] || { echo "KERNEL_REPORT_RC=$rc"; exit $rc; }
  fi
  for c in none seqx sc both gapsc; do
    for m in model_1_ptm model_2_ptm; do
      python -m scratchpad.d36w4_af2rank --cell $c --model $m --pids "$PIDS" --manifest "$M" \
        --out "$CR/native" --template-ids native --kernels "$KERNELS"
      rc=$?; [ $rc -eq 0 ] || { echo "CONTROL_NATIVE_RC=$rc ($c $m)"; exit $rc; }
    done
  done
  python -m scratchpad.d36w4_gate check --manifest "$M" --ctrl-dir "$CR" --kernels "$KERNELS" \
    --factorial-dir "$R/inputs/factorial"
  rc=$?
elif [ "$MODE" = prod ]; then
  params_gate
  ORDER=(none both seqx sc gapsc)
  i=${SLURM_ARRAY_TASK_ID:?}
  CELL=${ORDER[$((i / 4))]}
  if [ $(((i / 2) % 2)) -eq 0 ]; then MODEL=model_1_ptm; else MODEL=model_2_ptm; fi
  CHUNK=$((i % 2))
  PIDS=$(python -m scratchpad.d36w4_gate plan --manifest "$M" --what chunk --chunk $CHUNK) || exit 3
  echo "[prod] task $i: cell $CELL model $MODEL chunk $CHUNK pids $PIDS"
  python -m scratchpad.d36w4_af2rank --cell "$CELL" --model "$MODEL" --pids "$PIDS" --manifest "$M" \
    --out "$R/out_$KERNELS" --kernels "$KERNELS"
  rc=$?
else
  echo "unknown MODE=$MODE"; rc=2
fi
echo "W4_RC=$rc"
exit $rc
