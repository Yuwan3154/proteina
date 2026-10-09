#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# ⛔ No --error: loguru warnings go to stderr, the [EMA load] print to stdout; both must reach ONE log.
#
# D36tc tri pass: the ConFind tri (cfft) on the D36tc eval dataset, self arm, optionally conditioned on another
# chain's index row (MAP = topology_reference_map JSON). Env block = stageA_pass.sbatch (cfft).
#   CKPT=<tri EMA>  EXPECT_MD5=<full md5 of CKPT, required>  CHAINS=<fixed list, stem x N>  N=<samples per stem>  LABEL=<pass name>
#   optional: MAP  DATASET (default d36tc_tri_max384)  SEED (0)  REPO ($S/proteina_d36tc)
# Partition, --gres, --time and the job name go on the sbatch command line.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
REPO="${REPO:-$S/proteina_d36tc}"
DATASET="${DATASET:-d36tc_tri_max384}"
PROC=$P/data/processed
INDEX=$P/data/topology_index_d36tc.pt
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1
export ALIGN_W=0.3 MLM_W=0.3
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$P/logs" "$P/tri"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

: "${CKPT:?}" "${EXPECT_MD5:?}" "${CHAINS:?}" "${N:?}" "${LABEL:?}"
MAP="${MAP:-}"
[ -f "$CHAINS" ] || { echo "FATAL: no chain list $CHAINS"; exit 77; }
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint $CKPT"; exit 77; }
[ -z "$MAP" ] || [ -f "$MAP" ] || { echo "FATAL: no map $MAP"; exit 77; }
DUMP=$P/tri/$LABEL
[ -e "$DUMP" ] && { echo "FATAL: $DUMP exists -- one fresh dir per pass"; exit 78; }
RUNLOG=$P/logs/run_${LABEL}_${SLURM_JOB_ID}.log
MAPARGS=(); [ -n "$MAP" ] && MAPARGS=(--map "$MAP")
TRIMAP=(); [ -n "$MAP" ] && TRIMAP=(--topology_reference_map "$MAP")
echo "[d36tc] $LABEL dataset=$DATASET map=${MAP:-none} chains=$CHAINS ($(grep -c . "$CHAINS")) N=$N seed=${SEED:-0}"
MD5=$(md5sum < "$CKPT" | cut -d" " -f1)
echo "[d36tc] ckpt=$CKPT md5=$MD5 expect=$EXPECT_MD5"
[ "$MD5" = "$EXPECT_MD5" ] || { echo "FATAL: checkpoint md5 $MD5 != EXPECT_MD5 $EXPECT_MD5"; exit 75; }
# The frozen tri files are full Lightning checkpoints (optimizer_states present) whose state_dict holds the EMA
# weights written by EmaModelCheckpoint (-EMA name); there is no separate EMA dict to compare, so the md5 pin is the gate.
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h"

PYTHONPATH="$REPO" python scratchpad/d36tc_preflight.py --chains "$CHAINS" --processed-dir "$PROC" --index "$INDEX" "${MAPARGS[@]}"
prc=$?
[ "$prc" -eq 0 ] || { echo "PREFLIGHT_RC=$prc"; exit "$prc"; }

PYTHONPATH="$REPO" python scratchpad/tri_gen_eval.py \
    --config_name training_contact_tri_full384_confindsynth_ft_v1 --dataset "$DATASET" \
    --ema_ckpt "$CKPT" --arm self "${TRIMAP[@]}" \
    --fixed_chain_list "$CHAINS" --limit_val_batches 1 --seed "${SEED:-0}" \
    --dump_dir "$DUMP" --store "$TMPDIR/store" > "$RUNLOG" 2>&1
rc=$?
cat "$RUNLOG"
echo "PASS_RC[$LABEL]=$rc"

PYTHONPATH="$REPO" python scratchpad/d36tc_postcheck.py --log "$RUNLOG" --dump "$DUMP" --chains "$CHAINS" \
    --processed-dir "$PROC" --n "$N" "${MAPARGS[@]}"
vrc=$?
echo "POSTCHECK_RC=$vrc"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$vrc"
