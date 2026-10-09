#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc CONTROL B (GPU, L40S): does the d36tc harness reproduce the T7-NEWEST records?
# T7's first trajectory ran only after the WHOLE chunk's fixed batches were built (_fixed_val_batches +
# GlobalRotationTransform draw from the global RNG), so only a full-chunk rerun can match. Modes:
#   MODE=tri CHUNK=00|01 : tri_gen_eval on the FULL t7/quick/chunk<CHUNK>.txt (T7 dataset + synth_index_confind_v1,
#                          --arm nonself, seed 0, cfft step 48150 = the T7-NEWEST tri) -> $B/tri_c<CHUNK>
#   MODE=c2c CHUNK=00|01 : c2c ISOLATED -- stageB on the RECORD tri dump $S/stageA/t7qn_cf_c<CHUNK> with chains =
#                          chunk<CHUNK>_q.txt (same chain_index), twin s29000 EMA, T7 c2c dataset -> $B/B_c2c_c<CHUNK>.jsonl
# Compare with d36tc_controlB_compare.py (CPU). Partition/gres/time/name on sbatch.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
B=$P/controlB
Q=$S/t7/quick
REPO="${REPO:-$S/proteina_d36tc}"
TRI=$S/stageA/ckpt/cfft_tri_last-EMA_step48150.ckpt;      TRI_MD5=6fb81c1dffecff820f7f9c8e572d83be
C2C=$S/c2c_store/c2c_confind_tbeta/final_twin_s29000.ckpt; C2C_MD5=1f5693e773b05ac643a33b9a25ebec9f
: "${MODE:?tri|c2c}" "${CHUNK:?00|01}"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1 ALIGN_W=0.3 MLM_W=0.3
export SYNTH_INDEX_DIR=$S/synth_index_confind_v1 SIGMA_DATA=16.0 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID}; mkdir -p "$TMPDIR" "$B"; trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
if [ "$MODE" = tri ]; then
  [ "$(md5sum < "$TRI" | cut -d' ' -f1)" = "$TRI_MD5" ] || { echo "FATAL: tri md5"; exit 75; }
  OUT=$B/tri_c$CHUNK
  [ -e "$OUT" ] && { echo "FATAL: $OUT exists"; exit 78; }
  PYTHONPATH=$REPO python scratchpad/tri_gen_eval.py --config_name training_contact_tri_full384_confindsynth_ft_v1 \
    --ema_ckpt "$TRI" --arm nonself --fixed_chain_list "$Q/chunk$CHUNK.txt" --limit_val_batches 1 --seed 0 \
    --dump_dir "$OUT" --store "$TMPDIR/store" 2>&1
  rc=$?
else
  [ "$(md5sum < "$C2C" | cut -d' ' -f1)" = "$C2C_MD5" ] || { echo "FATAL: c2c md5"; exit 75; }
  OUT=$B/B_c2c_c$CHUNK.jsonl
  [ -e "$OUT" ] && { echo "FATAL: $OUT exists"; exit 78; }
  PYTHONPATH=$REPO:$REPO/scratchpad python scratchpad/stageB_c2c_from_maps.py --ckpt "$C2C" --weights ema \
    --dataset pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828 --chains "$Q/chunk${CHUNK}_q.txt" \
    --n_seeds 1 --out "$OUT" --label "controlB_c2c_c$CHUNK" --maps_dir "$S/stageA/t7qn_cf_c$CHUNK" 2>&1
  rc=$?
fi
echo "CONTROLB_${MODE}_c${CHUNK}_RC=$rc"
exit $rc
