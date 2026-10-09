#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc CONTROL B (GPU, L40S): the d36tc harness reproduces T7-NEWEST records. Per stem (1dmx_A = first of
# t7/quick/chunk00, 1csb_B = first of chunk01 => same tri RNG position; first in the sorted _q list => c2c chain_index 0):
#   tri  = tri_gen_eval, T7 dataset + synth_index_confind_v1, --arm nonself, seed 0, cfft step 48150 (= the T7-NEWEST tri)
#   c2c  = stageB_c2c_from_maps, the T7 c2c dataset, twin s29000 EMA, 1 draw/map, SIGMA_DATA 16
#   then d36tc_controlB_compare.py against $S/stageA/t7qn_cf_c0{0,1} and $S/t7/newest_out/B_t7qn_cf29000_c0{0,1}.jsonl.
# Env block = stageA_pass.sbatch (cfft) / stageB_c2c_from_maps.sbatch. Partition/gres/time/name on sbatch.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
B=$P/controlB
REPO="${REPO:-$S/proteina_d36tc}"
TRI=$S/stageA/ckpt/cfft_tri_last-EMA_step48150.ckpt;      TRI_MD5=6fb81c1dffecff820f7f9c8e572d83be
C2C=$S/c2c_store/c2c_confind_tbeta/final_twin_s29000.ckpt; C2C_MD5=1f5693e773b05ac643a33b9a25ebec9f
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
[ "$(md5sum < "$TRI" | cut -d' ' -f1)" = "$TRI_MD5" ] || { echo "FATAL: tri md5"; exit 75; }
[ "$(md5sum < "$C2C" | cut -d' ' -f1)" = "$C2C_MD5" ] || { echo "FATAL: c2c md5"; exit 75; }
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
rc=0
for STEM in 1dmx_A 1csb_B; do
  [ -e "$B/tri_$STEM" ] && { echo "FATAL: $B/tri_$STEM exists"; exit 78; }
  printf "$STEM\n%.0s" 1 2 3 4 5 6 7 8 > "$B/chains_${STEM}_x8.txt"
  echo "$STEM" > "$B/chains_${STEM}.txt"
  PYTHONPATH=$REPO python scratchpad/tri_gen_eval.py --config_name training_contact_tri_full384_confindsynth_ft_v1 \
    --ema_ckpt "$TRI" --arm nonself --fixed_chain_list "$B/chains_${STEM}_x8.txt" --limit_val_batches 1 --seed 0 \
    --dump_dir "$B/tri_$STEM" --store "$TMPDIR/store_$STEM" > "$B/run_tri_${STEM}.log" 2>&1 || rc=$?
  cat "$B/run_tri_${STEM}.log"
  PYTHONPATH=$REPO:$REPO/scratchpad python scratchpad/stageB_c2c_from_maps.py --ckpt "$C2C" --weights ema \
    --dataset pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828 --chains "$B/chains_${STEM}.txt" \
    --n_seeds 1 --out "$B/B_${STEM}.jsonl" --label "controlB_$STEM" --maps_dir "$B/tri_$STEM" 2>&1 || rc=$?
done
echo "RUN_RC=$rc"
PYTHONPATH=$REPO python scratchpad/d36tc_controlB_compare.py --mine "$B" \
  --pair 1dmx_A:$S/stageA/t7qn_cf_c00:$S/t7/newest_out/B_t7qn_cf29000_c00.jsonl \
  --pair 1csb_B:$S/stageA/t7qn_cf_c01:$S/t7/newest_out/B_t7qn_cf29000_c01.jsonl
crc=$?
echo "COMPARE_RC=$crc"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$crc"
