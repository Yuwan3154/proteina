#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# ONE Stage-A pass (Directive B, user 2026-09-22): one tri model x one reference arm x one condition,
# on a leak-free chain list, 1 sample per chain, seeded so every pass sees the same diffusion noise.
# Env block copied from tri_2x2_clean.sbatch (the harness's validated run).
#   MODEL=old|new  ARM=self|nonself|mask  REGIME=single|variable  ZERO_CA=0|1  CHAINS=<list>
#   CKPT=<frozen weights>  LABEL=<pass name>  SEED (default 0)
# ⛔ Each model runs in ITS OWN checkout (SSE alphabet 65 vs 44). ⛔ One fresh dump dir per pass.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1
export SYNTH_INDEX_DIR=$S/synth_index_v4 ALIGN_W=0.3 MLM_W=0.3
export TMPDIR=$S/.tmp
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "${MODEL:?}" in
  old) REPO=$S/proteina_cmhier; CFG=training_contact_tri_full384_v1 ;;
  new) REPO=$S/proteina_tri;    CFG=training_contact_tri_full384_cb8synth_v1 ;;
esac
[ -f "${CHAINS:?}" ] || { echo "FATAL: no chain list $CHAINS"; exit 77; }
[ -f "${CKPT:?}" ] || { echo "FATAL: no checkpoint $CKPT"; exit 77; }
DUMP=$S/stageA/${LABEL:?}
[ -e "$DUMP" ] && { echo "FATAL: $DUMP exists -- one fresh dir per pass"; exit 78; }
echo "[pass] $LABEL model=$MODEL arm=${ARM:?} regime=${REGIME:-single} zero_ca=${ZERO_CA:-0} seed=${SEED:-0}"
echo "[pass] chains=$CHAINS ($(grep -c . "$CHAINS")) ckpt=$CKPT md5=$(md5sum < "$CKPT" | cut -c1-12)"
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h"
PYTHONPATH="$REPO" python scratchpad/tri_gen_eval.py \
    --config_name "$CFG" --ema_ckpt "$CKPT" --arm "$ARM" --mask_regime "${REGIME:-single}" \
    --fixed_chain_list "$CHAINS" --limit_val_batches 1 --seed "${SEED:-0}" \
    --dump_dir "$DUMP" --store "$TMPDIR/stageA_store_${SLURM_JOB_ID}" \
    $( [ "${ZERO_CA:-0}" = 1 ] && echo --zero_ca_features )
rc=$?
echo "PASS_RC[$LABEL]=$rc"
# ⛔ Verify the ARTEFACT: every listed chain must have produced exactly one sample.
python - "$DUMP" "$CHAINS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/samples.jsonl")]
want = [l.split()[0] for l in open(sys.argv[2]) if l.strip()]
got = sorted(r["stem"] for r in rows)
print(f"[verify] samples {len(rows)}  chains listed {len(want)}  distinct sampled {len(set(got))}")
assert sorted(want) == got, "sampled chains != listed chains"
print("[verify] OK: one sample per listed chain")
PY
echo "VERIFY_RC=$?"
exit $rc
