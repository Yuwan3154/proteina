#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/toporef_xrun/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/toporef_xrun/logs/%x-%j.err
# Cross-run TopoRef comparison, stage C sampling pass: ONE model x ONE arm x ONE chain chunk (stageA_pass.sbatch with
# CB-8 / T8 / T8b entries). Each model runs in its own checkout at its training commit, its own experiment config,
# against its own EVAL index (SYNTH_INDEX_DIR); seed_everything(SEED) before validate, the same chunk lists for every
# model and arm (paired samples, T7 precedent); align-head first/last calls dumped (DUMP_ALIGN).
#   MODEL=cb8|t8|t8b  ARM=self|equal|indel  CHUNK=<chunk id>  (SEED default 0)
# self = the model's own native reference (tri_gen_eval --arm self) served from the indel eval index (all chains);
# equal / indel = --arm nonself on that arm's eval index (exactly one template per chain -> deterministic).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
C=/orcd/compute/so3/002/chenxi
X=$S/toporef_xrun
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S || { echo "FATAL: not an L40S (user 2026-10-28)"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1
export ALIGN_W=0.3 MLM_W=0.3
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
# compile caches on so3/002, never scratch (user 2026-10-11); one dir per arch, pruned after the comparison
export TORCHINDUCTOR_CACHE_DIR=$C/t8_compile_cache/xrun_sm${TORCH_CUDA_ARCH_LIST/./}/inductor
export TRITON_CACHE_DIR=$C/t8_compile_cache/xrun_sm${TORCH_CUDA_ARCH_LIST/./}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "${MODEL:?}" in
  cb8) REPO=$S/xrun_repo_cb8; CFG=training_contact_tri_full384_cb8synth_v1; CKPT=$X/ckpt/cb8/cb8_chk_epoch305_step123500-EMA.ckpt ;;
  t8)  REPO=$S/xrun_repo_t8;  CFG=training_contact_tri_t8_confind_v1;        CKPT=$X/ckpt/t8/t8_last-EMA_step125775.ckpt ;;
  t8b) REPO=$S/xrun_repo_t8b; CFG=training_contact_tri_t8b_confind_v1;       CKPT=${T8B_CKPT:?T8B_CKPT = the captured ckpt} ;;
  *) echo "FATAL: MODEL=$MODEL"; exit 77 ;;
esac
case "${ARM:?}" in
  self)  IDX=$X/index/${MODEL}_indel; TARM=self ;;
  equal) IDX=$X/index/${MODEL}_equal; TARM=nonself ;;
  indel) IDX=$X/index/${MODEL}_indel; TARM=nonself ;;
  *) echo "FATAL: ARM=$ARM"; exit 77 ;;
esac
export SYNTH_INDEX_DIR=$IDX
CHAINS=$X/chunks/${ARM}_c${CHUNK:?}.txt
LABEL=${MODEL}_${ARM}_c${CHUNK}
DUMP=$X/dumps/$LABEL
[ -f "$CHAINS" ] || { echo "FATAL: no chain list $CHAINS"; exit 77; }
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint $CKPT"; exit 77; }
[ -e "$DUMP" ] && { echo "FATAL: $DUMP exists -- one fresh dir per pass"; exit 78; }
mkdir -p "$X/dumps"
echo "[pass] $LABEL model=$MODEL arm=$ARM ($TARM) seed=${SEED:-0} SYNTH_INDEX_DIR=$SYNTH_INDEX_DIR REPO=$REPO"
echo "[pass] chains=$CHAINS ($(grep -c . "$CHAINS")) ckpt=$CKPT md5=$(md5sum < "$CKPT" | cut -c1-12)"
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h"
PYTHONPATH="$REPO" python scratchpad/tri_gen_eval.py \
    --config_name "$CFG" --ema_ckpt "$CKPT" --arm "$TARM" \
    --fixed_chain_list "$CHAINS" --limit_val_batches 1 --seed "${SEED:-0}" \
    --dump_dir "$DUMP" --store "$TMPDIR/store" --dump_align
rc=$?
echo "PASS_RC[$LABEL]=$rc"
python - "$DUMP" "$CHAINS" "$TARM" <<'PY'
import collections, json, sys
import numpy as np
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/samples.jsonl")]
want = collections.Counter(l.split()[0] for l in open(sys.argv[2]) if l.strip())
got = collections.Counter(r["stem"] for r in rows)
print(f"[verify] samples {len(rows)}  listed {sum(want.values())}  chains {len(want)}")
assert got == want, "sampled (chain, count) != listed"
refs = collections.defaultdict(set)
for r in rows:
    assert r.get("ref_id"), f"{r['stem']} s{r['sample_index']}: no reference id recorded (unconditioned?)"
    refs[r["stem"]].add(r["ref_id"])
    z = np.load(f"{sys.argv[1]}/{r['file']}")
    # MASK_TOKEN = 1: an all-MASK reference = the chain sampled unconditioned
    assert "he_tokens" in z.files and (z["he_tokens"] != 1).any(), f"{r['stem']} s{r['sample_index']}: unconditioned"
assert all(len(v) == 1 for v in refs.values()), "a chain got more than one reference"
if sys.argv[3] == "self":
    assert all(v == {k} for k, v in refs.items()), "self arm: reference is not the chain itself"
print("[verify] OK")
PY
vrc=$?
echo "VERIFY_RC=$vrc"
[ "$rc" -ne 0 ] && exit $rc
exit $vrc
