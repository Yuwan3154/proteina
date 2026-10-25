#!/bin/bash
#SBATCH --job-name=t8bidx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/t8b_index_logs/%x-%A_%a.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/t8b_index_logs/%x-%A_%a.err
# T8b synthetic-reference index (twin of t8_index.sbatch; builder --final-selection mode, see its docstring):
# templates = final_selection.npz of the t2_synth_v2 pack, REPLACED for the T8b regen list by the T8b-only pack's
# final selection (none there = none); TM 0.5-0.9 + clash gate <= 4 pairs < 3.4 A, no cap (user 2026-10-20 "Rebuild
# from final_selection"); natives from tri_t8b_512.pack; everything else as T8 (ConFind contacts, loops as elements,
# loop axis zero, proline donor mask). Outputs on /orcd/compute/so3/002 (file counts).
#   MODE=f2c_plan   CPU: count maps reusable from t8_f2c_maps vs to compute (no GPU, no writes but the plan TSVs)
#   MODE=f2c_smoke  GPU: first LIMIT chains of part 0 + NATIVE_CHECK natives vs stored contact_map_confind
#   MODE=f2c        GPU: template maps for one part (array task = part)
#   MODE=smoke      CPU: build of the first LIMIT chains of part 0 against SMOKE_MAPS
#   MODE=build      CPU: build of one part (array)
#   MODE=audit      CPU: audit the N_PARTS parts (strict: exits non-zero on any failure class)
#   MODE=merge      CPU: audit (strict) then merge
# F2C_BS=1 and REUSE_T8=0 are required (decided 2026-10-25): every template map recomputed unpadded, no T8 copies
# (T8 ran batch 8, padded); f2c_plan still counts what would have been reusable, for the record.
# Pass --gres / --partition / cpus / mem / time at submit (GPU: L40S / H100 / H200 only, checked below).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
C=/orcd/compute/so3/002/chenxi
REPO="${REPO:?REPO = the Engaging checkout of branch t8b-index}"
OUT="${OUT_DIR:-$C/t8b_index}"
MAPS="${MAPS_DIR:-$C/t8b_f2c_maps}"
T8_MAPS=$C/t8_f2c_maps
DATA="${DATA_DIR:-/orcd/pool/006/chenxiou/proteina/data/pdb_train_t8b}"
PACK=$C/t2_synth_v2/pack
PRE_IDX=$PACK/tmpl.idx.npz.pre-append    # the index the T8 maps were computed from (pack before the 10-16/18 appends)
T8B_PACK=$C/t2_synth_v2_t8b/pack
NATIVE_PACK=$C/proteina_pack/tri_t8b_512.pack
CHAINS="${CHAINS:-$S/t8_template_list/template_chains.txt}"
OVERRIDE=$S/t8b_template_list/template_chains.txt
N_PARTS="${N_PARTS:-60}"
MODE="${MODE:?MODE}"
mkdir -p "$OUT/parts" "$MAPS" "$S/t8b_index_logs"
# Frame2ConFind _make_s_init computes dihedrals without the mask -> padded chains get a corrupt C-term (diag 2026-10-25)
[ "${F2C_BS:-}" = 1 ] && [ "${REUSE_T8:-}" = 0 ] || { echo "FATAL: need F2C_BS=1 REUSE_T8=0 (decided 2026-10-25)"; exit 2; }
T8B_INDEX_SHA=__SET_TO_COMMIT_SHA__   # TODO(coordinator): the commit that introduces --final-selection (set after commit)
cd "$REPO" || exit 3
git log --oneline -1
case "$T8B_INDEX_SHA" in __*) echo "FATAL: T8B_INDEX_SHA placeholder not set"; exit 3;; esac
git merge-base --is-ancestor "$T8B_INDEX_SHA" HEAD || { echo "FATAL: $REPO predates $T8B_INDEX_SHA (final-selection mode)"; exit 3; }
for f in "$PACK/final_selection.npz" "$PRE_IDX" "$T8B_PACK/final_selection.npz" "$OVERRIDE" "$CHAINS"; do
  [ -s "$f" ] || { echo "FATAL: missing $f"; exit 3; }
done
[ "$(grep -c . "$OVERRIDE")" -eq 2313 ] || { echo "FATAL: $OVERRIDE is not the 2,313-chain regen list"; exit 3; }
NOT_IN=$(comm -23 <(grep . "$OVERRIDE" | sort -u) <(grep . "$CHAINS" | sort -u) | wc -l)
[ "$NOT_IN" -eq 0 ] || { echo "FATAL: $NOT_IN override chains are not in $CHAINS"; exit 3; }
SEL=(--chain-list "$CHAINS" --processed-dir "$DATA/processed" --manifest "$DATA/shard_manifest.json"
     --template-pack "$PACK" --final-selection "$PACK/final_selection.npz" --template-pack-tag t2_synth_v2
     --override-pack "$T8B_PACK" --override-selection "$T8B_PACK/final_selection.npz" --override-chains "$OVERRIDE"
     --override-pack-tag t2_synth_v2_t8b --tm-min 0.5 --tm-max 0.9)
REUSE=(--reuse-maps "$T8_MAPS" --reuse-pack-idx "$PRE_IDX")
PY="python -m proteinfoundation.utils.precompute_synthetic_topology_index"
cpu_env() {
  module load miniforge
  set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null; conda activate cue_openfold; set -u
  export CUE_HOST=engaging DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1
  export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
}
audit() {
  python scratchpad/t8_audit_parts.py --parts-dir "$OUT/parts" --n-parts "$N_PARTS" --chain-list "$CHAINS" \
      --override-chains "$OVERRIDE" --template-pack-tag t2_synth_v2 --override-pack-tag t2_synth_v2_t8b --strict
}
case "$MODE" in
  f2c_plan)
    cpu_env
    $PY f2c-plan "${SEL[@]}" "${REUSE[@]}" --part 0 --n-parts 1 --maps-out "$MAPS" --plan-out "$OUT/f2c_plan"
    RC=$?
    ;;
  f2c_smoke|f2c)
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
    echo "$GPU" | grep -Eq 'L40S|H100|H200' && ! echo "$GPU" | grep -Eq 'A100|A40|L4$' \
      || { echo "FATAL: GPU '$GPU' not in L40S / H100 / H200 (user 2026-10-13)"; exit 79; }
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
    [ -n "$CC" ] && [ "$(echo "$CC" | wc -l)" -eq 1 ] || { echo "FATAL: need exactly one GPU arch, got: '$CC'"; exit 79; }
    export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$CC"
    source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
    source /home/chenxiou/cue-openfold-env-sm120/activate.sh
    export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ "$MODE" = f2c_smoke ]; then
      $PY f2c "${SEL[@]}" --n-parts "$N_PARTS" --part 0 --limit "${LIMIT:-20}" \
          --native-check "${NATIVE_CHECK:-100}" --native-batch "${NATIVE_BATCH:-$F2C_BS}" --amp-dtype "${AMP:-bf16}" \
          --batch-size "$F2C_BS" --maps-out "$OUT/smoke_maps${SMOKE_TAG:-}"
    else
      $PY f2c "${SEL[@]}" --n-parts "$N_PARTS" --part "${SLURM_ARRAY_TASK_ID:?array task id}" \
          --batch-size "$F2C_BS" --maps-out "$MAPS"
    fi
    RC=$?
    ;;
  smoke|build|audit|merge)
    cpu_env
    [ -s "$NATIVE_PACK" ] && [ -s "$NATIVE_PACK.idx.npz" ] && [ ! -e "$NATIVE_PACK.partial" ] \
      || { echo "FATAL: native pack $NATIVE_PACK not finished"; exit 3; }
    B=(--contact-source confind --element-types 0,1,2 --loop-axis zero --proline-donor-mask --native-pack "$NATIVE_PACK"
       --workers "${SLURM_CPUS_PER_TASK:-32}" --usalign /home/chenxiou/.local/bin/USalign
       --max-ca-clashes 4 --ca-clash-a 3.4 --f2c-batch-size "$F2C_BS")  # gate: user 2026-10-13 "3.4 A" + "<=4"
    if [ "$MODE" = smoke ]; then
      $PY build "${SEL[@]}" "${B[@]}" --n-parts "$N_PARTS" --f2c-maps "${SMOKE_MAPS:-$OUT/smoke_maps}" --part 0 \
          --limit "${LIMIT:-20}" --out-dir "$OUT/smoke_parts"
      RC=$?
    elif [ "$MODE" = build ]; then
      $PY build "${SEL[@]}" "${B[@]}" --n-parts "$N_PARTS" --f2c-maps "$MAPS" \
          --part "${SLURM_ARRAY_TASK_ID:?array task id}" --out-dir "$OUT/parts"
      RC=$?
    elif [ "$MODE" = audit ]; then
      audit
      RC=$?
    else
      audit || { echo "FATAL: audit failed, not merging"; echo "T8BIDX_EXIT=4"; exit 4; }
      $PY merge --parts-dir "$OUT/parts" --n-parts "$N_PARTS" --out "$OUT/topology_index_confind_synth_t8b.pt" \
          --eligible-out "$OUT/eligible_ids_tm0.5-0.9.txt" --tm-range 0.5 0.9
      RC=$?
    fi
    ;;
  *) echo "unknown MODE=$MODE"; RC=2;;
esac
echo "T8BIDX_EXIT=$RC"
exit $RC
