#!/bin/bash
#SBATCH --job-name=t8idx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/t8_index_logs/%x-%A_%a.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/t8_index_logs/%x-%A_%a.err
# T8 synthetic-reference index from the t2 template PACK (user 2026-10-12/13: cap 64 per chain evenly in TM value,
# TM 0.5-0.9, USalign alignment target, no sequence / length gate, ConFind contacts, loops as elements with --loop-axis
# zero, proline donor mask). Twin of synth_index_confind.sbatch; outputs on /orcd/compute/so3/002 (file counts).
#   MODE=f2c_smoke  first LIMIT chains of part 0 + NATIVE_CHECK natives at NATIVE_BATCH vs stored contact_map_confind (GPU)
#   MODE=f2c        template maps for one part (array task = part; GPU)
#   MODE=smoke      ConFind build of the first LIMIT chains of part 0 (CPU)
#   MODE=build      ConFind build of one part (array; CPU)
#   MODE=merge      merge the N_PARTS parts (CPU)
# Pass --gres / --partition / cpus / mem / time at submit.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
C=/orcd/compute/so3/002/chenxi
REPO="${REPO:-$S/proteina_t8dev}"
OUT="${OUT_DIR:-$C/t8_index}"
MAPS="${MAPS_DIR:-$C/t8_f2c_maps}"
DATA=/orcd/pool/006/chenxiou/proteina/data/pdb_train
PACK="${TEMPLATE_PACK:-$C/t2_synth_v2/pack}"
NATIVE_PACK=$C/proteina_pack/tri_t8_512.pack
CHAINS=$S/t8_template_list/template_chains.txt
N_PARTS="${N_PARTS:-60}"
MODE="${MODE:?MODE}"
mkdir -p "$OUT/parts" "$MAPS" "$S/t8_index_logs"
cd "$REPO" || exit 3
git log --oneline -1
git merge-base --is-ancestor cc05b73 HEAD || { echo "FATAL: $REPO predates cc05b73 (pack mode)"; exit 3; }
SEL=(--chain-list "$CHAINS" --processed-dir "$DATA/processed" --manifest "$DATA/shard_manifest.json"
     --template-pack "$PACK" --max-templates "${MAX_TEMPLATES:-64}" --tm-min 0.5 --tm-max 0.9 --n-parts "$N_PARTS")
# MAX_TEMPLATES > the most in-range templates of any chain (161) = every in-range template: the f2c superset that any
# later gate + cap-64 selection draws from (maps are keyed per chain and draw, so f2c and build may differ in the cap)
PY="python -m proteinfoundation.utils.precompute_synthetic_topology_index"
case "$MODE" in
  f2c_smoke|f2c)
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
    [ -n "$CC" ] && [ "$(echo "$CC" | wc -l)" -eq 1 ] || { echo "FATAL: need exactly one GPU arch, got: '$CC'"; exit 79; }
    export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$CC"
    source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
    source /home/chenxiou/cue-openfold-env-sm120/activate.sh
    export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ "$MODE" = f2c_smoke ]; then
      $PY f2c "${SEL[@]}" --part 0 --limit "${LIMIT:-20}" --native-check "${NATIVE_CHECK:-100}" \
          --native-batch "${NATIVE_BATCH:-8}" --amp-dtype "${AMP:-bf16}" --maps-out "$OUT/smoke_maps${SMOKE_TAG:-}"
    else
      $PY f2c "${SEL[@]}" --part "${SLURM_ARRAY_TASK_ID:?array task id}" --maps-out "$MAPS"
    fi
    RC=$?
    ;;
  smoke|build|merge)
    module load miniforge
    set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null; conda activate cue_openfold; set -u
    export CUE_HOST=engaging DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1
    export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    B=(--contact-source confind --element-types 0,1,2 --loop-axis zero --proline-donor-mask --native-pack "$NATIVE_PACK"
       --workers "${SLURM_CPUS_PER_TASK:-32}" --usalign /home/chenxiou/.local/bin/USalign
       --max-ca-clashes 4 --ca-clash-a 3.4)  # user 2026-10-13: "gate at 3.4 A" + "<=4" (before the cap)
    if [ "$MODE" = smoke ]; then
      $PY build "${SEL[@]}" "${B[@]}" --f2c-maps "${SMOKE_MAPS:-$OUT/smoke_maps}" --part 0 --limit "${LIMIT:-20}" \
          --out-dir "$OUT/smoke_parts"
      RC=$?
    elif [ "$MODE" = build ]; then
      $PY build "${SEL[@]}" "${B[@]}" --f2c-maps "$MAPS" --part "${SLURM_ARRAY_TASK_ID:?array task id}" --out-dir "$OUT/parts"
      RC=$?
    else
      $PY merge --parts-dir "$OUT/parts" --n-parts "$N_PARTS" --out "$OUT/topology_index_confind_synth_t8.pt" \
          --eligible-out "$OUT/eligible_ids_tm0.5-0.9.txt" --tm-range 0.5 0.9
      RC=$?
    fi
    ;;
  *) echo "unknown MODE=$MODE"; RC=2;;
esac
echo "T8IDX_EXIT=$RC"
exit $RC
