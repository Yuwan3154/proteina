#!/bin/bash
#SBATCH --job-name=synthidxcf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/synth_index_confind_v1/logs/%x-%A_%a.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/synth_index_confind_v1/logs/%x-%A_%a.err
# ConFind twin of synth_index_v4 (user 2026-09-23: pure ConFind end to end). Same pool, band, alias and part
# split as synth_index_v4.sbatch; only the contact definition changes (Frame2ConFind >= 0.01).
#   MODE=f2c_smoke  first LIMIT chains of part 0 + NATIVE_CHECK natives vs stored contact_map_confind (GPU)
#   MODE=f2c        template maps for one part (array task = part; GPU any arch)
#   MODE=smoke      ConFind build of the first LIMIT chains of part 0 + merge (CPU)
#   MODE=build      ConFind build of one part (array; CPU)
#   MODE=merge      merge the N_PARTS parts (CPU)
# Pass --gres / --partition / cpus / mem / time at submit (GPU for f2c*, CPU otherwise).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
REPO="${REPO:-$S/proteina_tri}"
OUT="${OUT_DIR:-$S/synth_index_confind_v1}"
MAPS="${MAPS_DIR:-$S/f2c_template_maps}"
DATA=/orcd/pool/006/chenxiou/proteina/data/pdb_train
POOL=$S/t2_pool_auth
POOL_BAND=$S/t2_pool_auth_band.npz
ALIAS=$S/synth_alias/auth_key.tsv
N_PARTS="${N_PARTS:-60}"
MODE="${MODE:?MODE}"
mkdir -p "$OUT/parts" "$OUT/logs" "$MAPS"
cd "$REPO" && git log --oneline -1
SEL=(--base-index "$DATA/topology_index.pt" --processed-dir "$DATA/processed" --manifest "$DATA/shard_manifest.json"
     --templates "$POOL:$POOL_BAND" --chain-alias "$ALIAS" --n-parts "$N_PARTS")
PY="python -m proteinfoundation.utils.precompute_synthetic_topology_index"
case "$MODE" in
  f2c_smoke|f2c)
    # sm120 venv: every GPU arch; f2c needs no pydssp
    if [ "${CPU:-0}" = 1 ]; then
      CC=9.0   # CPU run: activate.sh requires a value, it steers nothing here
    else
      CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
      [ -n "$CC" ] && [ "$(echo "$CC" | wc -l)" -eq 1 ] || { echo "FATAL: need exactly one GPU arch, got: '$CC'"; exit 79; }
    fi
    export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$CC"
    source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
    source /home/chenxiou/cue-openfold-env-sm120/activate.sh
    export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO
    [ "${CPU:-0}" = 1 ] || nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ "$MODE" = f2c_smoke ]; then
      $PY f2c "${SEL[@]}" --part 0 --limit "${LIMIT:-20}" --native-check "${NATIVE_CHECK:-100}" --native-batch "${NATIVE_BATCH:-1}" --amp-dtype "${AMP:-bf16}" --maps-out "$OUT/smoke_maps${SMOKE_TAG:-}"
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
    B=(--contact-source confind --workers "${SLURM_CPUS_PER_TASK:-32}" --usalign /home/chenxiou/.local/bin/USalign)
    if [ "$MODE" = smoke ]; then
      $PY build "${SEL[@]}" "${B[@]}" --f2c-maps "$OUT/smoke_maps" --part 0 --limit "${LIMIT:-20}" --selftest --out-dir "$OUT/smoke_parts"
      RC=$?
      [ $RC -eq 0 ] && $PY merge --parts-dir "$OUT/smoke_parts" --n-parts 1 --out "$OUT/smoke_index.pt" \
          --eligible-out "$OUT/smoke_eligible.txt" --tm-range 0.5 0.9
      RC=$?
    elif [ "$MODE" = build ]; then
      $PY build "${SEL[@]}" "${B[@]}" --f2c-maps "$MAPS" --part "${SLURM_ARRAY_TASK_ID:?array task id}" --out-dir "$OUT/parts"
      RC=$?
    else
      $PY merge --parts-dir "$OUT/parts" --n-parts "$N_PARTS" --out "$OUT/topology_index_confind_synth.pt" \
          --eligible-out "$OUT/eligible_ids_tm0.5-0.9.txt" --tm-range 0.5 0.9
      RC=$?
    fi
    ;;
  *) echo "unknown MODE=$MODE"; RC=2;;
esac
echo "SYNTHIDXCF_EXIT=$RC"
exit $RC
