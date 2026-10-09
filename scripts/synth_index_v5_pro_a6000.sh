#!/bin/bash
# v5 synthetic topology index on the A6000 (user 2026-10-09): v4's inputs (pool rebuilt from its manifest, same band
# index, alias, base index) + --proline-donor-mask; natives from the training pack (--native-pack). Runs from the clone
# ~/proteina_sse1d_idx, never the training checkout; nice 19 next to the training runs.
#   bash scripts/synth_index_v5_pro_a6000.sh pool|control|pro0|build|merge [first_part last_part]
set -uo pipefail
MODE=$1
IN=$HOME/sse1d_idx/in                         # meta.tar + engaging_trees.tar extracted here (absolute Engaging paths)
E=$IN/orcd/scratch/orcd/011/chenxiou
OUT=$HOME/sse1d_idx/v5_pro
POOL=$HOME/sse1d_idx/t2_pool_auth
PACK=$HOME/sse1d_data/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack
WORKERS=${WORKERS:-32}
PY="nice -n 19 $HOME/miniconda3/bin/conda run --no-capture-output -p $HOME/miniconda3/envs/proteina_sse1d python"
cd "$HOME/proteina_sse1d_idx" || exit 2
: "${EXPECT_COMMIT:?}"
case "$(git rev-parse HEAD)" in "$EXPECT_COMMIT"*) ;; *) echo "FATAL: clone is not at $EXPECT_COMMIT"; exit 69;; esac
git diff --quiet HEAD || { echo "FATAL: clone has local changes"; exit 68; }
export PYTHONPATH=$HOME/proteina_sse1d_idx CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/parts" "$OUT/control" "$OUT/pro0"
echo "commit $(git log --oneline -1) mode $MODE $(date)"
COMMON=(--base-index "$IN/orcd/pool/006/chenxiou/proteina/data/pdb_train/topology_index.pt" --processed-dir /nonexistent
        --native-pack "$PACK" --templates "$POOL:$E/t2_pool_auth_band.npz" --chain-alias "$E/synth_alias/auth_key.tsv"
        --n-parts 60 --workers "$WORKERS" --usalign "$HOME/.local/bin/USalign")
B="$PY -m proteinfoundation.utils.precompute_synthetic_topology_index"
case "$MODE" in
  pool)
    $PY scripts/rebuild_pool_from_manifest.py --manifest "$E/pool_manifest.tsv" --dest "$POOL" \
      --remap "/orcd/compute/so3/002/chenxi/of_run/pp1c_work/templates_band=$HOME/sse1d_idx/t2_templates_band" \
      --remap "/orcd/scratch/orcd/011/chenxiou/t2_regen/templates_band=$E/t2_regen/templates_band" \
      --remap "/orcd/scratch/orcd/011/chenxiou/t2_extra/templates_band=$E/t2_extra/templates_band"
    RC=$?;;
  control)
    $B build "${COMMON[@]}" --part 0 --out-dir "$OUT/control" && \
    $PY scripts/compare_synth_parts_dssp.py "$OUT/control/part_0000.pt" "$E/synth_index_v4/parts/part_0000.pt" \
      --expect identical --new-skips "$OUT/control/skips_0000.tsv"
    RC=$?;;
  pro0)
    $B build "${COMMON[@]}" --part 0 --proline-donor-mask --out-dir "$OUT/pro0" && \
    $PY scripts/compare_synth_parts_dssp.py "$OUT/pro0/part_0000.pt" "$E/synth_index_v4/parts/part_0000.pt" \
      --expect dssp --new-skips "$OUT/pro0/skips_0000.tsv"
    RC=$?;;
  build)
    RC=0
    for p in $(seq "${2:-0}" "${3:-59}"); do
      [ -e "$OUT/parts/skips_$(printf %04d "$p").tsv" ] && continue  # restartable: skips are written after the part
      $B build "${COMMON[@]}" --part "$p" --proline-donor-mask --out-dir "$OUT/parts" || { RC=$?; break; }
    done;;
  merge)
    $B merge --parts-dir "$OUT/parts" --n-parts 60 --out "$OUT/topology_index_cb8_synth.pt" \
      --eligible-out "$OUT/eligible_ids_tm0.5-0.9.txt" --tm-range 0.5 0.9
    RC=$?;;
  *) echo "unknown MODE"; RC=2;;
esac
echo "SYNTHIDX_EXIT=$RC $(date)"
exit $RC
