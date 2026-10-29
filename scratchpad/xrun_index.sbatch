#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/toporef_xrun/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/toporef_xrun/logs/%x-%j.err
# Cross-run TopoRef comparison, stage B: per-model EVAL indexes over the shared (chain, template) pairs, each built with
# THAT model's production builder settings (synth_index.sbatch / t8_index.sbatch / t8b_index.sbatch), standardised with
# its production index's stats (merge --stats-from).
#   MODE=prep     CPU: xrun_select.py (indel arm + chain set) -> xrun_cb8pool.py resolve (v4's own pool / alias rule) ->
#                 rsync the T2 files from SuperCloud ~/pp1c_work/templates_band -> xrun_cb8pool.py pick (equal arm, controls)
#   MODE=f2c      GPU: Frame2ConFind maps of both arms' templates at batch size 1 (user 2026-10-28) -> $MAPS/{equal,indel}
#   MODE=build    CPU: MODEL=cb8|t8|t8b ARM=equal|indel -> $X/index/<model>_<arm>/ (build + merge + coverage check)
#   MODE=control  CPU: MODEL=cb8|t8|t8b: the control chains rebuilt with the PRODUCTION args, torch.equal vs production
#   MODE=compare  CPU: every eval index's natives vs production; t8b indel template rows vs production
# Pass --gres / --partition / cpus / mem / time at submit.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
C=/orcd/compute/so3/002/chenxi
X=$S/toporef_xrun
SEL=$X/select
POOL=$X/cb8pool
REPO=$S/proteina_xrun
MAPS=$C/xrun_f2c_maps
T2=$C/t2_synth_v2/pack
PP1C=$C/xrun_pp1c/templates_band            # copy of SuperCloud ~/pp1c_work/templates_band (the files used only)
XPOOL=$C/xrun_pool                          # private auth-keyed symlink pool (xrun_cb8pool.py pick)
ALIAS=$S/synth_alias/auth_key.tsv           # v4's --chain-alias (synth_index_confind.sbatch: "same ... alias as v4")
CTL=$X/cb8pool/control_chains.txt
USALIGN=/home/chenxiou/.local/bin/USalign
PDB=/orcd/pool/006/chenxiou/proteina/data
PY="python -m proteinfoundation.utils.precompute_synthetic_topology_index"
MODE="${MODE:?MODE}"
cd "$REPO" || exit 3
git log --oneline -1
cpu_env() {
  module load miniforge
  set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null; conda activate cue_openfold; set -u
  export CUE_HOST=engaging DATA_PATH=$PDB SKIP_CONFIND_PRECOMPUTE=1
  export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
}
model_args() {  # production settings per model
  case "${MODEL:?MODEL}" in
    cb8)  DATA=$PDB/pdb_train; PROD=$S/synth_index_v4/topology_index_cb8_synth.pt
          B=(--usalign "$USALIGN"); GATE=()                           # synth_index.sbatch: CB-8 contacts, H/E, no gate
          ;;
    t8)   DATA=$PDB/pdb_train; PROD=$C/t8_index/topology_index_confind_synth_t8.pt
          B=(--contact-source confind --element-types 0,1,2 --loop-axis zero --proline-donor-mask
             --native-pack "$C/proteina_pack/tri_t8_512.pack" --usalign "$USALIGN"); GATE=(--max-ca-clashes 4 --ca-clash-a 3.4)
          ;;
    t8b)  DATA=$PDB/pdb_train_t8b; PROD=$C/t8b_index_v2/topology_index_confind_synth_t8b.pt
          B=(--contact-source confind --element-types 0,1,2 --loop-axis zero --proline-donor-mask
             --native-pack "$C/proteina_pack/tri_t8b_512_v2.pack" --usalign "$USALIGN"); GATE=(--max-ca-clashes 4 --ca-clash-a 3.4)
          ;;
    *) echo "FATAL: MODEL=$MODEL"; exit 2;;
  esac
  B+=(--workers "${SLURM_CPUS_PER_TASK:-16}")
  DS=(--processed-dir "$DATA/processed" --manifest "$DATA/shard_manifest.json")
}
EQ_TPL=(--templates "$XPOOL:$POOL/band_equal.npz" --chain-alias "$ALIAS")
IN_TPL=(--template-pack "$T2" --final-selection "$SEL/sel_indel.npz" --template-pack-tag t2_synth_v2)
case "$MODE" in
  prep)
    cpu_env
    [ -e "$SEL" ] && { echo "FATAL: $SEL exists"; exit 78; }
    python scratchpad/xrun_select.py --inventory "$X/inventory" --out-dir "$SEL" || exit 4
    python scratchpad/xrun_cb8pool.py resolve --chains "$SEL/chains.txt" --out-dir "$POOL" || exit 5
    mkdir -p "$PP1C"
    rsync -a --files-from="$POOL/sc_relpaths.txt" SuperCloud:pp1c_work/templates_band/ "$PP1C/" || exit 6
    echo "[prep] T2 files: listed $(grep -c . "$POOL/sc_relpaths.txt"), present $(find "$PP1C" -name '*.npz' | wc -l)"
    python scratchpad/xrun_cb8pool.py pick --control-candidates "$SEL/control_candidates.txt" --sc-copy "$PP1C" \
        --xpool "$XPOOL" --out-dir "$POOL"
    RC=$?
    ;;
  f2c)
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
    echo "$GPU" | grep -Eq 'L40S|H100|H200' || { echo "FATAL: GPU '$GPU' not L40S / H100 / H200"; exit 79; }
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
    [ -n "$CC" ] && [ "$(echo "$CC" | wc -l)" -eq 1 ] || { echo "FATAL: need exactly one GPU arch, got: '$CC'"; exit 79; }
    export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$CC"
    source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
    source /home/chenxiou/cue-openfold-env-sm120/activate.sh
    export DATA_PATH=$PDB SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO
    DS=(--processed-dir "$PDB/pdb_train_t8b/processed" --manifest "$PDB/pdb_train_t8b/shard_manifest.json")
    RC=0
    # equal arm: no TM flags (production CB-8 passed none; the band holds exactly the one picked in-window rung)
    $PY f2c --chain-list "$POOL/chains_equal.txt" "${DS[@]}" "${EQ_TPL[@]}" \
        --n-parts 1 --part 0 --batch-size 1 --maps-out "$MAPS/equal" || RC=$?
    $PY f2c --chain-list "$SEL/chains.txt" "${DS[@]}" "${IN_TPL[@]}" --tm-min 0.5 --tm-max 0.9 \
        --n-parts 1 --part 0 --batch-size 1 --maps-out "$MAPS/indel" || RC=$?
    ;;
  build)
    cpu_env; model_args
    case "${ARM:?ARM}" in equal|indel) ;; *) echo "FATAL: ARM=$ARM"; exit 2;; esac
    O=$X/index/${MODEL}_$ARM
    [ -e "$O" ] && { echo "FATAL: $O exists (one fresh dir per build)"; exit 78; }
    MAPARGS=()
    if [ "$ARM" = equal ]; then
      CL=$POOL/chains_equal.txt
      [ "$MODEL" != cb8 ] && MAPARGS=(--f2c-maps "$MAPS/equal")
      $PY build --chain-list "$CL" "${DS[@]}" "${EQ_TPL[@]}" --n-parts 1 --part 0 \
          "${B[@]}" "${MAPARGS[@]}" --out-dir "$O/parts"
      RC=$?
    else
      CL=$SEL/chains.txt
      [ "$MODEL" != cb8 ] && MAPARGS=(--f2c-maps "$MAPS/indel" --f2c-batch-size 1)
      $PY build --chain-list "$CL" "${DS[@]}" "${IN_TPL[@]}" --tm-min 0.5 --tm-max 0.9 --n-parts 1 --part 0 \
          "${B[@]}" "${GATE[@]}" "${MAPARGS[@]}" --out-dir "$O/parts"
      RC=$?
    fi
    if [ $RC -eq 0 ]; then
      F=$(basename "$PROD")
      $PY merge --parts-dir "$O/parts" --n-parts 1 --out "$O/$F" --eligible-out "$O/eligible_ids_tm0.5-0.9.txt" \
          --tm-range 0.5 0.9 --stats-from "$PROD"
      RC=$?
      echo "[build] skips:"; cat "$O/parts/skips_0000.tsv"
    fi
    if [ $RC -eq 0 ]; then
      python - "$O/$F" "$CL" <<'PYEOF' || RC=5
import sys, torch
idx = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
want = [l.strip() for l in open(sys.argv[2]) if l.strip()]
ids, nat = list(idx["ids"]), idx["row_is_native"].tolist()
natives = {s for s, n in zip(ids, nat) if n}
tpl = {}
for s, n in zip(ids, nat):
    if not n:
        tpl[s.split("@")[0]] = tpl.get(s.split("@")[0], 0) + 1
miss = [s for s in want if s not in natives or tpl.get(s, 0) != 1]
print(f"[build-check] {len(want)} chains, {len(natives)} natives, {sum(tpl.values())} template rows; "
      f"chains without exactly native + 1 template: {len(miss)} {miss[:10]}")
sys.exit(1 if miss else 0)
PYEOF
    fi
    ;;
  control)
    cpu_env; model_args
    O=$X/control/$MODEL
    [ -e "$O" ] && { echo "FATAL: $O exists"; exit 78; }
    CL=$CTL
    case "$MODEL" in
      cb8)  $PY build --chain-list "$CL" "${DS[@]}" --templates "$XPOOL:$POOL/band_control.npz" --chain-alias "$ALIAS" \
                --n-parts 1 --part 0 "${B[@]}" --out-dir "$O/parts";;
      t8)   $PY build --chain-list "$CL" "${DS[@]}" --template-pack "$T2" --max-templates 64 --tm-min 0.5 --tm-max 0.9 \
                --n-parts 1 --part 0 "${B[@]}" "${GATE[@]}" --f2c-maps "$C/t8_f2c_maps" --out-dir "$O/parts";;
      t8b)  $PY build --chain-list "$CL" "${DS[@]}" --template-pack "$T2" --final-selection "$T2/final_selection.npz" \
                --template-pack-tag t2_synth_v2 --override-pack "$C/t2_synth_v2_t8b/pack" \
                --override-selection "$C/t2_synth_v2_t8b/pack/final_selection.npz" \
                --override-chains "$S/t8b_template_list/template_chains.txt" --override-pack-tag t2_synth_v2_t8b \
                --tm-min 0.5 --tm-max 0.9 --n-parts 1 --part 0 "${B[@]}" "${GATE[@]}" --f2c-maps "$C/t8b_f2c_maps" \
                --f2c-batch-size 1 --out-dir "$O/parts";;
    esac
    RC=$?
    if [ $RC -eq 0 ]; then
      $PY merge --parts-dir "$O/parts" --n-parts 1 --out "$O/control.pt" --tm-range 0.5 0.9
      python scratchpad/xrun_index_compare.py --a "$O/control.pt" --b "$PROD" --require-all --require-b-complete \
          --min-rows "$(( 2 * $(grep -c . "$CL") ))"
      RC=$?
    fi
    ;;
  compare)
    cpu_env
    RC=0
    for MODEL in cb8 t8 t8b; do
      model_args
      for ARM in equal indel; do
        CL=$SEL/chains.txt; [ "$ARM" = equal ] && CL=$POOL/chains_equal.txt
        echo "== $MODEL $ARM natives vs production"
        python scratchpad/xrun_index_compare.py --a "$X/index/${MODEL}_$ARM/$(basename "$PROD")" --b "$PROD" \
            --natives-only --require-all --min-rows "$(grep -c . "$CL")" || RC=6
      done
    done
    MODEL=cb8; model_args
    echo "== cb8 equal template rows vs production (the v4 rows themselves)"
    python scratchpad/xrun_index_compare.py --a "$X/index/cb8_equal/$(basename "$PROD")" --b "$PROD" --require-all \
        --min-rows "$(( 2 * $(grep -c . "$POOL/chains_equal.txt") ))" || RC=7
    MODEL=t8b; model_args
    echo "== t8b indel template rows vs production (same final-selection templates; maps recomputed at bs 1)"
    python scratchpad/xrun_index_compare.py --a "$X/index/t8b_indel/$(basename "$PROD")" --b "$PROD" --require-all \
        --min-rows "$(( 2 * $(grep -c . "$SEL/chains.txt") ))" || RC=7
    ;;
  *) echo "unknown MODE=$MODE"; RC=2;;
esac
echo "XRUN_INDEX_EXIT=$RC"
exit $RC
