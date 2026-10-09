#!/bin/bash
#SBATCH --job-name=synthidx5p
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/synth_index_v5_pro/logs/%x-%A_%a.out
# Synthetic topology index v5 = v4 (/orcd/scratch/orcd/011/chenxiou/synth_index_v4.sbatch, commit 802f136) rebuilt with
# --proline-donor-mask (user 2026-10-09): every native + template DSSP recomputed with proline as a non-donor.
# Same pool, band, alias, base index, 60 parts. Runs from its own clone of branch sse-template-1d at EXPECT_COMMIT
# (never the proteina_tri checkout the tri jobs import from). The log dir must exist before submitting.
#   MODE=control sbatch synth_index_v5_pro.sbatch    part 0 WITHOUT the mask -> must equal v4 part 0 byte for byte
#   MODE=pro0    sbatch synth_index_v5_pro.sbatch    part 0 WITH the mask (compared to v4 part 0)
#   MODE=build   sbatch --array=0-59 synth_index_v5_pro.sbatch
#   MODE=merge   sbatch synth_index_v5_pro.sbatch
set -uo pipefail
: "${EXPECT_COMMIT:?}"
ROOT=/orcd/scratch/orcd/011/chenxiou
REPO=$ROOT/proteina_sse1d_idx
OUT=$ROOT/synth_index_v5_pro
V4=$ROOT/synth_index_v4
DATA=/orcd/pool/006/chenxiou/proteina/data/pdb_train
mkdir -p "$OUT/parts" "$OUT/logs" "$OUT/control" "$OUT/pro0"
case "$(git -C "$REPO" rev-parse HEAD)" in "$EXPECT_COMMIT"*) ;; *) echo "FATAL: $REPO is not at $EXPECT_COMMIT"; exit 69;; esac
git -C "$REPO" diff --quiet HEAD || { echo "FATAL: $REPO has local changes"; exit 68; }
cd "$REPO"
set +u
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u
export CUE_HOST=engaging DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "commit $(git log --oneline -1) host $(hostname) mode ${MODE:-build} pkg $(python -c 'import proteinfoundation; print(proteinfoundation.__file__)')"
N_PARTS=60
POOL=$ROOT/t2_pool_auth
POOL_BAND=$ROOT/t2_pool_auth_band.npz
ALIAS=$ROOT/synth_alias/auth_key.tsv
COMMON=(--base-index "$DATA/topology_index.pt" --processed-dir "$DATA/processed" --manifest "$DATA/shard_manifest.json"
        --templates "$POOL:$POOL_BAND" --chain-alias "$ALIAS" --n-parts "$N_PARTS"
        --workers "${SLURM_CPUS_PER_TASK:-32}" --usalign /home/chenxiou/.local/bin/USalign)
B="python -m proteinfoundation.utils.precompute_synthetic_topology_index"
case "${MODE:-build}" in
  control)
    $B build "${COMMON[@]}" --part 0 --out-dir "$OUT/control"
    RC=$?; [ $RC -eq 0 ] && python scripts/compare_synth_parts_dssp.py "$OUT/control/part_0000.pt" "$V4/parts/part_0000.pt" --expect identical
    RC=$?;;
  pro0)
    $B build "${COMMON[@]}" --part 0 --proline-donor-mask --out-dir "$OUT/pro0"
    RC=$?; [ $RC -eq 0 ] && python scripts/compare_synth_parts_dssp.py "$OUT/pro0/part_0000.pt" "$V4/parts/part_0000.pt" --expect dssp
    RC=$?;;
  build)
    $B build "${COMMON[@]}" --part "${SLURM_ARRAY_TASK_ID:?array task id}" --proline-donor-mask --out-dir "$OUT/parts"
    RC=$?;;
  merge)
    $B merge --parts-dir "$OUT/parts" --n-parts "$N_PARTS" --out "$OUT/topology_index_cb8_synth.pt" \
        --eligible-out "$OUT/eligible_ids_tm0.5-0.9.txt" --tm-range 0.5 0.9
    RC=$?;;
  *) echo "unknown MODE"; RC=2;;
esac
echo "SYNTHIDX_EXIT=$RC"
exit $RC
