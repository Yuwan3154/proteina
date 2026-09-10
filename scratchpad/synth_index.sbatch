#!/bin/bash
#SBATCH --job-name=synthidx
#SBATCH --partition=pi_so3,mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/synth_index/logs/%x-%A_%a.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/synth_index/logs/%x-%A_%a.err
# Synthetic-reference topology index: one array task per part (MODE=build), or the merge (MODE=merge).
# usage:  N_PARTS=40 MODE=build sbatch --array=0-39 synth_index.sbatch
#         N_PARTS=40 MODE=merge sbatch synth_index.sbatch
#         MODE=smoke sbatch synth_index.sbatch      (first 100 chains of part 0 + self-alignment gate)
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
OUT=/orcd/scratch/orcd/011/chenxiou/synth_index
DATA=/orcd/pool/006/chenxiou/proteina/data/pdb_train
T2=/orcd/compute/so3/002/chenxi/of_run/pp1c_work
EXTRA=/orcd/scratch/orcd/011/chenxiou/t2_extra           # templates generated for uncovered clusters (B4)
mkdir -p "$OUT/parts" "$OUT/logs"
cd "$REPO"
# conda env cue_openfold (torch cu126, pydssp): the sm120 uv venv has no pydssp, and the training
# jobs import from that venv live, so it is not modified. CPU-only work; the CUDA build is moot.
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u
export CUE_HOST=engaging DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
git log --oneline -1
MODE="${MODE:-build}"
N_PARTS="${N_PARTS:-40}"
TPL_ARGS=(--templates "$T2/templates_band:$T2/index_band.npz")
[ -f "$EXTRA/index_band.npz" ] && TPL_ARGS+=(--templates "$EXTRA/templates_band:$EXTRA/index_band.npz")
COMMON=(--base-index "$DATA/topology_index.pt" --processed-dir "$DATA/processed" --manifest "$DATA/shard_manifest.json"
        "${TPL_ARGS[@]}" --n-parts "$N_PARTS" --out-dir "$OUT/parts" --workers "${SLURM_CPUS_PER_TASK:-32}"
        --usalign /home/chenxiou/.local/bin/USalign)
case "$MODE" in
  smoke)
    python -m proteinfoundation.utils.precompute_synthetic_topology_index build "${COMMON[@]}" \
        --part 0 --limit "${LIMIT:-100}" --selftest --out-dir "$OUT/smoke_parts"
    RC=$?
    [ $RC -eq 0 ] && python -m proteinfoundation.utils.precompute_synthetic_topology_index merge \
        --parts-dir "$OUT/smoke_parts" --n-parts 1 --out "$OUT/smoke_index.pt" \
        --eligible-out "$OUT/smoke_eligible.txt" --tm-range 0.5 0.9
    ;;
  build)
    python -m proteinfoundation.utils.precompute_synthetic_topology_index build "${COMMON[@]}" \
        --part "${SLURM_ARRAY_TASK_ID:?array task id}"
    RC=$?
    ;;
  merge)
    python -m proteinfoundation.utils.precompute_synthetic_topology_index merge \
        --parts-dir "$OUT/parts" --n-parts "$N_PARTS" --out "$OUT/topology_index_cb8_synth.pt" \
        --eligible-out "$OUT/eligible_ids_tm0.5-0.9.txt" --tm-range 0.5 0.9
    RC=$?
    ;;
  *) echo "unknown MODE=$MODE"; RC=2;;
esac
echo "SYNTHIDX_EXIT=$RC"
exit $RC
