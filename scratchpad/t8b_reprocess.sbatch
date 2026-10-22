#!/bin/bash
#SBATCH --job-name=t8b_reprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/t8b_seqres/logs/%x-%A_%a.out
#
# T8b SEQRES re-process (proteinfoundation/utils/reprocess_seqres_chains.py) and its regression
# (scratchpad/test_seqres_regression.py). CPU only; partition / --array / --time / --cpus-per-task at submit.
# Env: MODE=stems|part|all|regression. all: every pack stem, written only if changed (N_PARTS, OUT_ROOT; array index = part). stems: writes STEMS (merged + nonmonotone + ok with n_ins > 0 + collide, from AUDIT).
# part: STEMS, N_PARTS (the array index is the part), OUT_ROOT. regression: OUT_DIR (workers = SLURM_CPUS_PER_TASK).
# conda cue_openfold: the sm120 venv has no pydssp (dssp_target).
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
# sbatch runs a spooled copy of this file, so the checkout comes from the environment
REPO="${REPO_DIR:?REPO_DIR = the checkout to run}"
MODE="${MODE:?MODE=stems|part|all|regression}"
S=/orcd/scratch/orcd/011/chenxiou
PACK="${PACK:-/orcd/compute/so3/002/chenxi/proteina_pack/tri_t8_512.pack}"
AUDIT="${AUDIT:-$S/t8b_seqres/full_v2/seqres_index_t8_v2.pt.audit.tsv}"
DATA=/orcd/pool/006/chenxiou/proteina/data/pdb_train
OUT_ROOT="${OUT_ROOT:-/orcd/compute/so3/002/chenxi/pdb_train_seqres_fix}"
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null; conda activate cue_openfold; set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$REPO
cd "$REPO" || exit 3
git log --oneline -1
python -c "import biopandas, pandas, numpy, torch, pydssp; print('biopandas', biopandas.__version__, 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)"
if [ "$MODE" = stems ]; then
    awk -F'\t' 'NR > 1 && ($2 == "merged" || $2 == "nonmonotone" || $2 == "collide" || ($2 == "ok" && $6 > 0)) {print $1}' \
        "$AUDIT" > "${STEMS:?STEMS = output stems file}"
    rc=$?
    echo "stems: $(wc -l < "$STEMS") -> $STEMS"
elif [ "$MODE" = part ]; then
    python -m proteinfoundation.utils.reprocess_seqres_chains --pack "$PACK" --data-dir "$DATA" --audit "$AUDIT" \
        --stems "${STEMS:?STEMS}" --part "${SLURM_ARRAY_TASK_ID:?array index = part}" --n-parts "${N_PARTS:?N_PARTS}" \
        --out-root "$OUT_ROOT"
    rc=$?
elif [ "$MODE" = all ]; then
    python -m proteinfoundation.utils.reprocess_seqres_chains --pack "$PACK" --data-dir "$DATA" --audit "$AUDIT" \
        --all-stems --write-changed-only --part "${SLURM_ARRAY_TASK_ID:?array index = part}" --n-parts "${N_PARTS:?N_PARTS}" \
        --out-root "$OUT_ROOT"
    rc=$?
elif [ "$MODE" = regression ]; then
    python scratchpad/test_seqres_regression.py --pack "$PACK" --data-dir "$DATA" --audit "$AUDIT" \
        --out-dir "${OUT_DIR:?OUT_DIR}" --workers "${SLURM_CPUS_PER_TASK:-2}"
    rc=$?
else
    echo "FATAL: MODE=$MODE"; exit 2
fi
echo "REPROCESS_EXIT=$rc"
exit $rc
