#!/bin/bash
#SBATCH --job-name=t8b_seqres
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/t8b_seqres/logs/%x-%A_%a.out
#
# T8b SEQRES side table (proteinfoundation/utils/precompute_seqres_index.py). CPU only; partition / --array / --time at
# submit. Env: MODE=part|merge, STEMS (one stem per line), OUT_DIR (parts dir), N_PARTS (part mode; the array index is
# the part), OUT (merge mode). PACK defaults to the T8 pack.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
# sbatch runs a spooled copy of this file, so the checkout comes from the environment
REPO="${REPO_DIR:?REPO_DIR = the checkout to run}"
MODE="${MODE:?MODE=part|merge}"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export OMP_NUM_THREADS=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PACK="${PACK:-/orcd/compute/so3/002/chenxi/proteina_pack/tri_t8_512.pack}"
cd "$REPO" || exit 3
git log --oneline -1
python -c "import biopandas, pandas, numpy; print('biopandas', biopandas.__version__, 'pandas', pandas.__version__, 'numpy', numpy.__version__)"
if [ "$MODE" = part ]; then
    mkdir -p "$OUT_DIR"
    python -m proteinfoundation.utils.precompute_seqres_index part --pack "$PACK" \
        --data-dir /orcd/pool/006/chenxiou/proteina/data/pdb_train --stems "$STEMS" \
        --part "${SLURM_ARRAY_TASK_ID:?array index = part}" --n-parts "$N_PARTS" --out-dir "$OUT_DIR"
elif [ "$MODE" = merge ]; then
    python -m proteinfoundation.utils.precompute_seqres_index merge --stems "$STEMS" --parts-dir "$OUT_DIR" --out "$OUT"
else
    echo "FATAL: MODE=$MODE"; exit 2
fi
rc=$?
echo "SEQRES_EXIT=$rc"
exit $rc
