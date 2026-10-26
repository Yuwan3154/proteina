#!/bin/bash
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --exclude=node5101,node5204,node1800,node1801,node1305,node3006
#SBATCH -J t8b_data_dir
#SBATCH -o /orcd/compute/so3/002/chenxi/t8b_data/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
# T8b data_dir (REPAIR PLAN S8) = scratchpad/t8b_data_dir.py; MODE=build|verify; REPO = checkout holding it.
# CONFIG = dataset yaml. build: only its dataselector / datasplitter.split_sequence_similarity are read (CSV / cluster tsv
# names; the T8 and T8b yamls are identical there). verify: MUST be the T8b yaml (data_dir -> pdb_train_t8b,
# regenerate_missing False, overwrite False; asserted) + PACK (the verified overlay pack), N_SAMPLE stems per status class, SEED.
# build refuses an existing OUT and rebuilds OUT.partial from scratch, so a preempted+requeued build simply restarts.
# -c 2 / --mem 16G = t8b_reprocess.sbatch (single-threaded CSV + symlink work).
# Before the first submit (login node): mkdir -p /orcd/compute/so3/002/chenxi/t8b_data/logs
#   sbatch --export=ALL,MODE=build,REPO=<checkout>,CONFIG=<yaml> scratchpad/t8b_data_dir.sbatch
# OUT / TSV override the v1 defaults (June pad-fix v2: TSV=.../reproc_status_final_padfix.tsv and a new OUT).
# A command-line --dependency REPLACES the in-script singleton: always pass --dependency=afterok:<job>,singleton.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
: "${MODE:?MODE=build|verify}"; : "${REPO:?REPO=checkout with scratchpad/t8b_data_dir.py}"; : "${CONFIG:?CONFIG=dataset yaml}"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
# OUT and TSV: both default (v1) or both overridden, and an overridden TSV must be a pad-fix table
if [ -n "${OUT:-}" ] && [ -n "${TSV:-}" ]; then
  case "$(basename "$TSV")" in *padfix*) ;; *) echo "FATAL: overridden TSV=$TSV is not a padfix table"; exit 2;; esac
elif [ -n "${OUT:-}" ] || [ -n "${TSV:-}" ]; then
  echo "FATAL: OUT and TSV must both be default or both overridden (OUT=${OUT:-} TSV=${TSV:-})"; exit 2
fi
OUT=${OUT:-$DATA_PATH/pdb_train_t8b}
TSV=${TSV:-/orcd/scratch/orcd/011/chenxiou/t8b_seqres/reproc_status_final.tsv}
FIX=/orcd/compute/so3/002/chenxi/pdb_train_seqres_fix/processed
ORIG=$DATA_PATH/pdb_train
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null; conda activate cue_openfold; set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$REPO
cd "$REPO" || exit 3
echo "[$(date)] T8B DATA DIR out=$OUT tsv=$TSV"
echo "[$(date)] T8B DATA DIR mode=$MODE repo=$REPO ($(git rev-parse --short HEAD)) config=$CONFIG host=$(hostname)"
python -c "import proteinfoundation, omegaconf, pandas, torch_geometric; assert proteinfoundation.__file__.startswith('$REPO/'), proteinfoundation.__file__" || exit 4
ARGS=(--config "$CONFIG" --orig-data-dir "$ORIG" --fix-processed "$FIX" --status-tsv "$TSV" --out "$OUT")
if [ "$MODE" = verify ]; then
    ARGS+=(--pack "${PACK:?PACK=overlay pack}" --n-sample "${N_SAMPLE:?N_SAMPLE}" --seed "${SEED:?SEED}")
fi
python scratchpad/t8b_data_dir.py "$MODE" "${ARGS[@]}"
rc=$?
echo "[$(date)] T8B DATA DIR mode=$MODE done rc=$rc"
exit $rc
