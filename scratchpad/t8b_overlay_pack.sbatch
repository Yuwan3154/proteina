#!/bin/bash
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 48
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --exclude=node5101,node5204,node1800,node1801,node1305,node3006
#SBATCH -J t8b_overlay_pack
#SBATCH -o /orcd/compute/so3/002/chenxi/t8b_data/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
# T8b overlay pack (REPAIR PLAN S7) = scratchpad/t8b_overlay_pack.py; MODE=build|verify; REPO = checkout holding it.
# build refuses an existing OUT and rewrites OUT.partial from scratch, so a preempted+requeued build simply restarts.
# Run only after the ConFind full run's MODE=check passed (the build's guard also fails closed on any changed .pt without a
# float16 (L_new, L_new) contact_map_confind). -c 48 / -t 12 h = the t8b_confind_cpu.sbatch scan jobs (mit_normal MaxTime).
# Before the first submit (login node): mkdir -p /orcd/compute/so3/002/chenxi/t8b_data/logs
#   sbatch --export=ALL,MODE=build,REPO=<checkout> scratchpad/t8b_overlay_pack.sbatch
# A command-line --dependency REPLACES the in-script singleton: always pass --dependency=afterok:<job>,singleton.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
: "${MODE:?MODE=build|verify}"; : "${REPO:?REPO=checkout with scratchpad/t8b_overlay_pack.py}"
OLD=/orcd/compute/so3/002/chenxi/proteina_pack/tri_t8_512.pack
OUT=/orcd/compute/so3/002/chenxi/proteina_pack/tri_t8b_512.pack
TSV=/orcd/scratch/orcd/011/chenxiou/t8b_seqres/reproc_status_final.tsv
FIX=/orcd/compute/so3/002/chenxi/pdb_train_seqres_fix/processed
ORIG=/orcd/pool/006/chenxiou/proteina/data/pdb_train
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null; conda activate cue_openfold; set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$REPO
cd "$REPO" || exit 3
echo "[$(date)] T8B OVERLAY PACK mode=$MODE repo=$REPO ($(git rev-parse --short HEAD)) host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
python -c "import proteinfoundation, torch, torch_geometric; assert proteinfoundation.__file__.startswith('$REPO/'), proteinfoundation.__file__; print('torch', torch.__version__, 'pyg', torch_geometric.__version__)" || exit 4
python scratchpad/t8b_overlay_pack.py "$MODE" --old-pack "$OLD" --status-tsv "$TSV" --fix-processed "$FIX" \
    --orig-data-dir "$ORIG" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK"
rc=$?
echo "[$(date)] T8B OVERLAY PACK mode=$MODE done rc=$rc"
exit $rc
