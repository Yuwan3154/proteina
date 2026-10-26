#!/bin/bash
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 48
#SBATCH -t 12:00:00
#SBATCH --exclude=node5101,node5204,node1800,node1801,node1305,node3006
#SBATCH -J t8b_confind_cpu
#SBATCH -o /orcd/compute/so3/002/chenxi/t8b_confind/logs/%x_%j.out
#SBATCH --open-mode=append
# CPU steps of the T8b native ConFind maps (-c 48 = backfill scan/verify jobs; -t = mit_normal MaxTime).
# MODE=gate_prep | compare | cache | check | padfix_list | padfix_cache | padfix_check | padfix_table;
# REPO = checkout holding scratchpad/t8b_confind_*.py + t8b_padfix_list.py
# padfix_* = the June pad-fix (user 2026-10-26): list -> cache (copies originals into the fix tree) -> GPU
#   t8b_confind_maps.sbatch MODE=padfix -> check -> table (NEW status TSV; the old one is never modified).
#   JUNE_BS=8 / JUNE_MAXLEN=512 = band1 --batch-size / --max-len of backfill/winner_rerun.sh (job 15356272), which fed the
#   precompute caches/full_pdb_train_band1.json; N_PARTS 60 = t8b_index.sbatch default.
# gate_prep refuses an existing gate dir, so a preempted+requeued gate_prep fails loudly: rm -rf $OUT/gate and resubmit.
#   sbatch --export=ALL,MODE=gate_prep,REPO=<checkout> scratchpad/t8b_confind_cpu.sbatch
source /orcd/pool/006/chenxiou/backfill/env.sh
set -u
: "${MODE:?MODE=gate_prep|compare|cache|check|padfix_list|padfix_cache|padfix_check|padfix_table}"; : "${REPO:?REPO=checkout with scratchpad/}"
OUT=/orcd/compute/so3/002/chenxi/t8b_confind
TSV=/orcd/scratch/orcd/011/chenxiou/t8b_seqres/reproc_status_final.tsv
FIX=/orcd/compute/so3/002/chenxi/pdb_train_seqres_fix/processed
ORIG=/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed
W=${SLURM_CPUS_PER_TASK:-1}
S=$REPO/scratchpad
PF=$OUT/padfix
JUNE_CSV=/orcd/pool/006/chenxiou/backfill/winner_status/pdb_train/band1_15356272.csv
JUNE_BS=8
JUNE_MAXLEN=512
TSV_PADFIX=/orcd/scratch/orcd/011/chenxiou/t8b_seqres/reproc_status_final_padfix.tsv
mkdir -p "$PF"
echo "[$(date)] T8B CONFIND CPU mode=$MODE repo=$REPO ($(git -C "$REPO" rev-parse --short HEAD)) host=$(hostname)"
case "$MODE" in
  gate_prep) python "$S/t8b_confind_cache.py" --gate --status-tsv "$TSV" --tree "$ORIG" --gate-dir "$OUT/gate" ;;
  compare) python "$S/t8b_confind_compare.py" --gate --gate-dir "$OUT/gate" --status-dir "$OUT/status/gate" ;;
  cache) python "$S/t8b_confind_cache.py" --status-tsv "$TSV" --tree "$FIX" --workers "$W" \
           --out-cache "$OUT/cache_full.json" --out-stems "$OUT/stems_full.txt" ;;
  # one torch thread per worker: 48 workers x 48 intra-op threads stalled the first check (26832562 TIMEOUT at 12 h)
  check) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$S/t8b_confind_compare.py" --check --cache "$OUT/cache_full.json" --status-tsv "$TSV" \
           --status-dir "$OUT/status/full" --ref-maps "$OUT/gate/ref_maps.pt" --workers "$W" ;;
  # PYTHONPATH=$REPO: the list imports _processed_path_sharded from this checkout (env.sh puts proteina-catbalanced first)
  padfix_list) PYTHONPATH=$REPO:$PYTHONPATH python "$S/t8b_padfix_list.py" list --june-csv "$JUNE_CSV" \
           --length-cache /orcd/pool/006/chenxiou/backfill/caches/full_pdb_train_band1.json \
           --data-dir /orcd/pool/006/chenxiou/proteina/data/pdb_train --max-len "$JUNE_MAXLEN" --bs "$JUNE_BS" --status-tsv "$TSV" \
           --template-list /orcd/scratch/orcd/011/chenxiou/t8_template_list/template_chains.txt \
           --eligible /orcd/compute/so3/002/chenxi/t8b_index/eligible_ids_tm0.5-0.9.txt \
           --val-fixed /orcd/scratch/orcd/011/chenxiou/t8_valfixed/val_fixed32_t8_within.txt \
           --n-parts 60 --out-tsv "$PF/padfix_list.tsv" ;;
  padfix_cache) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$S/t8b_confind_cache.py" --padfix --padfix-list "$PF/padfix_list.tsv" --status-tsv "$TSV" \
           --tree "$ORIG" --fix-tree "$FIX" --workers "$W" --out-cache "$PF/cache_padfix.json" \
           --out-stems "$PF/stems_padfix.txt" ;;
  padfix_check) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$S/t8b_confind_compare.py" --padfix-check --cache "$PF/cache_padfix.json" --tree "$ORIG" \
           --fix-tree "$FIX" --status-dir "$OUT/status/padfix" --out-tsv "$PF/check_padfix.tsv" --dropped-tsv "$PF/dropped_padfix.tsv" \
           --workers "$W" ;;
  padfix_table) python "$S/t8b_padfix_list.py" table --status-tsv "$TSV" --padfix-list "$PF/padfix_list.tsv" \
           --check-tsv "$PF/check_padfix.tsv" --dropped-tsv "$PF/dropped_padfix.tsv" --out "$TSV_PADFIX" ;;
  *) echo "bad MODE=$MODE"; exit 2 ;;
esac
rc=$?
echo "[$(date)] T8B CONFIND CPU mode=$MODE done rc=$rc"
exit $rc
