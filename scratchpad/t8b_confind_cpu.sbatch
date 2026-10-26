#!/bin/bash
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 48
#SBATCH -t 12:00:00
#SBATCH --exclude=node5101,node5204,node1800,node1801,node1305,node3006
#SBATCH -J t8b_confind_cpu
#SBATCH -o /orcd/compute/so3/002/chenxi/t8b_confind/logs/%x_%j.out
#SBATCH --open-mode=append
# CPU steps of the T8b native ConFind maps (-c 48 = backfill scan/verify jobs; -t = mit_normal MaxTime).
# MODE=gate_prep | compare | cache | check; REPO = checkout holding scratchpad/t8b_confind_*.py
# gate_prep refuses an existing gate dir, so a preempted+requeued gate_prep fails loudly: rm -rf $OUT/gate and resubmit.
#   sbatch --export=ALL,MODE=gate_prep,REPO=<checkout> scratchpad/t8b_confind_cpu.sbatch
source /orcd/pool/006/chenxiou/backfill/env.sh
set -u
: "${MODE:?MODE=gate_prep|compare|cache|check}"; : "${REPO:?REPO=checkout with scratchpad/}"
OUT=/orcd/compute/so3/002/chenxi/t8b_confind
TSV=/orcd/scratch/orcd/011/chenxiou/t8b_seqres/reproc_status_final.tsv
FIX=/orcd/compute/so3/002/chenxi/pdb_train_seqres_fix/processed
ORIG=/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed
W=${SLURM_CPUS_PER_TASK:-1}
S=$REPO/scratchpad
echo "[$(date)] T8B CONFIND CPU mode=$MODE repo=$REPO ($(git -C "$REPO" rev-parse --short HEAD)) host=$(hostname)"
case "$MODE" in
  gate_prep) python "$S/t8b_confind_cache.py" --gate --status-tsv "$TSV" --tree "$ORIG" --gate-dir "$OUT/gate" ;;
  compare) python "$S/t8b_confind_compare.py" --gate --gate-dir "$OUT/gate" --status-dir "$OUT/status/gate" ;;
  cache) python "$S/t8b_confind_cache.py" --status-tsv "$TSV" --tree "$FIX" --workers "$W" \
           --out-cache "$OUT/cache_full.json" --out-stems "$OUT/stems_full.txt" ;;
  check) python "$S/t8b_confind_compare.py" --check --cache "$OUT/cache_full.json" --status-tsv "$TSV" \
           --status-dir "$OUT/status/full" --ref-maps "$OUT/gate/ref_maps.pt" --workers "$W" ;;
  *) echo "bad MODE=$MODE"; exit 2 ;;
esac
rc=$?
echo "[$(date)] T8B CONFIND CPU mode=$MODE done rc=$rc"
exit $rc
