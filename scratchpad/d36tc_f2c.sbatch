#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc S1 GPU: Frame2ConFind maps for every D36tc .pt (34 targets, 31 hits, 3 Control-A training chains) with the
# STORED-MAP recipe verbatim: /orcd/pool/006/chenxiou/backfill/env.sh (cue_openfold, cuda 13.0.1, the catbalanced
# checkout that wrote every stored contact_map_confind, winner ckpt) and winner_rerun.sh's bands
# (<=512 bs 8, <=768 bs 4, <=1024 bs 2, <=4096 bs 1; bf16). Each band only picks up .pt still without a map.
# Partition/gres/time/name (d36tc_gpu, singleton) on the sbatch command line.
set -uo pipefail
P=/orcd/pool/006/chenxiou/d36tc
PROC=$P/data/processed
I=$P/inputs
set +u; source /orcd/pool/006/chenxiou/backfill/env.sh; set -u
export PYTHONUNBUFFERED=1
echo "[f2c] CKPT=$CKPT PYTHONPATH=$PYTHONPATH"; git -C /orcd/pool/006/chenxiou/proteina-catbalanced log -1 --format="[f2c code] %h"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for band in "512 8" "768 4" "1024 2" "4096 1"; do
  set -- $band
  python -m proteinfoundation.utils.precompute_frame2confind_maps --checkpoint "$CKPT" --processed-dir "$PROC" \
    --max-len "$1" --batch-size "$2" --amp-dtype bf16 --save-workers 8 \
    --status-csv "$P/logs/f2c_band$1_${SLURM_JOB_ID}.csv" --log "$P/logs/f2c_band$1_${SLURM_JOB_ID}.log" || rc=$?
done
echo "F2C_RC=$rc"
# gate in the build env (the verify script lives in the d36tc clone)
REPO="${REPO:-/orcd/scratch/orcd/011/chenxiou/proteina_d36tc}"
PYTHONPATH=$REPO python "$REPO/scratchpad/d36tc_verify_inputs.py" --processed-dir "$PROC" --target-seqs "$I/d36tc_target_seqs.json" \
  --hit-manifest "$I/d36tc_hit_manifest.json" --extra "$I/controlA_stems.txt" --stage f2c
vrc=$?
echo "VERIFY_RC=$vrc"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$vrc"
