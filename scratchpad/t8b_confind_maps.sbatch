#!/bin/bash
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --exclude=node5101,node5204,node1800,node1801,node1305,node3006
#SBATCH -J t8b_confind
#SBATCH -o /orcd/compute/so3/002/chenxi/t8b_confind/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --chdir=/orcd/compute/so3/002/chenxi/t8b_confind
# T8b native ConFind maps on 1 GPU = the winner rerun (backfill/winner_rerun.sh, job 15356272) band1 settings.
# MODE=gate: the 100 copied originals (--overwrite); MODE=full: every status==changed chain of the fix tree.
# Resumable: remaining = cache minus status=processed basenames over all session CSVs of the mode; resubmit (singleton: one session at a time).
# Before the first submit (login node): mkdir -p /orcd/compute/so3/002/chenxi/t8b_confind/logs (the -o / --chdir dirs must exist).
# -t 06:00:00 = mit_normal_gpu MaxTime (mit_preemptable allows 2 days). GPU pinned to L40S = the original run's type (bf16
# kernels differ by architecture; the gate only covers the type it ran on). --chdir keeps the checkout off sys.path.
#   sbatch --export=ALL,MODE=gate scratchpad/t8b_confind_maps.sbatch
source /orcd/pool/006/chenxiou/backfill/env.sh
set -u
: "${MODE:?MODE=gate|full}"
OUT=/orcd/compute/so3/002/chenxi/t8b_confind
case "$MODE" in
  gate) FULL=$OUT/gate/cache_gate.json; PROC=$OUT/gate/processed; OVW=--overwrite ;;
  full) FULL=$OUT/cache_full.json; PROC=/orcd/compute/so3/002/chenxi/pdb_train_seqres_fix/processed; OVW= ;;
  *) echo "bad MODE=$MODE"; exit 2 ;;
esac
SDIR=$OUT/status/$MODE
REM=$OUT/remaining_${MODE}.json
JOB=${SLURM_JOB_ID:-manual$$}_r${SLURM_RESTART_COUNT:-0}  # requeue keeps the job id; never reopen an old CSV
mkdir -p "$SDIR" "$OUT/logs"
[ -s "$FULL" ] || { echo "missing cache $FULL (run the CPU cache step first)"; exit 3; }

echo "[$(date)] T8B CONFIND mode=$MODE ckpt=$CKPT job=$JOB host=$(hostname)"
nvidia-smi -L
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
case "$GPU" in
  *L40S*) echo "GPU ok: $GPU" ;;
  *) echo "GPU_NOT_APPROVED: $GPU"; exit 9 ;;
esac
MD5=$(md5sum "$CKPT" | cut -d' ' -f1)
[ "$MD5" = e928f9bff6a6020538e8dbc56e928202 ] || { echo "CKPT_MD5_MISMATCH $MD5"; exit 8; }

python - <<PY || { echo PE_ASSERT_FAILED; exit 7; }
import sys
from Frame2ConFind.inference.api import Frame2ConFindPredictor
p = Frame2ConFindPredictor(checkpoint="$CKPT", amp_dtype="bf16")
z = bool(getattr(p.model, "zero_absolute_positional_embedding", False))
print("RUNTIME zero_absolute_positional_embedding =", z)
sys.exit(0 if z else 7)
PY
F=$(python -c 'import proteinfoundation.utils.precompute_frame2confind_maps as m; print(m.__file__)')
[ "$(md5sum "$F" | cut -d' ' -f1)" = cbe6c67e747d85285909c89a70c0f14e ] || { echo "PRECOMPUTE_MD5_MISMATCH $F"; exit 6; }
echo "[code] $F; Frame2ConFind $(python -c 'import Frame2ConFind; print(Frame2ConFind.__file__)')"

python - "$FULL" "$SDIR" "$REM" "$PROC" <<'PY'
import csv, glob, json, os, sys
full_path, sdir, rem_path, proc = sys.argv[1:5]
full = json.load(open(full_path))
assert all(e[0].startswith(proc.rstrip("/") + "/") and "/orcd/pool/006/chenxiou/proteina/data/" not in e[0] for e in full), "cache row outside PROC"
done = set()
for c in glob.glob(os.path.join(sdir, "*.csv")):
    with open(c) as fh:
        done |= {r["file"] for r in csv.DictReader(fh) if r["status"] == "processed"}
rem = [e for e in full if os.path.basename(e[0]) not in done]
json.dump(rem, open(rem_path, "w"))
print(f"full={len(full)} done={len(done)} remaining={len(rem)}")
PY
rc=$?
[ "$rc" -eq 0 ] || { echo "REMAINING_FAILED rc=$rc"; exit $rc; }
n=$(python -c "import json;print(len(json.load(open('$REM'))))")
if [ "$n" -eq 0 ]; then echo "[$(date)] $MODE complete (0 remaining)"; exit 0; fi

echo "[$(date)] === $MODE: $n remaining, max-len 512, bs 8 $OVW ==="
python -m proteinfoundation.utils.precompute_frame2confind_maps \
  --checkpoint "$CKPT" --processed-dir "$PROC" --length-cache "$REM" \
  --max-len 512 --batch-size 8 --amp-dtype bf16 --save-workers 8 $OVW \
  --status-csv "$SDIR/band1_${JOB}.csv" --log "$OUT/logs/t8b_confind_${MODE}_${JOB}.log"
rc=$?
python - "$SDIR/band1_${JOB}.csv" <<'PY'
import collections, csv, sys
print("session status counts:", dict(collections.Counter(r["status"] for r in csv.DictReader(open(sys.argv[1])))))
PY
echo "[$(date)] === $MODE session done rc=$rc ==="
exit $rc
