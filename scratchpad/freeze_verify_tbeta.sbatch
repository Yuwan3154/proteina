#!/bin/bash
#SBATCH --job-name=freezetb
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Freeze the LIVE c2c_cb8_tbeta last.ckpt as the matched-step reference for the ConFind twin
# (Directive B, user 2026-09-22: "9,500 first, then ~18.5k"), and prove the copy holds RAW weights.
# ⛔ A last.ckpt written at a validation end holds EMA as state_dict (the 8,076 mechanism), so the
#    copy is only accepted if ckpt_ema_vs_raw says GENUINELY DIFFERENT; an EMA copy is deleted.
# ⛔ last.ckpt is rewritten every 10 steps: a copy is accepted only if src and dst md5 agree.
# ⛔ CPU job, not a login-node cp: 3.2 GB is real I/O. Only .venv is sourced (activate.sh needs a GPU).
set -uo pipefail
T=/orcd/scratch/orcd/011/chenxiou/c2c_store
SRC=$T/c2c_cb8_tbeta/last.ckpt
TMP=$T/branch_points/.freeze_tmp_${SLURM_JOB_ID}.ckpt
mkdir -p "$T/branch_points"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp

ok=0
for attempt in 1 2 3; do
    cp "$SRC" "$TMP" || { echo "copy failed (attempt $attempt)"; sleep 60; continue; }
    m1=$(md5sum "$SRC" | cut -d' ' -f1); m2=$(md5sum "$TMP" | cut -d' ' -f1)
    echo "attempt $attempt: src md5=$m1 dst md5=$m2"
    if [ "$m1" = "$m2" ]; then ok=1; break; fi
    rm -f "$TMP"; sleep 60
done
[ "$ok" = 1 ] || { echo "FATAL: no consistent copy in 3 attempts"; exit 3; }

GS=$(python - "$TMP" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert ck.get("global_step"), "no global_step"
assert "ema" in ck, "no ema in checkpoint"
print(int(ck["global_step"]))
PY
) || { echo "FATAL: checkpoint unreadable"; rm -f "$TMP"; exit 4; }
DST=$T/branch_points/c2c_cb8_tbeta_frozen_step${GS}.ckpt
[ -e "$DST" ] && { echo "FATAL: $DST exists -- refusing to overwrite"; rm -f "$TMP"; exit 2; }
mv "$TMP" "$DST"
echo "[frozen] global_step=$GS -> $DST ($(stat -c %s "$DST") bytes)"

V=$(python /orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad/ckpt_ema_vs_raw.py "$DST")
echo "$V"
if echo "$V" | grep -q "GENUINELY DIFFERENT"; then
    echo "[verdict] RAW state_dict -- accepted"
else
    echo "[verdict] state_dict is NOT raw -- deleting the copy"; rm -f "$DST"; exit 5
fi
echo "FREEZE_EXIT=0"
