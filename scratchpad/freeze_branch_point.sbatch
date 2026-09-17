#!/bin/bash
#SBATCH --job-name=freezebp
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Freeze a branch point for a fine-tune.
#
# ⛔ c2c_cb8_tbeta is LIVE and rewrites last.ckpt at every checkpoint, so a fine-tune pointed
# straight at it can read a half-written file. Copy first, verify, then branch from the copy -- the
# same discipline the wch arms use with branch_points/c2c_cb8_tbeta_step5500.ckpt.
# ⛔ CPU job, not a login-node cp: 3.2 GB is real I/O.
set -uo pipefail
SRC="${SRC:-/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_cb8_tbeta/last.ckpt}"
DST="${DST:?DST must be given}"
mkdir -p "$(dirname "$DST")"
[ -e "$DST" ] && { echo "FATAL: $DST already exists -- refusing to overwrite a branch point"; exit 2; }

cp "$SRC" "$DST" || { echo "FATAL: copy failed"; exit 3; }

# ⛔ Verify the ARTEFACT, not the exit status: a torn read from a live writer still exits 0.
s1=$(stat -c %s "$SRC"); s2=$(stat -c %s "$DST")
echo "src bytes=$s1  dst bytes=$s2"
[ "$s1" = "$s2" ] || { echo "FATAL: size mismatch -- source was being rewritten"; rm -f "$DST"; exit 4; }
m1=$(md5sum "$SRC" | cut -d' ' -f1); m2=$(md5sum "$DST" | cut -d' ' -f1)
echo "src md5=$m1"
echo "dst md5=$m2"
[ "$m1" = "$m2" ] || { echo "FATAL: md5 mismatch -- torn copy"; rm -f "$DST"; exit 5; }

ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export TORCH_CUDA_ARCH_LIST=9.0        # activate.sh requires it; inert on a CPU node
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
python - "$DST" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"global_step={ck.get('global_step')} epoch={ck.get('epoch')} has_ema={'ema' in ck}")
assert ck.get("global_step"), "checkpoint has no global_step -- not a usable branch point"
PY
echo "FREEZE_EXIT=$?"
