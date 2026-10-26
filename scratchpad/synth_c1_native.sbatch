#!/bin/bash
#SBATCH --job-name=synth_c1_native
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4-04:00:00
#SBATCH --output=/home/gridsan/cou/logs/%x-%j.out
#SBATCH --error=/home/gridsan/cou/logs/%x-%j.err
# Control C1 of the synthetic-template benchmark (synth_c1_native.py): c2c twin from each target's native ConFind map, one GPU.
#   PAIRS=<ladder pairs JSONL>  OUT  C2C_CKPT  F2C_CKPT_DIR  REPO  optional F2C_PARENT FRAME2SEQ_HOME K C1_ARGS (e.g. C1b: --map confind ...)
# Same environment as synth_tri_bench.sbatch; no -u (Z97-byobu.sh reads an unset LC_BYOBU).
source /etc/profile
set -eo pipefail
: "${PAIRS:?}" "${OUT:?}" "${C2C_CKPT:?}" "${F2C_CKPT_DIR:?}" "${REPO:?}"
ENV_DIR=/home/gridsan/cou/cue-openfold-env
export CUE_HOST=supercloud
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export TMPDIR=/home/gridsan/cou/.tmp_synth_tri
mkdir -p "$TMPDIR" "$OUT"
export SIGMA_DATA=16.0
F2C_PARENT=${F2C_PARENT:-/home/gridsan/cou}
export PYTHONPATH="$REPO:$F2C_PARENT${PYTHONPATH:+:$PYTHONPATH}"
for f in "$PAIRS" "$C2C_CKPT" "$F2C_CKPT_DIR/best.pt"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 77; }
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
git -C "$REPO" log -1 --format="[repo] $REPO HEAD %h"
git -C "$F2C_PARENT/Frame2ConFind" log -1 --format="[f2c] $F2C_PARENT/Frame2ConFind HEAD %h"
echo "[ckpt] c2c md5=$(md5sum < "$C2C_CKPT" | cut -c1-12)"
cd "$REPO"
rc=0
python scratchpad/synth_c1_native.py --pairs "$PAIRS" --c2c-ckpt "$C2C_CKPT" --f2c-ckpt-dir "$F2C_CKPT_DIR" \
    --out "$OUT" --k "${K:-8}" ${C1_ARGS:-} || rc=$?
echo "[rc] $rc"
exit $rc
