#!/bin/bash
#SBATCH --job-name=synth_tri_bench
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --time=4-04:00:00
#SBATCH --output=/home/gridsan/cou/logs/%x-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/%x-%A_%a.err
# Synthetic-template benchmark stage 1 on SuperCloud (V100 x 2 per node, one process = one shard per GPU).
# Submit as an array over nodes (e.g. --array=0-3); shard = array_task * 2 + local GPU, n_shards = array_count * 2.
#   PAIRS=<one arm's pairs JSONL>  OUT=<that arm's output dir>  TRI_CKPT  C2C_CKPT  SYNTH_INDEX_DIR  F2C_CKPT_DIR
#   DATA_PATH  REPO (this checkout, 8c1ef48)  N_NODES (= nodes of the FULL run; a single-task resubmit keeps it)
#   optional K (8)  F2C_DTYPE (fp32)  MAX_PAIRS  PAIRS_FILTER (shakedown subset)
# The two shards run as two background processes pinned by CUDA_VISIBLE_DEVICES (stageB_native_supercloud.sbatch).
# /etc/profile before -e (its profile.d scripts are not -e clean); no -u: Z97-byobu.sh reads an unset LC_BYOBU.
source /etc/profile
set -eo pipefail
: "${PAIRS:?}" "${OUT:?}" "${TRI_CKPT:?}" "${C2C_CKPT:?}" "${SYNTH_INDEX_DIR:?}" "${F2C_CKPT_DIR:?}" "${DATA_PATH:?}" "${REPO:?}" "${N_NODES:?}"
ENV_DIR=/home/gridsan/cou/cue-openfold-env
export CUE_HOST=supercloud
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export TMPDIR=/home/gridsan/cou/.tmp_synth_tri   # compute-node /tmp is a 512 MB per-user quota
mkdir -p "$TMPDIR" "$OUT" /home/gridsan/cou/logs
# SIGMA_DATA: the c2c twin's training value, read by af3_diffusion at import. ALIGN_W / MLM_W: the cfft yaml's
# ${oc.env:...} loss weights (stageA_pass.sbatch values); SKIP_CONFIND_PRECOMPUTE as stageA_pass / stageB.
export SIGMA_DATA=16.0 ALIGN_W=0.3 MLM_W=0.3 SKIP_CONFIND_PRECOMPUTE=1
export SYNTH_INDEX_DIR DATA_PATH
# Frame2ConFind is imported as a package from its parent directory (the builder's --f2c-parent default: ~)
F2C_PARENT=${F2C_PARENT:-/home/gridsan/cou}   # directory holding the Frame2ConFind package (FRAME2SEQ_HOME, if set, locates Frame2seq)
export PYTHONPATH="$REPO:$F2C_PARENT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=20
INDEX=$SYNTH_INDEX_DIR/topology_index_confind_synth.pt
for f in "$PAIRS" "$TRI_CKPT" "$C2C_CKPT" "$INDEX" "$F2C_CKPT_DIR/best.pt"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 77; }
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
git -C "$REPO" log -1 --format="[repo] $REPO HEAD %h"
git -C "$F2C_PARENT/Frame2ConFind" log -1 --format="[f2c] $F2C_PARENT/Frame2ConFind HEAD %h"; echo "[f2s] FRAME2SEQ_HOME=${FRAME2SEQ_HOME:-unset}"
echo "[ckpt] tri md5=$(md5sum < "$TRI_CKPT" | cut -c1-12) c2c md5=$(md5sum < "$C2C_CKPT" | cut -c1-12)"
NODE=${SLURM_ARRAY_TASK_ID:-0}
[[ "$N_NODES" =~ ^[1-9][0-9]*$ ]] && [ "$NODE" -lt "$N_NODES" ] || { echo "FATAL: N_NODES=$N_NODES, task $NODE"; exit 77; }
NSH=$((N_NODES * 2))
EXTRA=()
[ -n "${MAX_PAIRS:-}" ] && EXTRA+=(--max-pairs "$MAX_PAIRS")
[ -n "${PAIRS_FILTER:-}" ] && EXTRA+=(--pairs-filter "$PAIRS_FILTER")
cd "$REPO"
PIDS=()
for g in 0 1; do
  SH=$((NODE * 2 + g))
  CUDA_VISIBLE_DEVICES=$g python scratchpad/synth_tri_bench.py --pairs "$PAIRS" --shard $SH --n-shards $NSH \
      --tri-ckpt "$TRI_CKPT" --c2c-ckpt "$C2C_CKPT" --index "$INDEX" --f2c-ckpt-dir "$F2C_CKPT_DIR" \
      --f2c-dtype "${F2C_DTYPE:-fp32}" --out "$OUT" --k "${K:-8}" "${EXTRA[@]}" \
      > "$OUT/shard${SH}.log" 2>&1 &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do
  wait "$p" || rc=$?
done
echo "[rc] node $NODE shards $((NODE * 2)),$((NODE * 2 + 1)) rc=$rc"
for g in 0 1; do tail -n 1 "$OUT/shard$((NODE * 2 + g)).log"; done
exit $rc
