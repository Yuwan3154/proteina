#!/bin/bash
#SBATCH --job-name=mirmech
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Baseline mirror behaviour of the RETIRED step-15500 model, i.e. the run that produced the 44%
# mirror rate. This is the number the A+B run has to beat, so it must be measured on the same
# checkpoint the 44% came from.
#
# ⛔ mit_preemptable ON PURPOSE. mit_normal_gpu's per-user cap is 2 GPUs and the A+B training run
# is queued for exactly 2 there -- a diagnostic landing in that partition would consume the cap and
# training could never start.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
cd /orcd/scratch/orcd/011/chenxiou/proteina_sh

CKPT="${CKPT:-/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_v1_qknorm_15500_22168866/last.ckpt}"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT"; exit 1; }

python scratchpad/mirror_mechanism.py \
    --ckpt "$CKPT" \
    --n_chains "${NCHAINS:-6}" --n_seeds "${NSEEDS:-8}" \
    --steps "${STEPS:-50}" --trace_chains "${NTRACE:-3}"
echo "MIRMECH_EXIT=$?"
