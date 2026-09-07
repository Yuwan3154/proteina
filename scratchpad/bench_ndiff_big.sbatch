#!/bin/bash
#SBATCH --job-name=ndiffbench
#SBATCH --partition=pi_so3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Fix B (diffusion multiplicity 8 -> 16) needs a card the measurement can actually fit on.
# n_diff=8 peaks at 38.10 GiB, so an L40S (48 GB) OOMs at 16 before telling us anything useful.
# ⛔ The point is the PEAK at the production shape (L=384), not a throughput headline.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
cd /orcd/scratch/orcd/011/chenxiou/proteina_sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== re-run the two gates fixed after the first pass ==="
python scratchpad/test_contact2coord.py; R1=$?
python scratchpad/test_local_attention.py; R2=$?

echo; echo "=== n_diff sweep at the production length ==="
python scratchpad/bench_c2c.py --ndiff 4 8 16 --lengths 384; R3=$?

echo; echo "RESULTS c2c=$R1 local=$R2 bench=$R3"
