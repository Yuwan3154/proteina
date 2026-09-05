#!/bin/bash
#SBATCH --job-name=blkbench
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_sh/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_sh/logs/%x-%j.err
#SBATCH --open-mode=append
#
# L40S on purpose: the plan's central claim is that head-only training fits on a mid-size card, and
# an L40S (48 GB) is the cheapest thing that tests it. Nothing here competes with the three training
# arms, which hold rtx_pro_6000 and h200.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_sh
cd "$REPO"
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70"; exit 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/orcd/home/002/chenxiou/cue-openfold-env-sm120/.venv/bin/python3

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scratchpad/bench_blocks.py --lengths 256 384 --batch 1
rc=$?
echo "BLKBENCH_EXIT=$rc"
exit $rc
