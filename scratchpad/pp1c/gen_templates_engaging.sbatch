#!/bin/bash
#SBATCH --job-name=pp1c_gen_eng
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/pp1c/logs/%x-%A_%a.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/pp1c/logs/%x-%A_%a.err
# Synthetic templates for the proteina chains whose cluster has no T2 template, generated ON ENGAGING
# (SuperCloud's shared cou GPU cap is fully held by another user's multi-day jobs).
# Same recipe as T2 round 1: 64 rungs 90..375, tiered <=300 / grouped 8x8, cc89 (span<=484) / cc91,
# seed 0, max-length 512 -- so these chains match the existing pool's distribution.
# Resumable by construction: a chain whose npz exists is skipped, so a preemption loses only the
# chain in flight (--requeue re-runs this script). One shard per array task, one GPU each.
# usage: NUM_SHARDS=4 sbatch --array=0-3 gen_templates_engaging.sbatch
set -uo pipefail
ROOT=/orcd/scratch/orcd/011/chenxiou/pp1c
REPO=$ROOT/protpardelle-1c
GEN=/orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad/pp1c/generate_templates.py
MAN=/orcd/scratch/orcd/011/chenxiou/t2_extra/export/manifest.csv
OUT=/orcd/scratch/orcd/011/chenxiou/t2_extra/templates
NUM_SHARDS="${NUM_SHARDS:?}"
SHARD="${SLURM_ARRAY_TASK_ID:?array task id = shard}"
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT/env"
set -u
cd "$REPO"
git log --oneline -1
export PROTPARDELLE_OUTPUT_DIR="$ROOT/results"
export PROTPARDELLE_KEEP_TRAJECTORY=0        # 4 blocking D2H copies/step nobody reads
export PROTPARDELLE_EMPTY_CACHE_PER_STEP=0   # syncs + dumps the allocator pool every step
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$PROTPARDELLE_OUTPUT_DIR" "$OUT" "$TMPDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$GEN" \
    --manifest "$MAN" \
    --out-root "$OUT" \
    --n-templates 64 --rmin 90 --rmax 375 \
    --max-length 512 \
    --tiered-max-length 300 --groups 8 \
    --span-cutoff 484 --short-model cc89 --long-model cc91 \
    --shard "$SHARD" --num-shards "$NUM_SHARDS" --seed 0
rc=$?
echo "PP1C_GEN_EXIT=$rc"
exit $rc
