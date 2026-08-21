#!/bin/bash
#SBATCH --job-name=desync
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# BOTH instruments in one job, so the hunt costs one 2x L40S slot instead of two:
#   1. TORCH_NCCL_DESYNC_DEBUG=1 -- PyTorch's own desync analysis names the culprit rank and the
#      collective each rank was in, with no ptrace at all.
#   2. py-spy as the PARENT of torchrun (`py-spy record -- torchrun ...`). The previous attempt
#      failed with Permission Denied because the watcher was a SIBLING of torchrun and this
#      cluster's yama.ptrace_scope permits tracing DESCENDANTS only. As parent, every rank is a
#      descendant, so the same tool works with no elevated privileges.
# raw format = collapsed text stacks, greppable; a flamegraph SVG would need a viewer.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
OUT=$REPO/logs/desync_${SLURM_JOB_ID}
mkdir -p "$OUT"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data

module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$OUT/nccl_trace_rank_"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
cd "$REPO"

# heartbeat so the state is readable from login without racing the job
( for i in $(seq 1 110); do sleep 30; echo "[$(date +%H:%M:%S)] ${i}x30s alive" >> "$OUT/STATUS.txt"; done ) &
WATCHER=$!

py-spy record --subprocesses --rate 2 --duration 780 --format raw --output "$OUT/pyspy_raw.txt" -- \
  torchrun --standalone --nproc_per_node=2 proteinfoundation/train.py \
    --config_name training_contact_hier_topology_full512_v1 \
    --ngpus_per_node 2 --nnodes 1 run_name_=desync_diag \
    opt.dist_timeout_s=600 || echo "wrapped run exited rc=$?"

kill $WATCHER 2>/dev/null || true
echo "=== outputs ==="
ls -l "$OUT"
