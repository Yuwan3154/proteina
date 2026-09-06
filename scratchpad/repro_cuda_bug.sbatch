#!/bin/bash
#SBATCH --job-name=c2crepro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Reproduce the `CUDA error: an illegal memory access` that killed BOTH 2-GPU QK-norm jobs
# (22072674, 22072676) at ~step 500, and that the single-GPU c2cdbg run has NOT hit in 4000+ steps.
#
# ⛔ Deliberately does NOT take the single-writer lock and writes to its OWN store, so it cannot
# disturb the good c2cdbg run that currently holds the lock and the c2c_v1 checkpoints.
#
# Diagnostics, in order of what they buy:
#   CUDA_LAUNCH_BLOCKING=1   -> the fault is raised at the offending LAUNCH, so the Python traceback
#                               names the real op instead of surfacing later in the NCCL watchdog.
#   TORCH_USE_CUDA_DSA=1     -> device-side assertions, which turn an OOB index into a named assert.
#   TORCH_DISTRIBUTED_DEBUG=DETAIL -> logs collective shapes/order per rank; a shape or ordering
#                               mismatch between ranks is the leading hypothesis, since the per-GPU
#                               computation is IDENTICAL to the single-GPU run that does not fault.
#   NCCL_DEBUG=WARN          -> surfaces NCCL's own complaint if it is the origin.
#
# ⚠️ If this runs clean past ~1000 steps, that is itself informative: it would mean blocking launches
# suppress it, i.e. a RACE rather than a deterministic out-of-bounds index.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_sh
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR too long"; exit 78; }
# ⭐ expandable_segments is a LEADING SUSPECT for the 2-GPU illegal access. It changes the CUDA
# allocator to virtual-memory-backed expandable segments, which has documented interactions with
# NCCL (which registers buffers for IPC/RDMA); a segment being remapped under a registered buffer
# gives exactly this signature -- deterministic, multi-GPU only, surfacing on a later free.
# Single-GPU has run 6300 steps clean with identical per-GPU shapes, which fits.
# ALLOC_CONF=none unsets it entirely for the one-variable test.
if [ "${ALLOC_CONF:-expandable}" = "none" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
    echo "PYTORCH_CUDA_ALLOC_CONF: UNSET (expandable_segments disabled)"
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
fi

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING TORCH_DISTRIBUTED_DEBUG=$TORCH_DISTRIBUTED_DEBUG"

# Fresh store: the crash appeared ~500 steps from a FRESH start, so reproduce that trajectory
# rather than resuming into the middle of a different one.
STORE=/orcd/scratch/orcd/011/chenxiou/c2c_repro
rm -rf "$STORE/c2c_repro"
mkdir -p "$STORE"
torchrun --standalone --nproc_per_node=2 \
    scratchpad/train_c2c.py --devices 2 --accum 8 --n_diff 8 \
    --lr 0.0006 --precision bf16-mixed --val_every 100 --warmup 1000 \
    --store "$STORE" --name c2c_repro --n_dump "${NDUMP:-2}"
rc=$?
echo "REPRO_EXIT=$rc"
exit $rc
