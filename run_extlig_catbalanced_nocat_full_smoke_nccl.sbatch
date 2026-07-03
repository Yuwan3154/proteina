#!/bin/bash
#SBATCH --job-name=elft_nocat_smoke_nccl
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=00:08:00
#SBATCH --output=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/smoke-nccl-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/smoke-nccl-%j.err
#SBATCH --no-requeue

# NCCL-backend smoke test -- SHORT wall (8min) is a deliberate hard backstop against the
# indefinite-hang risk the user warned about (NCCL has hung before on this cluster). Actively
# monitored; will be scancel'd well before the wall limit if it looks stuck. NCCL_DEBUG=INFO
# for diagnosing exactly which operation stalls, if it does.

set -uo pipefail
RUN=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont
REPO=/home/chenxiou/proteina

mkdir -p "$REPO/store/$RUN/slurm"

module load miniforge cuda/13.0.1
source "$(conda info --base 2>/dev/null || echo /opt/conda)/etc/profile.d/conda.sh"
conda activate cue_openfold

echo "[$(date)] NODE: $(hostname)"
nvidia-smi topo -m 2>&1 || echo "nvidia-smi topo unavailable"

export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DIAG_DATALOADER=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/compute/so3/001/chenxi/torchinductor_cache_extlig
export TRITON_CACHE_DIR=/orcd/compute/so3/001/chenxi/triton_cache_extlig
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Attempt 1: vanilla NCCL, no defensive workarounds pre-applied, so the diagnostic signal is
# clean (if a workaround turns out to be needed, later attempts add exactly one at a time).
export NCCL_DEBUG=INFO

cd "$REPO"
echo "[$(date)] NCCL SMOKE TEST (attempt 1, vanilla): Launching $RUN on $(hostname) jobid=${SLURM_JOB_ID:-?}"

RDZV_PORT=$(( 20000 + (RANDOM % 40000) ))
echo "[$(date)] torchrun rendezvous 127.0.0.1:$RDZV_PORT"
torchrun --nnodes=1 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --rdzv-id="${SLURM_JOB_ID:-$$}" proteinfoundation/train.py \
    --config_name "training_$RUN" \
    --ngpus_per_node 2 \
    --nnodes 1 \
    --batch_size 8 \
    --accumulate_grad_batches 1 \
    --resume_option allow
RC=$?
echo "[$(date)] NCCL SMOKE TEST: torchrun exited rc=$RC"
exit $RC
