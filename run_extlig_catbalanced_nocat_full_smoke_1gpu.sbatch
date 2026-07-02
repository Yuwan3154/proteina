#!/bin/bash
#SBATCH --job-name=elft_nocat_smoke1g
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:20:00
#SBATCH --output=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/smoke1g-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/smoke1g-%j.err
#SBATCH --no-requeue

# 1-GPU variant of the smoke test -- mit_normal_gpu's per-user QOS caps chenxiou at 2 GPUs
# TOTAL (shared with another concurrent agent's job), and that job was holding 1 GPU when
# the 2-GPU smoke test was submitted, so it sat PENDING (QOSMaxCpuPerUserLimit). This
# validates everything the 2-GPU version would EXCEPT the DDP-2-rank nocat cluster split
# (already unit-tested in isolation, simulated 3-rank DDP, this session). Batch/accum
# adjusted to match the ORIGINAL 1-GPU effective batch (8 x 2 = 16).

set -uo pipefail
RUN=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont
REPO=/home/chenxiou/proteina

mkdir -p "$REPO/store/$RUN/slurm"

module load miniforge cuda/13.0.1
source "$(conda info --base 2>/dev/null || echo /opt/conda)/etc/profile.d/conda.sh"
conda activate cue_openfold

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

cd "$REPO"
echo "[$(date)] SMOKE TEST (1 GPU): Launching $RUN on $(hostname) jobid=${SLURM_JOB_ID:-?}"

RDZV_PORT=$(( 20000 + (RANDOM % 40000) ))
echo "[$(date)] torchrun rendezvous 127.0.0.1:$RDZV_PORT"
torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --rdzv-id="${SLURM_JOB_ID:-$$}" proteinfoundation/train.py \
    --config_name "training_$RUN" \
    --ngpus_per_node 1 \
    --nnodes 1 \
    --batch_size 8 \
    --accumulate_grad_batches 2 \
    --resume_option allow
RC=$?
echo "[$(date)] SMOKE TEST (1 GPU): torchrun exited rc=$RC"
exit $RC
