#!/bin/bash
#SBATCH --job-name=elft_nocat_smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=00:20:00
#SBATCH --output=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/smoke-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_nocat_full_cont/slurm/smoke-%j.err
#SBATCH --no-requeue

# ONE-OFF smoke test -- NOT a chain link, no resubmit-on-SIGTERM trap. 20-min wall time is
# just a safety cap; the run is expected to be killed by `scancel` once enough steps have
# been observed, or by SLURM at 20min if left unattended. Distinct job name
# (elft_nocat_smoke) so it's never confused with the real chain (elft_nocat).

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
echo "[$(date)] SMOKE TEST: Launching $RUN on $(hostname) jobid=${SLURM_JOB_ID:-?}"

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
echo "[$(date)] SMOKE TEST: torchrun exited rc=$RC"
exit $RC
