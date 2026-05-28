#!/bin/bash
#SBATCH --job-name=proteina_stage2
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2/slurm/%x-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2/slurm/%x-%j.err

# Stage-2 UDLM contact-flow finetune. Boots from stage-1 best ckpt via
# pretrain_ckpt_path (auto-resolved when checkpoint_mode=best is set, or
# pass an explicit ckpt path). Submit AFTER stage-1 finishes; this file
# is otherwise structurally identical to run_stage1_sbatch.sh.

set -euo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

mkdir -p /home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2/slurm

module load miniforge cuda/13.0.1
source $(conda info --base 2>/dev/null || echo /opt/conda)/etc/profile.d/conda.sh
conda activate cue_openfold

export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/compute/so3/001/chenxi/torchinductor_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR=/orcd/compute/so3/001/chenxi/triton_cache
mkdir -p $TRITON_CACHE_DIR

cd /home/chenxiou/proteina

echo "[$(date)] Launching stage-2 on $(hostname) jobid=${SLURM_JOB_ID:-?}"
echo "[$(date)] Restart count: ${SLURM_RESTART_COUNT:-0}"

# Stage-2 bootstraps from stage-1 best ckpt. Update pretrain_ckpt_path below
# to the actual best ckpt path after stage-1 finishes.
STAGE1_BEST=$(ls -t /home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1/checkpoints/best_tmscore_median*.ckpt 2>/dev/null | head -1)
if [ -z "$STAGE1_BEST" ]; then
    STAGE1_BEST=$(ls -t /home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1/checkpoints/last*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$STAGE1_BEST" ]; then
    echo "[ERROR] No stage-1 checkpoint found. Stage-2 needs stage-1 to complete first."
    exit 1
fi
echo "[$(date)] Stage-2 bootstrap from: $STAGE1_BEST"

# srun: SLURM spawns 4 tasks so Lightning wires 4-rank DDP via SLURM env.
exec srun python proteinfoundation/train.py \
    --config_name training_dssp_contact_20M_udlm_pb_v2 \
    --ngpus_per_node 4 \
    --nnodes 1 \
    --batch_size 4 \
    --accumulate_grad_batches 4 \
    --resume_option allow \
    pretrain_ckpt_path="$STAGE1_BEST" \
    af2_ipa_weights_path=/home/chenxiou/params/params_model_1_ptm.npz
