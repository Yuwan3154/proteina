#!/bin/bash
#SBATCH --job-name=proteina_stage1
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1/slurm/%x-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1/slurm/%x-%j.err

# Stage-1 UDLM contact-flow training, sharded SSD layout, in_memory=True.
# Designed for preemption: SBATCH --requeue puts the job back in queue when
# evicted; train.py --resume_option=allow resumes from the last checkpoint
# automatically. Use ``sbatch run_stage1_sbatch.sh`` once and let it cycle.
#
# Per-step state (loss, val tmscore, ckpt) goes to ./store/<run_name_>/...
# wandb run id is persisted under that path so re-submissions chain into the
# same wandb run.

set -euo pipefail
trap 'echo "[$(date)] Received SIGUSR1 (preempt warning); checkpoint should land on next step. Letting job continue until SIGTERM."' USR1

mkdir -p /home/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1/slurm

module load miniforge cuda/13.0.1
source $(conda info --base 2>/dev/null || echo /opt/conda)/etc/profile.d/conda.sh
conda activate cue_openfold

export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1
# OMP/MKL: keep 1 thread per worker so DataLoader workers do not over-subscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Persistent torchinductor cache on shared SSD so cross-restart compiles are
# instant (vs ~20 min cold compile every requeue). Same dir for both stages.
export TORCHINDUCTOR_CACHE_DIR=/orcd/compute/so3/001/chenxi/torchinductor_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR
# DeepSpeed/Triton autotune cache (avoid NFS warning, share across restarts).
export TRITON_CACHE_DIR=/orcd/compute/so3/001/chenxi/triton_cache
mkdir -p $TRITON_CACHE_DIR

cd /home/chenxiou/proteina

echo "[$(date)] Launching stage-1 on $(hostname) GPUs $(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed s/,$//) jobid=${SLURM_JOB_ID:-?}"
echo "[$(date)] Restart count (requeue chain): ${SLURM_RESTART_COUNT:-0}"

# srun is required so SLURM spawns 4 tasks (one per GPU); Lightning reads
# SLURM_PROCID/SLURM_NTASKS to wire DDP. Running `python` alone gets a single
# rank (MEMBER 1/1) and Lightning's auto-spawn fallback is unreliable here.
# --resume_option=allow: create new run if no ckpt yet, resume if present.
exec srun python proteinfoundation/train.py \
    --config_name training_dssp_contact_20M_udlm_pb_v2_stage1 \
    --ngpus_per_node 4 \
    --nnodes 1 \
    --batch_size 4 \
    --accumulate_grad_batches 8 \
    --resume_option allow \
    af2_ipa_weights_path=/home/chenxiou/params/params_model_1_ptm.npz
