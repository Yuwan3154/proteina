#!/bin/bash
#SBATCH --job-name=cb_domaincrop_s1
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/orcd/pool/006/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.out
#SBATCH --error=/orcd/pool/006/chenxiou/proteina/store/dssp_contact_20M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined/slurm/%x-%j.err

# CATH-balanced domain-crop Stage-1, combined PDB+AFDB, on mit_normal_gpu (2-GPU
# per-user cap, non-preemptable, 6h job limit). Effective batch = 2 GPU * 4 micro
# * 16 accum = 128. Runs from the pip -e source checkout so train.py + imports
# both resolve to the same (merged-main) code -- no PYTHONPATH trap.
#
# 6h chain: --resume_option=allow resumes from last.ckpt; on a non-zero srun exit
# (time-limit SIGTERM or app crash) we resubmit a fresh job, but only if a
# checkpoint advanced this run (loop-guard against a deterministic crash). A clean
# finish / max_steps reached exits 0 and the chain self-terminates. No --requeue
# (mit_normal_gpu is non-preemptable; the resubmit chain covers the time limit).

set -euo pipefail
RUN=dssp_contact_20M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined
REPO=/orcd/pool/006/chenxiou/proteina
trap 'echo "[$(date)] SIGUSR1 (time-limit warning); checkpoint should land soon."' USR1

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
export TORCHINDUCTOR_CACHE_DIR=/orcd/compute/so3/001/chenxi/torchinductor_cache
export TRITON_CACHE_DIR=/orcd/compute/so3/001/chenxi/triton_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO"
echo "[$(date)] Launching $RUN on $(hostname) GPUs $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed s/,$//) jobid=${SLURM_JOB_ID:-?} restart=${SLURM_RESTART_COUNT:-0}"

CKPT=$REPO/store/$RUN/checkpoints/last.ckpt
START_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)

if srun python proteinfoundation/train.py \
    --config_name "$RUN" \
    --ngpus_per_node 2 \
    --nnodes 1 \
    --batch_size 4 \
    --accumulate_grad_batches 16 \
    --resume_option allow \
    af2_ipa_weights_path=/orcd/pool/006/chenxiou/params/params_model_1_ptm.npz; then
  RC=0
else
  RC=$?
fi
echo "[$(date)] srun exited rc=$RC"

END_MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
if [ "$RC" -ne 0 ] && [ "$END_MTIME" -gt "$START_MTIME" ]; then
  echo "[$(date)] non-zero exit after checkpoint progress; auto-resubmitting fresh 6h job"
  cd "$REPO" && sbatch run_stage1_catbalanced_domaincrop_sbatch.sh
elif [ "$RC" -ne 0 ]; then
  echo "[$(date)] non-zero exit with NO new checkpoint (loop-guard: not resubmitting). Investigate."
fi
exit $RC
