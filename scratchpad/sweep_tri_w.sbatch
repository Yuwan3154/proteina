#!/bin/bash
#SBATCH --job-name=tri_sw
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_tri/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_tri/logs/%x-%j.err
#SBATCH --open-mode=append
# Short single-GPU arms of the tri_cb8synth recipe to pick the auxiliary loss weights (user 2026-09-10:
# "do short run small sweeps on salloc GPUs"). Every arm is identical except ALIGN_W / MLM_W, so the
# comparison is the main contact loss trajectory vs the w=0 control at equal steps. Also the 1-GPU
# SMOKE of the whole data path (MAX_STEPS small, SMOKE=1 pointing at the 100-chain smoke index).
# usage: ARM=ctrl ALIGN_W=0 MLM_W=0 MAX_STEPS=1500 sbatch scratchpad/sweep_tri_w.sbatch
#        ARM=smoke ALIGN_W=0.3 MLM_W=0.3 MAX_STEPS=20 SYNTH_INDEX_DIR=<smoke dir> sbatch ...
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SYNTH_INDEX_DIR="${SYNTH_INDEX_DIR:-/orcd/scratch/orcd/011/chenxiou/synth_index}"
export ALIGN_W="${ALIGN_W:?}" MLM_W="${MLM_W:?}"
ARM="${ARM:?ARM names the run (tri_cb8sw_<ARM>)}"
MAX_STEPS="${MAX_STEPS:?}"
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_sm120
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_sm120
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs" "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID} arm=$ARM align_w=$ALIGN_W mlm_w=$MLM_W max_steps=$MAX_STEPS index=$SYNTH_INDEX_DIR"
git log --oneline -1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 1 rank: effective batch = accum 8 (the real run uses 4 ranks = 32). Equal across arms, so the
# RELATIVE comparison holds; absolute loss values are not the 4-GPU run's.
torchrun --standalone --nproc_per_node=1 proteinfoundation/train.py \
    --config_name training_contact_tri_full384_cb8synth_v1 \
    --ngpus_per_node 1 --nnodes 1 \
    run_name_="tri_cb8sw_${ARM}" \
    hardware.ngpus_per_node_=1 opt.max_steps="$MAX_STEPS" log.checkpoint=False \
    opt.val_check_interval_optim_steps=100 log.log_every_n_steps=10 log.wandb_project=tri_cb8_sweep \
    log.single_step_metrics_every_n_steps=10 ${OVERRIDES:-}
rc=$?
echo "TRI_SWEEP_EXIT=$rc"
exit $rc
