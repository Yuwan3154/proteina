#!/bin/bash
#SBATCH --job-name=elft_sc_r2
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=4-04:00:00
#SBATCH --output=/home/gridsan/cou/proteina/store/%x-%j.out
#SBATCH --error=/home/gridsan/cou/proteina/store/%x-%j.err
#SBATCH --requeue

# REAL full launch of ext_lig SuperCloud round 2 (fresh from v1.6, seq_emb_dim=21, 3:1
# no-CAT:CAT ratio, 1000 TOTAL epochs = 642,000 steps). 4 nodes x 2 V100, gloo backend, fp32 --
# same proven infra as round 1. Expected duration ~251h (~10.5 days) at round 1's measured
# 1.41 s/step throughput -- WAY beyond this partition's 100h (4d4h) single-job cap, so this
# WILL need multiple chained resubmits. --requeue covers scheduler-initiated preemption
# (PreemptMode=REQUEUE on this partition); internal-crash resubmits are covered by the
# separate elft_watchdog_r2 SLURM job (run_elft_watchdog_r2.sbatch, download partition,
# sacct-based completion detection -- same fix as round 1's watchdog after the false-completion
# bug found there).

set -uo pipefail
RUN=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_21-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_combined_3to1nocat_supercloud
REPO=/home/gridsan/cou/proteina

mkdir -p "$REPO/store/$RUN/checkpoints"

source /etc/profile.d/modules.sh
module load anaconda/2023a
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cathfold_extlig

echo "[$(date)] NODES: $SLURM_JOB_NODELIST"

export DATA_PATH=/home/gridsan/cou/proteina/data
export PACK_PATH=$DATA_PATH/pdb_dFS_combined_pack/combined.pack
export WANDB_MODE=offline
export PATH="$HOME/.local/bin:$PATH"
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + (RANDOM % 40000) ))
echo "[$(date)] rendezvous $MASTER_ADDR:$MASTER_PORT"

cd "$REPO"
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    proteinfoundation/train.py \
    --config_name "training_$RUN" \
    --ngpus_per_node 2 \
    --nnodes "$SLURM_NNODES" \
    --batch_size 2 \
    --accumulate_grad_batches 1 \
    --resume_option allow
RC=$?
echo "[$(date)] torchrun exited rc=$RC"
exit $RC
