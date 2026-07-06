#!/bin/bash
#SBATCH --job-name=elft_keepalive_test
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/cou/proteina/store/%x-%j.out
#SBATCH --error=/home/gridsan/cou/proteina/store/%x-%j.err
#SBATCH --no-requeue

# ISOLATED verification of the gloo object-collective keepalive fix + re-enabled
# checkpoint_best_tmscore_median -- runs against a COPY of the production checkpoint in
# its own store dir (run_name_ override below), NOT the live production job's checkpoint,
# so this cannot interfere with or corrupt the actually-running production job. Target:
# 1000 steps (133600 -> 134600), spanning ~3 validation cycles -- previous crashes hit on
# the 1st or 2nd cycle every time, so 3 clean cycles is meaningful evidence the fix holds.

set -uo pipefail
RUN=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_combined_supercloud
TEST_RUN=elft_keepalive_test
REPO=/home/gridsan/cou/proteina

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
    --resume_option allow \
    run_name_="$TEST_RUN" \
    opt.max_steps=134600
RC=$?
echo "[$(date)] torchrun exited rc=$RC"
exit $RC
