#!/bin/bash
#SBATCH --job-name=elft_sc_r2_smoke
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=00:45:00
#SBATCH --output=/home/gridsan/cou/proteina/store/%x-%j.out
#SBATCH --error=/home/gridsan/cou/proteina/store/%x-%j.err
#SBATCH --no-requeue

# 4-node x 2-V100 multi-node DDP smoke test for ext_lig SuperCloud round 2 (fresh start from
# v1.6, seq_emb_dim=21, 3:1 no-CAT:CAT ratio). Validates: (a) the pretrain_ckpt_path branch
# actually fires (no last.ckpt present yet in this run's checkpoints/ dir); (b) the
# shape-tolerant loader (train.py, commit 06be98e) correctly cold-starts seq_emb (2->21 shape
# mismatch) + ext_lig embeddings while warm-starting everything else from v1.6 -- check the log
# for the "Cold-starting N shape-mismatched param(s)" line and confirm seq_emb/ext_lig are in
# that list and nothing else is; (c) the new nocat_bucket_inverse_freq_ratio=3.0 sampler
# constructs/logs correctly (expect "draw 3846/epoch, 3:1 with CAT topologies" in the log); (d)
# cross-node DDP init across 4 nodes; (e) at least one validation cycle (val_check_interval=1000)
# completes and writes a best-tmscore checkpoint without crashing (the real fix from round 1,
# commit dc7fcb0, still in effect). opt.max_steps=1100 override (NOT the config's real
# max_steps=642000 target) -- short verification only.

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
export DIAG_DATALOADER=1

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
    opt.max_steps=1100
RC=$?
echo "[$(date)] torchrun exited rc=$RC"
exit $RC
