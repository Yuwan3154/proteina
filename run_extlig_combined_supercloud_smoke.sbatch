#!/bin/bash
#SBATCH --job-name=elft_sc_smoke
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

# 4-node x 2-V100 multi-node DDP smoke test for the ext_lig SuperCloud port (combined PDB+AFDB,
# gloo backend, fp32). Validates: Hydra config loads, checkpoint resume (global_step 123200),
# packed-mmap data loading, cross-node DDP init, the NEW checkpoint-persisted inverse-frequency
# no-CAT sampler constructing/logging correctly on real data, AND (2026-07-05, after a real crash)
# a full validation epoch actually completing and logging validation_sampling/tmscore_median via
# USalign -- opt.max_steps=123900 (700 verification steps) runs safely past the ~step-123800 point
# where validation previously fired and crashed (USalign wasn't on PATH; now fixed). The config's
# own max_steps=187600 is the REAL target for the full run (run_extlig_combined_supercloud_full.sbatch),
# not used here.

set -uo pipefail
RUN=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_pdb_purge-test_warmup_maxlen-256_cutoff-190828_catbalanced_combined_supercloud
REPO=/home/gridsan/cou/proteina

mkdir -p "$REPO/store/$RUN/checkpoints"

source /etc/profile.d/modules.sh
module load anaconda/2023a
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cathfold_extlig

echo "[$(date)] NODES: $SLURM_JOB_NODELIST"

export DATA_PATH=/home/gridsan/cou/proteina/data
export PACK_PATH=$DATA_PATH/pdb_dFS_combined_pack/combined.pack
# SuperCloud compute nodes have no outbound internet (confirmed: curl to api.wandb.ai fails,
# unlike Engaging) -- wandb.init() hangs retrying forever without this. Sync later from a node
# with internet access via `wandb sync <run dir>` if the logs are wanted in the cloud dashboard.
export WANDB_MODE=offline
# USalign (validation_sampling TM-score, feeds the best-tmscore ModelCheckpoint monitor) --
# copied from Engaging's working binary to ~/.local/bin, not on PATH by default in this shell.
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
    opt.max_steps=123900
RC=$?
echo "[$(date)] torchrun exited rc=$RC"
exit $RC
