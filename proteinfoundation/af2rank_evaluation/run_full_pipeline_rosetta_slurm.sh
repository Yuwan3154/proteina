#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=40
#SBATCH --time=4-00:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/gridsan/cou/logs/slurm-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/slurm-%A_%a.err

# Load your shell environment to activate your Conda environment
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6

# Then use the module command to load the module needed for your work
source activate cathfold
# 40 cores / 2 GPUs = 20 threads per GPU worker
export OMP_NUM_THREADS=20

cd /home/gridsan/cou/proteina

# SLURM_ARRAY_TASK_ID (0 or 1) and SLURM_ARRAY_TASK_COUNT (2)
# are auto-detected by parallel_proteina_inference.py for sharding.
# Each array task runs on 1 node with 2 GPUs, processing its shard
# of proteins sorted by length for balanced workload.
# OOM errors automatically halve max_nsamples and retry.
python proteinfoundation/af2rank_evaluation/run_full_pipeline.py \
    --dataset_file /home/gridsan/cou/data/af2rank_single/af2rank_single_set_combined_tms_cutoff-190828_in_train_length.csv \
    --id_col natives_rcsb \
    --tms_col tms_single \
    --cif_dir /home/gridsan/cou/data/af2rank_single/pdb \
    --cross_protein_output_dir /home/gridsan/cou/proteina/inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise/rosetta_decoys_cross_protein_analysis \
    --inference_config inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
    --scorer proteinebm \
    --proteinebm_checkpoint /home/gridsan/cou/ProteinEBM/weights/proteinebm_v2_cathmd_weights.pt \
    --proteinebm_config /home/gridsan/cou/ProteinEBM/protein_ebm/config/proteinebm_v2_cathmd_config.yaml \
    --proteinebm_t 0.05 \
    --skip_diversity \
    --top_k 8 \
    --no-use_deepspeed_evoformer_attention \
    --use_cuequivariance_attention \
    --use_cuequivariance_multiplicative_update \
    --proteinebm_batch_size 16 \
    --recycles 6 \
    --force_compile \
    --backend openfold \
    --direct_python \
    --num_gpus 2
