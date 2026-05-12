#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=40
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/slurm-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/slurm-%A_%a.err

# Load your shell environment to activate your Conda environment
source /etc/profile
module load conda/Python-ML-2025b-pytorch

# Then use the module command to load the module needed for your work
source activate cathfold
export OMP_NUM_THREADS=40

cd /home/gridsan/cou/proteina

# SLURM_ARRAY_TASK_ID (0 or 1) and SLURM_ARRAY_TASK_COUNT (2)
# are auto-detected by parallel_proteina_inference.py for sharding.
# Each array task runs on 1 node with 2 GPUs, processing its shard
# of proteins sorted by length for balanced workload.
# OOM errors automatically halve max_nsamples and retry.
python proteinfoundation/prediction_pipeline/compare_replicas_topk.py \
  --input /home/gridsan/cou/data/af2rank_single/af2rank_single_set_combined_tms_cutoff-190828_in_train_length.csv \
  --id_col natives_rcsb \
  --cif_dir /home/gridsan/cou/data/af2rank_single/pdb \
  --use_ground_truth \
  --replica_a_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_4-seq-S25_128-eff-bs_pdb_last_045-noise \
  --replica_b_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
  --replica_a_label 4-seq --replica_b_label 21-seq \
  --max_sample_index 512 \
  --output_dir prediction/rosetta_decoys_replica_compare_4-seq_vs_21-seq \
  --ptm_cutoff 0.7 \
  --num_workers 40

python proteinfoundation/prediction_pipeline/compare_replicas_topk.py \
  --input /home/gridsan/cou/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_monomer_length_50-640_tm-05_coverage-08_identity-07_cutoff-190828_in-train.csv \
  --id_col pdb \
  --cif_dir /home/gridsan/cou/data/bad_afdb/pdb \
  --use_ground_truth \
  --replica_a_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_4-seq-S25_128-eff-bs_pdb_last_045-noise \
  --replica_b_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
  --replica_a_label 4-seq --replica_b_label 21-seq \
  --max_sample_index 512 \
  --output_dir prediction/failed_afdb_replica_compare_4-seq_vs_21-seq \
  --ptm_cutoff 0.7 \
  --num_workers 40

python proteinfoundation/prediction_pipeline/compare_replicas_topk.py \
  --input /home/gridsan/cou/data/af2rank_single/af2rank_single_set_combined_tms_cutoff-190828_in_train_length.csv \
  --id_col natives_rcsb \
  --cif_dir /home/gridsan/cou/data/af2rank_single/pdb \
  --use_ground_truth \
  --replica_a_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
  --replica_a_label 21-seq \
  --max_sample_index 256 \
  --output_dir prediction/rosetta_decoys_replica_compare_21-seq-two-halves \
  --ptm_cutoff 0.7 \
  --num_workers 40

python proteinfoundation/prediction_pipeline/compare_replicas_topk.py \
  --input /home/gridsan/cou/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_monomer_length_50-640_tm-05_coverage-08_identity-07_cutoff-190828_in-train.csv \
  --id_col pdb \
  --cif_dir /home/gridsan/cou/data/bad_afdb/pdb \
  --use_ground_truth \
  --replica_a_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
  --replica_a_label 21-seq \
  --max_sample_index 256 \
  --output_dir prediction/failed_afdb_replica_compare_21-seq-two-halves \
  --ptm_cutoff 0.7 \
  --num_workers 40
