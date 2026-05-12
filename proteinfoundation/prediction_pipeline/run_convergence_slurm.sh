#!/bin/bash

#SBATCH --job-name=convergence
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=40
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/slurm-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/slurm-%A_%a.err

# Convergence analysis (merged from run_eval_convergence_slurm.sh and
# run_prediction_convergence_slurm.sh). Each invocation runs
# proteinfoundation/prediction_pipeline/compare_replicas_topk.py over a
# (dataset, replica pair) combination.

source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate cathfold
export OMP_NUM_THREADS=40

cd /home/gridsan/cou/proteina

CMP=proteinfoundation/prediction_pipeline/compare_replicas_topk.py

# 1) AFDB (no GT) replica compare: 4-seq vs 21-seq
python $CMP \
    --input /home/gridsan/cou/data/afdb/afdb_model_org_plddt-05_aiupred-02_max768.csv \
    --id_col accession_id \
    --cif_dir /home/gridsan/cou/data/afdb/pdb \
    --use_ground_truth \
    --replica_a_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_4-seq-S25_128-eff-bs_pdb_last_045-noise \
    --replica_b_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
    --replica_a_label 4-seq --replica_b_label 21-seq \
    --max_sample_index 512 \
    --output_dir prediction/replica_compare_4-seq_vs_21-seq \
    --ptm_cutoff 0.7 \
    --num_workers 40

# 2) AFDB (no GT) replica compare: 21-seq two halves
python $CMP \
    --input /home/gridsan/cou/data/afdb/afdb_model_org_plddt-05_aiupred-02_max768.csv \
    --id_col accession_id \
    --cif_dir /home/gridsan/cou/data/afdb/pdb \
    --use_ground_truth \
    --replica_a_dir inference/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
    --replica_a_label 21-seq \
    --max_sample_index 512 \
    --output_dir prediction/replica_compare_21-seq-two-halves \
    --ptm_cutoff 0.7 \
    --num_workers 40

# 3) Rosetta decoys (af2rank_single) replica compare: 4-seq vs 21-seq
python $CMP \
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

# 4) Bad AFDB replica compare: 4-seq vs 21-seq
python $CMP \
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

# 5) Rosetta decoys replica compare: 21-seq two halves
python $CMP \
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

# 6) Bad AFDB replica compare: 21-seq two halves
python $CMP \
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
