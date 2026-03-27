#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=40
#SBATCH --time=4-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/gridsan/cou/logs/slurm-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/slurm-%A_%a.err

# Load your shell environment to activate your Conda environment
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6

# Then use the module command to load the module needed for your work
source activate cue_openfold
# 40 cores / 2 GPUs = 20 threads per GPU worker
export OMP_NUM_THREADS=20

cd /home/gridsan/cou/proteina

# SLURM_ARRAY_TASK_ID (0 or 1) and SLURM_ARRAY_TASK_COUNT (2)
# are auto-detected by parallel_proteina_inference.py for sharding.
# Each array task runs on 1 node with 2 GPUs, processing its shard
# of proteins sorted by length for balanced workload.
# OOM errors automatically halve max_nsamples and retry.
python af2rank_evaluation/parallel_proteina_inference.py \
    --csv_file /home/gridsan/cou/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_monomer_length_50-640_tm-05_cutoff-190828_in-train.csv \
    --csv_column pdb \
    --cif_dir /home/gridsan/cou/data/bad_afdb/pdb \
    --inference_config inference_seq_cond_sampling_ca_beta-2.5-2.0_finetune-all_v1.6_default-fold_4-seq-S25_64-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise \
    --num_gpus 2 \
    --skip_existing
