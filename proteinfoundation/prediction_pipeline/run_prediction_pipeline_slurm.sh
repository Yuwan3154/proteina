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
module load conda/Python-ML-2025b-pytorch
module switch cuda/13.0 cuda/12.6

# Then use the module command to load the module needed for your work
source activate cathfold
export OMP_NUM_THREADS=20
export CUTLASS_PATH=/home/gridsan/cou/openfold/cutlass
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/gridsan/cou/proteina

# SLURM_ARRAY_TASK_ID (0 or 1) and SLURM_ARRAY_TASK_COUNT (2)
# are auto-detected by parallel_proteina_inference.py for sharding.
# Each array task runs on 1 node with 2 GPUs, processing its shard
# of proteins sorted by length for balanced workload.
# OOM errors automatically halve max_nsamples and retry.
python proteinfoundation/prediction_pipeline/run_prediction_pipeline.py \
    --input /home/gridsan/cou/data/afdb/afdb_model_org_plddt-05_aiupred-02_max768.csv \
    --id_col accession_id \
    --inference_config inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise \
    --output_dir /home/gridsan/cou/proteina/prediction/inference_seq_cond_sampling_ca_dssp_extlig_no-sin-pos-emb_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_pdb_last_045-noise/ \
    --proteinebm_checkpoint /home/gridsan/cou/ProteinEBM/weights/proteinebm_v2_cathmd_weights.pt \
    --proteinebm_config /home/gridsan/cou/ProteinEBM/protein_ebm/config/proteinebm_v2_cathmd_config.yaml \
    --proteinebm_t 0.05 \
    --top_k 16 \
    --no-use_deepspeed_evoformer_attention \
    --use_cuequivariance_attention \
    --use_cuequivariance_multiplicative_update \
    --force_compile \
    --backend openfold \
    --recycles 6 \
    --proteinebm_batch_size 16 \
    --direct_python \
    --num_gpus 2 \
    --num_workers 40 \
    --skip_diversity 
