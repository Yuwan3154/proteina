#!/bin/bash
#
# Wrapper script to run generate_tms_single.py in the correct conda environment
#
# Usage:
#   bash run_generate_tms_single.sh

set -e

# Activate conda environment (proteina has BioPython)
source /opt/tljh/user/etc/profile.d/conda.sh
conda activate proteina

# Run the script
python generate_tms_single.py \
    --dataset_csv /home/jupyter-chenxi/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_plddt_50_length_50-768.csv \
    --indices_csv /home/jupyter-chenxi/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_indices.csv \
    --cif_dir /home/jupyter-chenxi/data/bad_afdb/pdb \
    --output_csv /home/jupyter-chenxi/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_plddt_50_length_50-768_tms.csv \
    --afdb_cache_dir /home/jupyter-chenxi/data/afdb_cache \
    --usalign_path USalign \
    --num_workers 8 \
    --verbose

echo ""
echo "✅ tms_single generation complete!"
echo "Output: /home/jupyter-chenxi/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_plddt_50_length_50-768_tms.csv"

