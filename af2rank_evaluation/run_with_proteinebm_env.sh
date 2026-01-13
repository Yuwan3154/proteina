#!/bin/bash
# Initialize conda and activate ProteinEBM environment
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"

# Default env name per ProteinEBM README, but allow override
conda activate "${PROTEINEBM_CONDA_ENV:-protebm}"

exec "$@"



