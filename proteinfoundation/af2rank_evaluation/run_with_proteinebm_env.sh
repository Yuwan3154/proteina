#!/bin/bash
# Initialize conda and activate ProteinEBM environment
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"

# Default env name per ProteinEBM README, but allow override
source activate "${PROTEINEBM_CONDA_ENV:-cue_openfold}"

exec "$@"



