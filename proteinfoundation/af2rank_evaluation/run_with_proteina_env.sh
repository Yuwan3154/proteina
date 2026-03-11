#!/bin/bash
# Initialize conda and activate proteina environment
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate cue_openfold
exec "$@"
