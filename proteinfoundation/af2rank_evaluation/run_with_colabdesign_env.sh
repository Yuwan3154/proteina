#!/bin/bash
# Initialize conda and activate colabdesign environment
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate colabdesign
unset LD_LIBRARY_PATH
exec "$@"
