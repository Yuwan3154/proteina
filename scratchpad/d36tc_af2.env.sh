# D36tc AF2Rank env on Engaging (sourced by the af2rank / Control C sbatch): the D36 AF2-arm env (cue_openfold,
# CUDA 13.0.1, flash-attn off = the D2 recipe) + the AF2 params dir via the default-off AF2RANK_PARAMS_DIR override.
module load miniforge/25.11.0-0
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true; conda activate cue_openfold; set -u
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export OPENFOLD_DISABLE_FLASH_ATTN=1
export AF2RANK_PARAMS_DIR=/orcd/scratch/orcd/011/chenxiou/params
export USALIGN_PATH=/home/chenxiou/.local/bin/USalign
export PATH=/home/chenxiou/.local/bin:$PATH
export PYTHONUNBUFFERED=1
