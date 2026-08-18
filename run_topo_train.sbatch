#!/bin/bash
#SBATCH --job-name=topo_ab
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60

# One arm of the pair-feature A/B. Pick the arm with CONFIG:
# One arm per GPU, so both arms fit inside the 2-GPU-per-user cap on mit_normal_gpu and run
# concurrently on identical data. Set FEAT and RUN:
#   FEAT=local_attn RUN=overfit_localattn
#   FEAT=conv_next  RUN=overfit_convnext
# Everything else is shared, so the comparison isolates the block featurizer.
set -uo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

: "${FEAT:?set FEAT to local_attn or conv_next}"
: "${RUN:?set RUN to the run name}"
CONFIG="${CONFIG:-training_contact_hier_topology_overfit_v1}"
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data

module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u

export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_cmhier
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_cmhier
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs"

# cue_openfold editable-installs proteinfoundation from a DIFFERENT checkout (branch main); running
# a script puts the script's directory on sys.path, not the repo root, so without this hydra
# resolves targets against that tree and cannot find the topology transform.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?} config=$CONFIG feat=$FEAT run=$RUN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import proteinfoundation; print('proteinfoundation from', proteinfoundation.__file__)"

# torchrun, not bare srun: train.py reads RANK/WORLD_SIZE/LOCAL_RANK, which srun does not set.
exec torchrun --standalone --nproc_per_node=1 proteinfoundation/train.py \
    --config_name "$CONFIG" \
    --ngpus_per_node 1 \
    --nnodes 1 \
    run_name_="$RUN" \
    model.nn.block_featurizer="$FEAT"
