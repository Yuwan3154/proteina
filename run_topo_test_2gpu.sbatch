#!/bin/bash
#SBATCH --job-name=topo30M_test
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err

# Authorised test run: the 28.07M topology-conditioned contact model, `both` featurization arm,
# on 2 L40S. Current settings verbatim -- the point is to confirm it trains and to measure step
# time and memory headroom, not to change hyperparameters.
#
# 6h is mit_normal_gpu's maximum. The cluster-wide maintenance reservation begins 2026-08-18T00:00,
# so this must be submitted with enough margin to finish before then.
set -uo pipefail
trap 'echo "[$(date)] SIGUSR1 received; waiting for SIGTERM."' USR1

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
# One BLAS thread per rank: the dataloader workers do real tensor work per sample (the circuit
# topology recomputation), and letting each open a full thread pool is what drove a 96-core node
# to load average 1773 during the index build.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/torchinductor_cache_cmhier
export TRITON_CACHE_DIR=/orcd/scratch/orcd/011/chenxiou/triton_cache_cmhier
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$REPO/logs"

# This checkout must win over the editable install of `proteinfoundation` in cue_openfold, which
# points at /orcd/scratch/.../proteina (branch main) and has no topology_reference module. Running
# `python proteinfoundation/train.py` puts the SCRIPT's directory on sys.path, not the repo root,
# so without this hydra resolves targets against the other tree.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"
echo "[$(date)] host=$(hostname) jobid=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

python -c "import proteinfoundation; print('proteinfoundation from', proteinfoundation.__file__)"

# torchrun, not bare srun: train.py reads RANK/WORLD_SIZE/LOCAL_RANK, which srun does not set (it
# sets SLURM_PROCID/SLURM_NTASKS instead), and the run would otherwise proceed with rank=-1.
exec torchrun --standalone --nproc_per_node=2 proteinfoundation/train.py \
    --config_name training_contact_hier_topology_30M_v1 \
    --ngpus_per_node 2 \
    --nnodes 1
