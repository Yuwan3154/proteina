#!/bin/bash
#SBATCH --job-name=dumpsplit
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
cd "$REPO"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

# The max384 dataset is the one BOTH arms train on, so its split is the one the reported
# validation numbers come from.
python scratchpad/dump_splits.py \
    --dataset pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828 \
    --subdir pdb \
    --out_dir /orcd/scratch/orcd/011/chenxiou/valset_analysis/splits
