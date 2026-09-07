#!/bin/bash
#SBATCH --job-name=coordcen
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# CPU-only. Dataset setup alone takes >2 min, so this cannot run inside an interactive ssh timeout.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
cd /orcd/scratch/orcd/011/chenxiou/proteina_sh
echo "=== coordinate centering / EDM scale ==="
python scratchpad/check_coord_centering.py --n 40
echo "DATACHIR_EXIT=$?"
