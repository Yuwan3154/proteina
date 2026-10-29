#!/bin/bash
#SBATCH --job-name=audit_sse_junk
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gridsan/cou/logs/audit_sse_junk-%j.out
#SBATCH --error=/home/gridsan/cou/logs/audit_sse_junk-%j.err
source /etc/profile
set -eo pipefail

W=/home/gridsan/cou/tri_synth_bench
REPO="${REPO:-/home/gridsan/cou/proteina_toporef_regimes}"
ENV_DIR=/home/gridsan/cou/cue-openfold-env
export CUE_HOST=supercloud
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
F2C_PARENT="${F2C_PARENT:-$W/pkgs}"
export PYTHONPATH="$REPO:$F2C_PARENT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd "$REPO" || exit 76
python scratchpad/audit_template_sse_junk.py --n-coord-samples 1000
