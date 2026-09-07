#!/bin/bash
#SBATCH --job-name=wstest
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# CPU-only, no GPU: this only builds the model and loads a state_dict. Deliberately NOT on a GPU
# partition -- the A+B training run is competing for scarce pi_so3 GPUs and must not queue behind
# its own pre-flight check.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
cd /orcd/scratch/orcd/011/chenxiou/proteina_sh
python scratchpad/test_warmstart_load.py
echo "WSTEST_EXIT=$?"
