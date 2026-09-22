#!/bin/bash
#SBATCH --job-name=valendckpt
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
# CPU-only toy: does a validation-end checkpoint hold the EMA-swapped weights under the production
# env's lightning? Informs the retention-ladder design (a raw ladder must be step-triggered).
set -uo pipefail
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
python /orcd/scratch/orcd/011/chenxiou/proteina_tri/scratchpad/test_valend_ckpt_holds_swapped_weights.py
echo "VALEND_EXIT=$?"
