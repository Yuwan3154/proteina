#!/bin/bash
#SBATCH --job-name=fixab
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Gate for fixes A (masked AF3 reference offsets) and B (diffusion multiplicity 8 -> 16).
# ⛔ Nothing is relaunched until every one of these passes AND the n_diff=16 memory fits.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
cd /orcd/scratch/orcd/011/chenxiou/proteina_sh

echo "===================== 1. does anything see the mirror? ====================="
python scratchpad/test_chirality_signal.py; R1=$?

echo; echo "===================== 2. reference-offset gate (fix A) ====================="
python scratchpad/test_ref_offsets.py; R2=$?

echo; echo "===================== 3. pre-existing c2c gates still green ====================="
python scratchpad/test_contact2coord.py; R3=$?
python scratchpad/test_diffusion_batch_alignment.py; R4=$?
python scratchpad/test_local_attention.py; R5=$?
python scratchpad/test_qk_norm.py; R6=$?

echo; echo "===================== 4. n_diff memory + speed (fix B) ====================="
# ⛔ The published lr is calibrated for a gradient averaged over the diffusion mini-batch, so
# n_diff is not free to change on memory grounds alone -- measure before committing.
python scratchpad/bench_c2c.py --ndiff 8 16 --lengths 384; R7=$?

echo
echo "RESULTS chirality=$R1 refoffsets=$R2 c2c=$R3 diffalign=$R4 local=$R5 qk=$R6 bench=$R7"
if [ $((R1+R2+R3+R4+R5+R6+R7)) -eq 0 ]; then echo "ALL GREEN"; else echo "SOMETHING FAILED"; fi
