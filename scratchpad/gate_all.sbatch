#!/bin/bash
#SBATCH --job-name=gateall
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# ⛔ Runs against a SCRATCH CLONE at the candidate commit, never the live repo. The A+B training run
# reads its Python from proteina_sh at job start, so pushing an ungated commit before 14:05 would
# change what trains, unattended. Gate first, push only on ALL GREEN.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
cd "${GATE_REPO:?GATE_REPO must point at the clone to test}"
echo "gating: $(pwd) @ $(git log --oneline -1)"

python scratchpad/test_ref_offsets.py;             R1=$?
python scratchpad/test_chirality_signal.py;        R2=$?
python scratchpad/test_contact2coord.py;           R3=$?
python scratchpad/test_local_attention.py;         R4=$?
python scratchpad/test_qk_norm.py;                 R5=$?
python scratchpad/test_diffusion_batch_alignment.py; R6=$?
python scratchpad/test_warmstart_load.py;          R7=$?

echo
echo "RESULTS refoff=$R1 chir=$R2 c2c=$R3 local=$R4 qk=$R5 diffalign=$R6 warmstart=$R7"
if [ $((R1+R2+R3+R4+R5+R6+R7)) -eq 0 ]; then echo "ALL GREEN"; else echo "SOMETHING FAILED"; fi
