#!/bin/bash
#SBATCH --job-name=zero_ca_check
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
# CPU-only. Real-weight check of the CA-feature ablation on tri_full384 @71,950 -EMA (Directive B).
# The script lives in proteina_tri; PYTHONPATH makes proteinfoundation resolve to the OLD checkout (30e4ec8).
set -uo pipefail
# activate.sh refuses without a GPU; the bare venv is enough for CPU torch.load
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
B=/orcd/scratch/orcd/011/chenxiou
OLD_REPO=$B/proteina_cmhier
cd "$OLD_REPO" || exit 77
PYTHONPATH="$OLD_REPO" python "$B/proteina_tri/scratchpad/check_zero_ca.py" \
    --index /orcd/pool/006/chenxiou/proteina/data/pdb_train/topology_index.pt \
    --ckpt "$OLD_REPO/store/tri_full384/checkpoints/chk_best_contact_precL_epoch=00000175_step=000000071950-EMA.ckpt"
echo "CHECK_RC=$?"
