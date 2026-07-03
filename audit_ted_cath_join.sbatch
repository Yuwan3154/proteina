#!/bin/bash
#SBATCH --job-name=elft_ted_audit
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/home/chenxiou/proteina/store/ted_audit/audit-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/ted_audit/audit-%j.err
#SBATCH --requeue

# Single sequential pass over the 128GB TED domain-summary TSV (pool disk, can be slow) --
# CPU-only, generous 8h wall as a safety margin, not expected to need it.
set -uo pipefail
mkdir -p /home/chenxiou/proteina/store/ted_audit
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cue_openfold

cd /orcd/scratch/orcd/011/chenxiou/proteina
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
echo "[$(date)] starting TED join audit on $(hostname)"
python scripts/analysis/audit_ted_cath_join_afdb.py --data-path "$DATA_PATH" --n-sample 500
RC=$?
echo "[$(date)] exited rc=$RC"
exit $RC
