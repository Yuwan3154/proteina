#!/bin/bash
#SBATCH --job-name=elft_ted_audit_v2
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=47:00:00
#SBATCH --output=/home/chenxiou/proteina/store/ted_audit/audit-v2-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/ted_audit/audit-v2-%j.err
#SBATCH --requeue

# v2: job 17079103 (v1) TIMED OUT at 8h without finishing -- the 128GB TED TSV lives on the
# known-degraded hstor013 pool disk (confirmed via a direct dd test: <7MB/s, couldn't read 2GB
# in 5 min). Fix: (1) rsync a resumable (--partial) copy to scratch first, so a preemption/retry
# doesn't restart the slow read from zero; (2) run the audit against the scratch copy;
# (3) unbuffered/flushed progress every 5 min in the script itself, so this is observable this
# time. 47h wall (mit_preemptable's cap is 48h) is a generous margin given uncertain disk speed.
set -uo pipefail
mkdir -p /home/chenxiou/proteina/store/ted_audit
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cue_openfold

cd /orcd/scratch/orcd/011/chenxiou/proteina
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
SRC=$DATA_PATH/d_FS/ted_365m.domain_summary.cath.globularity.taxid.tsv
DST_DIR=/orcd/scratch/orcd/011/chenxiou/proteina_hot_data/d_FS
DST=$DST_DIR/ted_365m.domain_summary.cath.globularity.taxid.tsv
mkdir -p "$DST_DIR"

echo "[$(date)] copying TSV pool -> scratch (resumable) on $(hostname)"
rsync -avP --partial "$SRC" "$DST"
echo "[$(date)] copy done, starting audit against scratch copy"

python -u scripts/analysis/audit_ted_cath_join_afdb.py --data-path "$DATA_PATH" --tsv-path "$DST" --n-sample 500
RC=$?
echo "[$(date)] exited rc=$RC"
exit $RC
