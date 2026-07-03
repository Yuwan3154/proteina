#!/bin/bash
#SBATCH --job-name=elft_pack_xfer
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/home/chenxiou/proteina/store/supercloud_transfer/xfer-%j.out
#SBATCH --error=/home/chenxiou/proteina/store/supercloud_transfer/xfer-%j.err
#SBATCH --requeue

# CPU-only rsync of the mmap-ready combined PDB+AFDB pack (505GiB) from Engaging to SuperCloud,
# for the ext_lig finetune's SuperCloud (4x2 V100) port. --partial keeps a resumable partial file
# across preemption/requeue; -P shows progress in the log. Never run on a login node (this rule).

set -uo pipefail
mkdir -p /home/chenxiou/proteina/store/supercloud_transfer

SRC=/orcd/data/so3/001/chenxi/pdb_dFS_combined_pack/
DST=SuperCloud:/home/gridsan/cou/proteina/data/pdb_dFS_combined_pack/

echo "[$(date)] starting rsync $SRC -> $DST on $(hostname)"
rsync -avP --partial "$SRC" "$DST"
RC=$?
echo "[$(date)] rsync exited rc=$RC"
exit $RC
