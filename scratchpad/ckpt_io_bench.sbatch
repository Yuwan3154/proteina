#!/bin/bash
#SBATCH --job-name=ckptio
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# How expensive is one checkpoint write, really?
#
# The user asked to confirm checkpoints land on SSD. The mount is NFS4 over RDMA (fstor018.ib), so
# the backing media is NOT visible from the client -- `df`/`findmnt` cannot answer it. What actually
# matters for the decision is the THROUGHPUT the training job sees, so measure that instead of
# arguing about the media. A real c2c checkpoint is 3.2 GB.
#
# ⛔ Writes into the SAME directory the checkpoints go to, not /tmp -- a node-local /tmp would
# measure the wrong filesystem entirely and flatter the result.
# ⛔ conv=fsync so the timing includes the flush to the server, not just filling page cache.
set -uo pipefail
DIR=/orcd/scratch/orcd/011/chenxiou/c2c_store/.iobench
mkdir -p "$DIR"
F="$DIR/probe-$SLURM_JOB_ID.bin"

echo "== target =="
findmnt -T "$DIR" -o SOURCE,FSTYPE 2>/dev/null | tail -1

echo "== write 3.2 GB (one checkpoint), fsync included =="
dd if=/dev/zero of="$F" bs=1M count=3200 conv=fsync 2>&1 | tail -1

echo "== drop what we can, then read it back =="
dd if="$F" of=/dev/null bs=1M 2>&1 | tail -1

rm -f "$F"
rmdir "$DIR" 2>/dev/null || true
echo "CKPTIO_EXIT=$?"
