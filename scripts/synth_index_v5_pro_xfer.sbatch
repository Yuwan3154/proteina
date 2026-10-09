#!/bin/bash
#SBATCH --job-name=synthidx5p_xfer
#SBATCH --partition=mit_data_transfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/synth_index_v5_pro/logs/%x-%j.out
# v5 index is built on the A6000 (user 2026-10-09: the T2 templates are no longer on Engaging). Sends to SuperCloud
# ~/sse1d_idx_in/: engaging_trees.tar (the t2_regen + t2_extra npz the pool manifest uses, absolute paths kept),
# meta.tar (pool manifest, band index, auth alias, base topology index, v4 part 0 for the control), t2_list.txt (the
# T2 files, relative to templates_band, for the SuperCloud side), and sha256s.
set -uo pipefail
R=/orcd/scratch/orcd/011/chenxiou
M=$R/pool_manifest.tsv
W=$R/synth_index_v5_pro/xfer; mkdir -p "$W"
awk -F'\t' 'NR > 1 && $3 != "t2" {print substr($4, 2)}' "$M" | sort -u > "$W/engaging_files.txt"
awk -F'\t' 'NR > 1 && $3 == "t2" {sub(".*/templates_band/", "", $4); print $4}' "$M" | sort -u > "$W/t2_list.txt"
echo "engaging npz: $(wc -l < "$W/engaging_files.txt")  t2 npz: $(wc -l < "$W/t2_list.txt")"
ssh -o BatchMode=yes SuperCloud 'mkdir -p sse1d_idx_in'
rc=$?; [ $rc -eq 0 ] || exit $rc
scp -q "$W/t2_list.txt" SuperCloud:sse1d_idx_in/t2_list.txt || exit 3
date; tar cf - -C / -T "$W/engaging_files.txt" | tee >(sha256sum > "$W/engaging_trees.sha") | ssh -o BatchMode=yes SuperCloud 'cat > sse1d_idx_in/engaging_trees.tar'
rc=$?; echo "trees tar rc=$rc"; [ $rc -eq 0 ] || exit $rc
date; tar cf - -C / ${M#/} ${R#/}/pool_unplaced.tsv ${R#/}/t2_pool_auth_band.npz ${R#/}/synth_alias/auth_key.tsv \
    orcd/pool/006/chenxiou/proteina/data/pdb_train/topology_index.pt ${R#/}/synth_index_v4/parts/part_0000.pt \
    ${R#/}/synth_index_v4/parts/skips_0000.tsv | tee >(sha256sum > "$W/meta.sha") | ssh -o BatchMode=yes SuperCloud 'cat > sse1d_idx_in/meta.tar'
rc=$?; echo "meta tar rc=$rc"; [ $rc -eq 0 ] || exit $rc
sleep 5; cat "$W/engaging_trees.sha" "$W/meta.sha"
scp -q "$W/engaging_trees.sha" "$W/meta.sha" SuperCloud:sse1d_idx_in/ || exit 4
ssh -o BatchMode=yes SuperCloud 'ls -la sse1d_idx_in'
date; echo XFER_DONE
