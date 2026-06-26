#!/bin/bash
# WS1 full AF2Rank Rosetta-decoy ranking: slim vs stock, 4-way across GPUs.
# Splits the 133 targets into 4 shards (one GPU each); each shard scores slim+stock.
# Decoy set: ~/data/af2rank_rosetta/decoys/<target>/*.pdb + decoys/natives/<target>.pdb.
cd /home/jupyter-chenxi/proteina
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold:/home/jupyter-chenxi/proteina
ulimit -n 65536
SLIM=/home/jupyter-chenxi/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/best-037-009500.ckpt
DR=/home/jupyter-chenxi/data/af2rank_rosetta/decoys
ND=/home/jupyter-chenxi/data/af2rank_rosetta/decoys/natives
US=/home/jupyter-chenxi/USalign/USalign
T=/home/jupyter-chenxi/data/af2rank_rosetta/targets.txt
OUT=/home/jupyter-chenxi/runs/ws1_af2rank
MAXD=${MAXD:-50}
RECYCLES=${RECYCLES:-1}  # AF2Rank protocol (total iters = recycles+1)
rm -rf "$OUT"; mkdir -p "$OUT"
split -n l/4 -d "$T" "$OUT/targets_part_"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/run_slim_af2rank_calibration.py \
    --decoy_root "$DR" --native_dir "$ND" --targets_file "$OUT/targets_part_0$i" \
    --slim_ckpt_path "$SLIM" --usalign_path "$US" --max_decoys "$MAXD" --recycles "$RECYCLES" \
    --modes slim,stock --out_dir "$OUT/part_$i" > "$OUT/part_$i.log" 2>&1 &
done
wait
echo "ALL PARTS DONE"
