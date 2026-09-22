#!/bin/bash
#SBATCH --job-name=freezetri
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
# Freeze the CURRENT CB tri model's EMA weights (tri_cb8synth_v5 last-EMA.ckpt, rewritten every
# 50 steps) so every Stage-A pass evaluates one fixed file. Accepted only on an md5-consistent copy.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
SRC=$S/proteina_tri/store/tri_cb8synth_v5/checkpoints/last-EMA.ckpt
mkdir -p $S/stageA/ckpt
TMP=$S/stageA/ckpt/.tmp_${SLURM_JOB_ID}.ckpt
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
ok=0
for attempt in 1 2 3; do
    cp "$SRC" "$TMP" || { sleep 30; continue; }
    [ "$(md5sum < "$SRC")" = "$(md5sum < "$TMP")" ] && { ok=1; break; }
    rm -f "$TMP"; sleep 30
done
[ "$ok" = 1 ] || { echo "FATAL: no consistent copy"; exit 3; }
GS=$(python -c "import torch,sys; ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(int(ck['global_step']))" "$TMP") || exit 4
DST=$S/stageA/ckpt/new_tri_last-EMA_step${GS}.ckpt
[ -e "$DST" ] && { echo "FATAL: $DST exists"; rm -f "$TMP"; exit 2; }
mv "$TMP" "$DST"; echo "[frozen] global_step=$GS -> $DST"
echo "FREEZE_EXIT=0"
