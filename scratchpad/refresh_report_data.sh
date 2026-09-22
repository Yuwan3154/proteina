#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
# Re-fetch the training-status report's three data snapshots (user 2026-09-22: "Re-fetch data").
# MODE=tri (tri history CSV -> tri_epochs.json) | c2c (c2c_steps.json) | pal (tri_val_pal.json).
# ⛔ Each old snapshot is copied to .tmp/curves/bak_<jobid>/ FIRST: the published report was built
#    from it, and an overwrite without a copy would make that page unreproducible.
# ⛔ A real bash file, never `sbatch --wrap` (/bin/sh: `source` silently no-ops).
set -uo pipefail
T=/orcd/scratch/orcd/011/chenxiou/.tmp
B=$T/curves/bak_${SLURM_JOB_ID}
mkdir -p "$B"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export WANDB_SILENT=true
case "${MODE:?MODE=tri|c2c|pal}" in
  tri) cp -p $T/curves/tri_cb8synth_v5_history.csv $T/curves/tri_epochs.json "$B"/
       python $T/fetch_curves.py; echo "FETCH_EXIT=$?"
       python $T/extract_tri.py; echo "EXTRACT_EXIT=$?" ;;
  c2c) cp -p $T/curves/c2c_steps.json "$B"/
       python $T/fetch_c2c.py; echo "C2C_EXIT=$?" ;;
  pal) cp -p $T/tri_val_pal.json "$B"/
       python $T/tri_val_pal_history.py > $T/tri_val_pal.json.new; rc=$?
       echo "PAL_EXIT=$rc"
       # replace only on success and a non-empty result, so a failed fetch cannot blank the snapshot
       [ "$rc" = 0 ] && [ -s $T/tri_val_pal.json.new ] && mv $T/tri_val_pal.json.new $T/tri_val_pal.json
       python -c "import json; d=json.load(open('$T/tri_val_pal.json')); print('[pal] keys', sorted(d), 'epoch', d.get('epoch'))" ;;
esac
ls -la "$B" $T/curves/*.json $T/tri_val_pal.json
