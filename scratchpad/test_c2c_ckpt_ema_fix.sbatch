#!/bin/bash
#SBATCH --job-name=ckptfix
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
# CPU-only, production env. (1) the EMA-at-validation-end fix + ladder, (2) the existing cadence
# test, (3) train_c2c.py imports and parses args (so a broken import cannot reach tbeta's next segment).
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
# SP lets the tests run from an extracted copy BEFORE the change is pulled into the live checkout
SP=${SP:-$REPO/scratchpad}
cd "$REPO"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
python $SP/test_c2c_ckpt_ema_fix.py; echo "EMAFIX_EXIT=$?"
python $SP/test_ckpt_cadence.py; echo "CKPTCAD_EXIT=$?"
python $SP/train_c2c.py --help > /dev/null; echo "TRAIN_C2C_IMPORT_EXIT=$?"
python $SP/train_c2c.py --help | grep -E -- "--keep_steps|--keep_every"
