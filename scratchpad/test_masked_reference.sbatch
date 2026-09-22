#!/bin/bash
#SBATCH --job-name=maskref
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
# CPU-only, production env. SRC lets it run from an extracted copy BEFORE the change is pulled into
# the live checkout (tri_cb8synth_v5's next segment imports proteinfoundation from proteina_tri).
set -uo pipefail
SRC=${SRC:-/orcd/scratch/orcd/011/chenxiou/proteina_tri}
cd "$SRC/scratchpad"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SYNTH_INDEX_DIR=/orcd/scratch/orcd/011/chenxiou/synth_index_v4
export SKIP_CONFIND_PRECOMPUTE=1 TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp PYTHONPATH=$SRC
python "$SRC/scratchpad/test_masked_reference.py"; echo "MASKREF_EXIT=$?"
python -c "import proteinfoundation.proteinflow.model_trainer_base as m, proteinfoundation.datasets.topology_reference as t; print('import ok', m.__file__)"
