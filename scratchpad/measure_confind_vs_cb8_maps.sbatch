#!/bin/bash
#SBATCH --job-name=cfcbmap
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
# CPU-only. ConFind vs CB-8A maps on the same processed chains via the real ContactMapTransform
# (diagonal, |i-j|=1/2, density, missing contact_map_confind, CPU-ConFind vs Frame2ConFind signature).
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
# activate.sh refuses without a GPU; the bare venv is enough on CPU
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
python "$REPO/scratchpad/measure_confind_vs_cb8_maps.py" --n "${NCHAINS:-400}"
echo "CFCBMAP_EXIT=$?"
