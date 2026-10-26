#!/bin/bash
#SBATCH --job-name=c1b_stored_diag
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-04:00:00
#SBATCH --output=/home/gridsan/cou/logs/%x-%j.out
# synth_c1b_stored_diag.py on CPU, synth_c1_native.sbatch's environment. ARGS = the script's arguments; REPO, F2C_PARENT.
source /etc/profile
set -eo pipefail
: "${ARGS:?}" "${REPO:?}" "${F2C_PARENT:?}"
ENV_DIR=/home/gridsan/cou/cue-openfold-env
export CUE_HOST=supercloud
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export SIGMA_DATA=16.0
export PYTHONPATH="$REPO:$F2C_PARENT${PYTHONPATH:+:$PYTHONPATH}"
git -C "$REPO" log -1 --format="[repo] $REPO HEAD %h"
cd "$REPO"
rc=0
python scratchpad/synth_c1b_stored_diag.py $ARGS || rc=$?
echo "[rc] $rc"
exit $rc
