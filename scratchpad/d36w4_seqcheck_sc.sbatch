#!/bin/bash
#SBATCH --job-name=d36w4_seqchk
#SBATCH -p xeon-p8
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o /home/gridsan/cou/d36/w4_seqcheck/logs/%x-%j.out
# D36 W4 candidate sequence eligibility on SuperCloud (CPU): the SuperCloud-hosted candidates of candidates_all.csv.
D=/home/gridsan/cou/d36/w4_seqcheck
module load conda/Python-ML-2025b-pytorch
conda deactivate 2>/dev/null
source activate cathfold
set -uo pipefail
[ "${CONDA_DEFAULT_ENV:-}" = cathfold ] || { echo "FATAL: env is '${CONDA_DEFAULT_ENV:-}', not cathfold"; exit 3; }
python -c "import sys, Bio; print('[env]', sys.prefix, Bio.__version__)"
cd "$D/repo"; git log -1 --format="[repo] %h"
python scratchpad/d36w4_seqcheck.py --candidates "$D/candidates_all.csv" --host SuperCloud --natives "$D/natives" \
  --out "$D/candidates_seqcheck_SuperCloud.csv" --workers 16
rc=$?
echo "SEQCHECK_RC=$rc"
exit $rc
