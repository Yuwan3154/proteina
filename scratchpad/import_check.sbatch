#!/bin/bash
#SBATCH --job-name=impchk
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# Protects the LIVE arms: model_trainer_base.py was edited, and both arms re-import it at
# their next chain segment. A bad import would crash the rollover, not this job.
source /etc/profile >/dev/null 2>&1
set -uo pipefail
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u
cd /orcd/scratch/orcd/011/chenxiou/proteina_cmhier
export PYTHONPATH=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier${PYTHONPATH:+:$PYTHONPATH}
echo "--- stdlib-only unit tests in the REAL env ---"
python tests/test_logtrace.py
echo "--- importing the edited live module ---"
python -c "
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase
from proteinfoundation.utils import logtrace
print('model_trainer_base imported OK')
print('logtrace.enabled() with LOGTRACE unset ->', logtrace.enabled())
print('ModelTrainerBase.log overridden ->', 'log' in ModelTrainerBase.__dict__)
import inspect
print('log() references logtrace ->', 'logtrace' in inspect.getsource(ModelTrainerBase.log))
"
