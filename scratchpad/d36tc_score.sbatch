#!/bin/bash
#SBATCH --job-name=d36tc_score
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc final scoring (CPU): d36tc_score.py -> $P/d36tc_per_target.csv + counts per arm.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
REPO="${REPO:-$S/proteina_d36tc}"
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true; conda activate cue_openfold; set -u
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
python scratchpad/d36tc_score.py --out "$P/d36tc_per_target.csv"
rc=$?
echo "SCORE_RC=$rc"
exit $rc
