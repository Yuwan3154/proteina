#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc AF2Rank (GPU, L40S): d36tc_af2rank.py for MODEL over LABELS (comma list of c2c labels). Resumable per pool, so a
# timed-out job is simply resubmitted. Partition/gres/time/name (d36tc_gpu, singleton) on the sbatch command line.
#   MODEL=model_1_ptm|model_2_ptm  LABELS=<c2c labels>   optional REPO
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
REPO="${REPO:-$S/proteina_d36tc}"
source "$REPO/scratchpad/d36tc_af2.env.sh"
export PYTHONPATH=$REPO
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
: "${MODEL:?}" "${LABELS:?}"
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scratchpad/d36tc_af2rank.py --model "$MODEL" --labels "$LABELS"
rc=$?
echo "AF2RANK_RC=$rc"
exit $rc
