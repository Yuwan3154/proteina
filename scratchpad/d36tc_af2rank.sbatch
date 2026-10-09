#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc AF2Rank (GPU, L40S): d36tc_af2rank.py for MODEL over LABELS (comma list of c2c labels). Resumable per pool, so a
# timed-out job is simply resubmitted. Partition/gres/time/name (d36tc_gpu, singleton) on the sbatch command line.
#   MODEL=model_1_ptm|model_2_ptm  LABELS=<c2c labels>  EXPECT_POOLS=<pool count>   optional REPO
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
REPO="${REPO:-$S/proteina_d36tc}"
source "$REPO/scratchpad/d36tc_af2.env.sh"
export PYTHONPATH=$REPO
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
: "${MODEL:?}" "${LABELS:?}" "${EXPECT_POOLS:?}"
# `python -m` from $REPO: running scratchpad/<x>.py puts scratchpad/ first on sys.path, and the tracked stub
# scratchpad/openfold/ (utils/tm_score.py only) then shadows the real openfold (Control C 25383711: no openfold.utils.loss)
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# AF2 params must be the SuperCloud production files (md5 from SuperCloud ~/openfold/openfold/resources/params)
for pm in "params_model_1_ptm.npz dfca7cd972028fc6adcc0c8995857a2a" "params_model_2_ptm.npz edb0729d11e83734b820c348c5ffe42a"; do
  set -- $pm; h=$(md5sum < "$AF2RANK_PARAMS_DIR/$1" | cut -d' ' -f1); echo "[params] $1 $h (SC $2)"
  [ "$h" = "$2" ] || { echo "FATAL: $1 differs from the SuperCloud production params"; exit 72; }
done
python -m scratchpad.d36tc_af2rank --model "$MODEL" --labels "$LABELS" --expect-pools "$EXPECT_POOLS"
rc=$?
echo "AF2RANK_RC=$rc"
exit $rc
