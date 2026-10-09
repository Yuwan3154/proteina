#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc CONTROL C (GPU, L40S): rescore the D1 manifest's 130 shipped decoys (cell O6w) on Engaging and gate against
# o6w.csv (any 0.7/0.5 bin flip of tm_ref_pred = FAIL). Inputs staged beforehand under $P/controlC/ (manifest.csv,
# o6w.csv, sc/ = rsync copy of the manifest's /home/gridsan/cou/ files). Partition/gres/time/name on sbatch.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
C=$P/controlC
REPO="${REPO:-$S/proteina_d36tc}"
source "$REPO/scratchpad/d36tc_af2.env.sh"
export PYTHONPATH=$REPO
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum "$AF2RANK_PARAMS_DIR/params_model_1_ptm.npz"
python scratchpad/d36tc_controlC.py --mode score --manifest "$C/manifest.csv" --root "$C/sc" --out "$C/o6w_engaging.csv"
rc=$?
echo "SCORE_RC=$rc"
python scratchpad/d36tc_controlC.py --mode compare --manifest "$C/manifest.csv" --root "$C/sc" --out "$C/o6w_engaging.csv" \
  --reference "$C/o6w.csv"
grc=$?
echo "CONTROL_C_RC=$grc"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$grc"
