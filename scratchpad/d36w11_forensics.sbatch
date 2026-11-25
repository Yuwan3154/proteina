#!/bin/bash
#SBATCH --job-name=d36w11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/orcd/pool/006/chenxiou/d36_w11_forensics/logs/%x-%j.out
# D36 W11 AF2Rank kernel forensics (scratchpad/d36w11_forensics.py). Partition/time/exclude on the sbatch command line.
#   MODE=shake : e1 run1, e2 on 6UCK_A only, e3 at N 67/101 and stack N 67 (anchor control: plain/cuEq must reproduce W4-diag)
#   MODE=full  : e1 x3 processes + e1cmp, e2 (6UCK_A full hooks, 8AUC_B light), e3 (all sizes, captured inputs)
#   MODE=e2b   : e2 on 8AUC_B only without plain_nochunk (the unchunked L=493 forward OOMs on a 46 GB L40S; job 30401110)
set -uo pipefail
W=/orcd/pool/006/chenxiou/d36_w11_forensics
REPO=$W/repo
: "${MODE:?}"
cd "$REPO" || exit 3; echo "[repo] $REPO HEAD $(git log -1 --format=%h) tracked changes $(git status --porcelain --untracked-files=no | wc -l)"
source "$REPO/scratchpad/d36tc_af2.env.sh"
export PYTHONPATH=$REPO
export TMPDIR=/tmp/chenxiou_w11_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export TRITON_CACHE_DIR=$TMPDIR/triton
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for pm in "params_model_1_ptm.npz dfca7cd972028fc6adcc0c8995857a2a" "params_model_2_ptm.npz edb0729d11e83734b820c348c5ffe42a"; do
  set -- $pm; h=$(md5sum < "$AF2RANK_PARAMS_DIR/$1" | cut -d' ' -f1); echo "[params] $1 $h (SC $2)"
  [ "$h" = "$2" ] || { echo "FATAL: $1 differs from the SuperCloud production params"; exit 72; }
done
OUT=$W/out_${MODE}_${SLURM_JOB_ID}; mkdir -p "$OUT"
P=scratchpad.d36w11_forensics  # -m from REPO: the script dir holds scratchpad/openfold/ which shadows openfold.utils
rc=0
run() { "$@"; local r=$?; echo "[rc] $r : $*"; [ $r -eq 0 ] || rc=$r; }
cd "$REPO" || exit 3
if [ "$MODE" = shake ]; then
  run python -m "$P" --out "$OUT" --mode e1 --tag run1
  run python -m "$P" --out "$OUT" --mode e2 --e2-pids 6UCK_A
  run python -m "$P" --out "$OUT" --mode e3 --sizes 67 101 --stack-sizes 67
elif [ "$MODE" = full ]; then
  for t in run1 run2 run3; do run python -m "$P" --out "$OUT" --mode e1 --tag $t; done
  run python -m "$P" --out "$OUT" --mode e1cmp
  run python -m "$P" --out "$OUT" --mode e2
  run python -m "$P" --out "$OUT" --mode e3
elif [ "$MODE" = e2b ]; then
  run python -m "$P" --out "$OUT" --mode e2 --e2-pids 8AUC_B --skip-arms plain_nochunk
else
  echo "unknown MODE=$MODE"; rc=2
fi
ls "$OUT" | wc -l
echo "W11_RC=$rc"
exit $rc
