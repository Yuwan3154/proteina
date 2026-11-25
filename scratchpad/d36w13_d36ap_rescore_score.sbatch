#!/bin/bash
#SBATCH --job-name=d36w13_d2score
#SBATCH -p xeon-p8
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 4-00:00:00
#SBATCH -o /home/gridsan/cou/logs/d36w13_d2score-%j.out
# W13-B: score the fixed-OpenFold D2 predictions with the ORIGINAL d2_score.py (md5-pinned), same args as
# d2_score.sbatch / d2_score_unm.sbatch; only pred_root and out_prefix point at rescore_w13. CPU only: no openfold import.
# STAGE=shake: phase 0, cells sc,unm. STAGE=d2: d2_ctrl (worklist, phase<=1, sc,both) + d2_unmctrl (worklist_c01, unm).
set -o pipefail
source /etc/profile
module load conda/Python-ML-2025b-pytorch
conda deactivate 2>/dev/null
source activate cathfold
STAGE=${STAGE:?}
EXPECT_OF_SHA=${EXPECT_OF_SHA:?}
D2=/home/gridsan/cou/d36/d36ap/d2
R=/home/gridsan/cou/d36/d36ap/rescore_w13/$STAGE
U=/home/gridsan/cou/.local/bin/USalign
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID STAGE=$STAGE LAUNCH_SHA=${LAUNCH_SHA:-unset} START=$(date)"
[ "$(git -C /home/gridsan/cou/openfold rev-parse HEAD)" = "$EXPECT_OF_SHA" ] || { echo "GATE FAIL openfold HEAD"; exit 3; }
echo "9e4072a743e2bdd0fc53ae638eab41d9  $D2/d2_score.py
5888c852a7f45da82f2da2cc9aba7493  $D2/worklist.csv
50e12eeeb1cf4923069e99e785387383  $D2/worklist_c01.csv" | md5sum -c - || { echo "GATE FAIL md5"; exit 3; }
cd $D2
case $STAGE in
  shake)
    mkdir -p $R/d2/scores
    python -u $D2/d2_score.py --worklist $D2/worklist.csv --pred_root $R/d2/pred --conf_dir $D2/conformers \
      --usalign $U --max_phase 0 --out_prefix $R/d2/scores/d2_shake --workers 8 --cells sc,unm
    rc=$?
    ;;
  d2)
    mkdir -p $R/scores
    python -u $D2/d2_score.py --worklist $D2/worklist.csv --pred_root $R/pred --conf_dir $D2/conformers \
      --usalign $U --max_phase 1 --out_prefix $R/scores/d2_ctrl --workers 8
    rc=$?
    if [ $rc -eq 0 ]; then
      python -u $D2/d2_score.py --worklist $D2/worklist_c01.csv --pred_root $R/pred_unmasked --conf_dir $D2/conformers \
        --usalign $U --max_phase 1 --out_prefix $R/scores/d2_unmctrl --workers 8 --cells unm
      rc=$?
    fi
    ;;
  *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
echo "D2_SCORE_EXIT=$rc END=$(date)"
exit $rc
