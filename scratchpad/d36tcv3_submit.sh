#!/bin/bash
# D36tc-v3 submitter (W10 arm tc, 25 new non-NMR targets; run on the Engaging LOGIN node, sbatch only). Same scripts,
# checkpoints and settings as the published d36tc arm (d36tc_submit.sh); only the work dir (P), natives, groups and the
# max509 dataset pair for 7ZU9_A differ. Template arm NOT here (S12a parked). STAGE picks what to submit:
#   s1        build (CPU) -> F2C (GPU) -> index + Control A2/A3 (CPU)
#   controls  Control C (130 published decoys, cuEq, vs o6w.csv) + plain-kernel control (d36tc_A_tmpl 8TVL_A/6UCK_A
#             x m1/m2 plain vs W4b control (iii), d36w8_tc_ctrl3 rule)
#   gen       GROUP=S|M|L: tri self (8/stem) -> c2c self; c2c native-map (8 seeds)       [DEP = afterok job list]
#   af2rank   KERNELS=cueq|plain LABELS=<comma list> POOLS=<n>: model_1_ptm + model_2_ptm  [DEP]
#   score     KERNELS=cueq|plain SIZES=S=<n>,M=<n>,L=<n>                                    [DEP]
# Seed pairing as published: arms of one group share one chain list (tri group<G>_x8.txt; c2c group<G>.txt).
set -euo pipefail
S=/orcd/scratch/orcd/011/chenxiou
R=/orcd/pool/006/chenxiou/d36_val43
T=$R/arms/tc
I=$T/inputs
REPO=$T/repo
RS=$REPO/scratchpad
TRI=$S/stageA/ckpt/cfft_tri_last-EMA_step60505.ckpt;      TRI_MD5=f8328020bde6267c632d5c8345f8eca2
C2C=$S/c2c_store/c2c_confind_tbeta/final_twin_s29000.ckpt; C2C_MD5=1f5693e773b05ac643a33b9a25ebec9f
EXCL=node4400,node5101,node5204,node4800,node4801,node4802,node4803,node3803,node1381
LOG="--output=$T/logs/%x-%j.out"
# at most 2 GPUs at once: one singleton chain for F2C/tri/c2c (mit_normal_gpu only, as published: tri/c2c refuse an
# existing output on requeue) and one for Control C / AF2Rank (resumable, so preemptable + requeue is safe)
GEN="--job-name=w10_tc_gen --partition=mit_normal_gpu --gres=gpu:l40s:1 --time=06:00:00 --exclude=$EXCL --no-requeue $LOG"
AF2="--job-name=w10_tc_af2 --partition=mit_normal_gpu,mit_preemptable --gres=gpu:l40s:1 --time=06:00:00 --exclude=$EXCL --requeue $LOG"
CPU="--partition=mit_normal --exclude=$EXCL $LOG"
DEPA=""; [ -n "${DEP:-}" ] && DEPA=",afterok:${DEP}"
STAGE="${STAGE:?s1|controls|gen|af2rank|score}"
for f in d36tc_build d36tc_f2c d36tc_index d36tc_tri d36tc_c2c d36tc_controlC d36tc_af2rank d36tc_score d36tcv3_ctrl3; do
  bash -n "$RS/$f.sbatch"
done
git -C "$REPO" log -1 --format="[repo] $REPO HEAD %h"
export P=$T REPO
mkdir -p "$T/logs"

case "$STAGE" in
  s1)
    B=$(env DS=d36tcv3_tri_max384 NATIVES=$R/natives N_TARGETS=25 N_HITS=0 N_EXTRA=3 \
        sbatch --parsable --export=ALL $CPU --job-name=w10_tc_build "$RS/d36tc_build.sbatch")
    F=$(env N_TARGETS=25 N_HITS=0 N_EXTRA=3 sbatch --parsable --export=ALL $GEN --dependency=afterok:$B,singleton \
        "$RS/d36tc_f2c.sbatch")
    X=$(env N_INDEX=25 sbatch --parsable --export=ALL $CPU --job-name=w10_tc_index --dependency=afterok:$F \
        "$RS/d36tc_index.sbatch")
    echo "s1 build $B f2c $F index $X" ;;
  controls)
    # Control C resumes from an existing o6w_engaging.csv: require a fresh real dir (inputs symlinked inside it)
    [ -d "$T/controlC" ] && [ ! -L "$T/controlC" ] && [ ! -e "$T/controlC/o6w_engaging.csv" ] || { echo "FATAL: controlC must be a fresh real dir"; exit 3; }
    C=$(sbatch --parsable --export=ALL $AF2 --dependency=singleton "$RS/d36tc_controlC.sbatch")
    J=""
    for M in model_1_ptm model_2_ptm; do
      j=$(env MODEL=$M LABELS=d36tc_A_tmpl TC_STEMS=8TVL_A,6UCK_A EXPECT_POOLS=2 TC_KERNELS=plain TC_OUT=$T/ctrl_plain \
          sbatch --parsable --export=ALL $AF2 --dependency=singleton "$RS/d36tc_af2rank.sbatch")
      J="$J:$j"; echo "ctrl_plain $M $j"
    done
    K=$(env NEW=$T/ctrl_plain sbatch --parsable --export=ALL $CPU --job-name=w10_tc_ctrl3 --time=01:00:00 \
        --dependency=afterok$J "$RS/d36tcv3_ctrl3.sbatch")
    echo "controlC $C ctrl3 $K" ;;
  gen)
    G="${GROUP:?S|M|L}"
    case "$G" in S) TD=d36tcv3_tri_max384; CD=d36tcv3_c2c_max384;; M) TD=d36tcv3_tri_max493; CD=d36tcv3_c2c_max493;;
                 L) TD=d36tcv3_tri_max509; CD=d36tcv3_c2c_max509;; *) echo "bad GROUP $G"; exit 2;; esac
    L=d36tcv3_${G}_self
    T1=$(env CKPT=$TRI EXPECT_MD5=$TRI_MD5 CHAINS=$I/group${G}_x8.txt N=8 LABEL=$L DATASET=$TD \
         sbatch --parsable --export=ALL $GEN --dependency=singleton$DEPA "$RS/d36tc_tri.sbatch")
    C1=$(env CKPT=$C2C EXPECT_MD5=$C2C_MD5 CHAINS=$I/group${G}.txt MAPS_DIR=$T/tri/$L LABEL=$L DATASET=$CD \
         sbatch --parsable --export=ALL $GEN --dependency=afterok:$T1,singleton "$RS/d36tc_c2c.sbatch")
    N1=$(env CKPT=$C2C EXPECT_MD5=$C2C_MD5 CHAINS=$I/group${G}.txt NATIVE=1 N_SEEDS=8 LABEL=d36tcv3_${G}_native \
         DATASET=$CD sbatch --parsable --export=ALL $GEN --dependency=singleton$DEPA "$RS/d36tc_c2c.sbatch")
    echo "$G tri $T1 c2c $C1 native $N1" ;;
  af2rank)
    K="${KERNELS:?cueq|plain}"
    for M in model_1_ptm model_2_ptm; do
      echo "af2rank $K $M $(env MODEL=$M LABELS="${LABELS:?}" EXPECT_POOLS="${POOLS:?}" TC_KERNELS=$K TC_OUT=$T/af2rank_$K \
          TC_C2C_PDB=$T/c2c_pdb TC_NATIVES=$R/natives TC_SCORE_CHAINS=$I/score_chains.json \
          sbatch --parsable --export=ALL $AF2 --dependency=singleton$DEPA "$RS/d36tc_af2rank.sbatch")"
    done ;;
  score)
    K="${KERNELS:?cueq|plain}"
    echo "score $K $(env TC_OUTCSV=d36tcv3_per_target_$K.csv TC_PREFIX=d36tcv3 TC_GROUPS_DIR=$I TC_C2C_PDB=$T/c2c_pdb \
        TC_AF2RANK=$T/af2rank_$K TC_NATIVES=$R/natives TC_GROUPS="S=self,native;M=self,native;L=self,native" \
        TC_SIZES="${SIZES:?}" TC_SCORE_CHAINS=$I/score_chains.json sbatch --parsable --export=ALL $CPU --job-name=w10_tc_score ${DEP:+--dependency=afterok:$DEP} \
        "$RS/d36tc_score.sbatch")" ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
