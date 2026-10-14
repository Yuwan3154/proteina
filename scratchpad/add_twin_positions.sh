#!/bin/bash
# Extra queue positions for the ConFind c2c twin on other card types (user 2026-09-23: any free GPU, incl.
# non-H200). Same env as launch_c2c_confind_tbeta.sh; -J is the lock key, so whichever position starts
# first trains and any later one exits on the lock. Card types chosen by memory: tbeta's measured peak is
# 31.8 GiB (~35 GiB nvidia-smi) at bs 1 x n_diff 48, so L40S 48 GB is the smallest card that fits.
set -uo pipefail
NAME=c2c_confind_tbeta
FROZEN=18620
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
DATASET=pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828
KEEP="8000,8500,9500,${FROZEN}"
sub() {  # $1 = gres type, $2 = partitions
    env CHAIN=60 DEVICES=1 ACCUM=8 NDIFF=48 LR=0.0003 PRECISION=bf16-mixed \
        VAL_EVERY=500 WARMUP=2000 REPO="$REPO" SIGMA_DATA=16.0 GRES="gpu:$1:1" \
        EXTRA="--name $NAME --no_lddt --n_dump 16 --diff_chunk 8 --t_beta 1.3,2.0 --dataset $DATASET --keep_steps $KEEP --keep_every 2000" \
        sbatch --parsable -J "$NAME" -p "$2" --gres="gpu:$1:1" --time=06:00:00 \
               --cpus-per-task=8 --mem=100G "$REPO/scratchpad/train_c2c.sbatch"
}
echo "l40s: $(sub l40s mit_normal_gpu,mit_preemptable)"
echo "rtx_pro_6000: $(sub rtx_pro_6000 pi_so3,mit_preemptable)"
echo "h100: $(sub h100 mit_preemptable)"
squeue -h -u chenxiou -n "$NAME" -o "%i %j %P %b %l %T %R"
