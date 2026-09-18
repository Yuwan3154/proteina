#!/bin/bash
# Resume c2c_cb8_tbeta from its own last.ckpt (step 10,290) to test whether the mirror resolution
# is STABLE past the point training stopped. User-authorised 2026-09-18.
#
# ⛔ This is a RESUME, not a fresh start, so the guards are the INVERSE of launch_c2c_cb8_fape.sh:
# last.ckpt MUST exist. train_c2c.py auto-resumes from it (and ignores INIT_FROM when it is present),
# so no INIT_FROM is passed.
#
# ⛔⛔ EVERY value below is taken from tbeta's OWN training log, not from the sbatch defaults, which
# differ in three places that would silently change the experiment:
#     NDIFF   sbatch default 8      -> tbeta ran 48
#     LR      sbatch default 6e-4   -> tbeta ran 3e-4
#     WARMUP  sbatch default 1000   -> tbeta ran 2000
# A resume on the defaults would look healthy while training a DIFFERENT model.
# Authoritative line: "[model] 202.76 M parameters, 24 diffusion blocks, n_diffusion_samples=48,
#   lr=0.0003, warmup=2000, t_beta=(1.3, 2.0), diff_chunk=8, smooth_lddt=False, w_chiral=0.0,
#   w_fape=0.0"
# ⛔ --t_beta carries a COMMA, so it travels inside EXTRA as an env var. NEVER put it in
#   `--export=ALL,...`, which splits on commas and would deliver only "1.3".
# ⛔ pi_so3 + rtx_pro_6000:1 + 100G + 8 cpus + 2 days are what the last tbeta job actually held
#   (sacct 22756317). pi_so3 does NOT preempt, which is why the control lives there.
set -uo pipefail
NAME="${NAME:-c2c_cb8_tbeta}"
S=/orcd/scratch/orcd/011/chenxiou/c2c_store
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
L="$REPO/scratchpad/train_c2c.sbatch"
TIME="${TIME:-2-00:00:00}"
CHAIN="${CHAIN:-38}"
VAL_EVERY="${VAL_EVERY:-500}"     # tbeta's own historical cadence -- keeps its series comparable
DEP="${DEP:-}"

bash -n "$L" || exit 1
git -C "$REPO" log --oneline -1

# ⛔ INVERSE guard: refuse if there is NOTHING to resume from, which would silently start a fresh
# run into the control's directory and destroy the very history this run exists to extend.
[ -e "$S/$NAME/last.ckpt" ] || { echo "FATAL: $S/$NAME/last.ckpt missing -- nothing to resume"; exit 2; }
if [ -z "$DEP" ] && [ -e "$S/.run.lock.$NAME" ]; then
    echo "FATAL: stale lock $S/.run.lock.$NAME -- a previous segment did not clean up"; exit 3
fi
if [ -z "$DEP" ] && squeue -h -u chenxiou -n "$NAME" -o "%i %T" | grep -q .; then
    echo "FATAL: $NAME already queued/running -- refusing to double-submit"; exit 4
fi

# ⛔ Disk gate. Recorded: quota 1 TB, and a write at 1022 G killed three running jobs on 09-17.
# A chained training run holds (save_top_k+1) x 3.2 GB and _atomic_save needs 2x the file size.
USED_GB=$(du -sm /orcd/scratch/orcd/011/chenxiou 2>/dev/null | awk '{printf "%d", $1/1024}')
FREE_GB=$((1024 - USED_GB))
echo "[disk] scratch ${USED_GB} G of 1024 G; ${FREE_GB} G free"
[ "$FREE_GB" -ge 20 ] || { echo "FATAL: only ${FREE_GB} G free; a chained run needs >=20 G"; exit 5; }

echo "[resume] $NAME from $S/$NAME/last.ckpt"
ls -la "$S/$NAME/last.ckpt"

env CHAIN="$CHAIN" DEVICES=1 ACCUM=8 NDIFF=48 LR=0.0003 PRECISION=bf16-mixed \
    VAL_EVERY="$VAL_EVERY" WARMUP=2000 REPO="$REPO" SIGMA_DATA=16.0 \
    GRES=gpu:rtx_pro_6000:1 \
    EXTRA="--name $NAME --no_lddt --n_dump 16 --diff_chunk 8 --t_beta 1.3,2.0 --dataset pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828" \
    sbatch --parsable -J "$NAME" -p pi_so3 --gres=gpu:rtx_pro_6000:1 --time="$TIME" \
           ${DEP:+--dependency=afterany:$DEP} --cpus-per-task=8 --mem=100G "$L"
squeue -h -u chenxiou -n "$NAME" -o "%i %j %T %P %b %l %R"
