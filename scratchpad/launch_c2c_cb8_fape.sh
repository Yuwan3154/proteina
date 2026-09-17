#!/bin/bash
# c2c CB-8 + backbone FAPE FINE-TUNE (user directive 2026-09-17).
#
# Clone of launch_c2c_cb8_tbeta.sh with EXACTLY ONE SCIENTIFIC change: --w_fape. Everything that
# affects the model -- CB-8 dataset, --no_lddt, NDIFF 48, diff_chunk 8, lr 3e-4, warmup 2000,
# n_dump 16, t_beta 1.3,2.0, effective batch 8 structures x 48 samples -- is reproduced from
# c2c_cb8_tbeta, so the comparison against the parent has one variable.
#
# WHY: AF2 trains with FAPE and does not exhibit this model's mirror failure. The earlier
# "FAPE carries 290x less signal than the MSE" argument was WRONG twice over -- it compared raw
# magnitudes of differently-scaled losses (which is what a weight is for), and it built frames from
# predicted atoms whereas AF2's IPA emits frames directly. Both retracted.
#
# ⛔⛔ WARM START, NOT RESUME. INIT_FROM is a FROZEN copy of tbeta's last.ckpt at global_step 8076
# (job 22876715 verified size + md5 + torch-loadable + has_ema). tbeta is LIVE and rewrites its own
# last.ckpt, so pointing at it directly risks a torn read. train_c2c.py ignores --init_from once
# this run's OWN last.ckpt exists, so chained successors resume normally.
#
# ⭐ MEASURED SETTINGS, not guessed:
#  - fape_sigma_max defaults to SIGMA_DATA (16.0). Job 22876261 measured FAPE's mirror gap at
#    +0.5389 (sigma<1), +0.2974 (1-4), +0.1132 (4-16), +0.0396 (16-64), +0.0001 (>256). t_beta puts
#    ~75% of samples above sigma 256, where the gap is ZERO -- ungated, FAPE would be mostly
#    saturation noise. The gate keeps the samples that carry handedness information.
#  - clamp stays at AF2's published 10 A. Tightening it DESTROYS the gap rather than sharpening it:
#    peak gap 0.539 -> 0.206 -> 0.040 -> 0.009 for clamp 10 -> 5 -> 2 -> 1 A (same job).
#  - w_fape 1.4: job 22875900 measured the diffusion term's contribution at ALPHA_DIFFUSION x 0.8838
#    = 3.5353, and FAPE inside the gate averages ~0.615, so 1.4 puts FAPE at ~24% of the diffusion
#    term. ⛔ This is the ONE number here that is a CHOICE, not a measurement -- the user's
#    "tune the loss magnitude" instruction. It is recorded with the run and trivially changed.
#
# ⛔ mit_preemptable, NOT mit_normal_gpu: the two w_chiral arms have held the 2-GPU mit_normal_gpu
# cap at Reason=Priority for 12+ h and are the user's own experiment -- this must not displace them.
# 6 h segments match sd10's proven pattern on this partition; with the 10-step checkpoint cap a lost
# segment now costs ~7.5 min.
#
# ⚠️ Watch `fape_frac` in the logs. It is the fraction of diffusion samples clearing the sigma gate.
# If it collapses toward 0 the term is silently absent and the run would look healthy while testing
# nothing. Expect roughly 0.2 under t_beta(1.3,2.0).
set -uo pipefail
# NAME and W_FAPE are overridable so the LIGHT arm forks from the SAME branch point as the
# heavy one -- both must start from c2c_cb8_tbeta_fape.ckpt or the dose-response is confounded.
NAME="${NAME:-c2c_cb8_fape}"
S=/orcd/scratch/orcd/011/chenxiou/c2c_store
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
L="$REPO/scratchpad/train_c2c.sbatch"
BP="$S/branch_points/c2c_cb8_tbeta_fape.ckpt"
W_FAPE="${W_FAPE:-1.4}"
DIFF_CHUNK="${DIFF_CHUNK:-8}"
FAPE_CHUNK="${FAPE_CHUNK:-8}"
T_BETA="${T_BETA:-1.3,2.0}"
LR="${LR:-0.0003}"
WARMUP="${WARMUP:-2000}"

bash -n "$L" || exit 1
git -C "$REPO" log --oneline -1
[ -e "$BP" ] || { echo "FATAL: branch point $BP missing -- run freeze_branch_point.sbatch first"; exit 2; }
[ -e "$S/$NAME/last.ckpt" ] && { echo "FATAL: $S/$NAME/last.ckpt exists -- would RESUME, not start fresh"; exit 3; }
[ -e "$S/.run.lock.$NAME" ] && { echo "FATAL: stale lock $S/.run.lock.$NAME"; exit 4; }
squeue -h -u chenxiou -n "$NAME" -o "%i %T" | grep -q . && { echo "FATAL: $NAME already queued"; exit 5; }
# ⛔ The FAPE code must actually be present in the checkout this job will run.
grep -q "fape_sigma_max" "$REPO/scratchpad/train_c2c.py" || { echo "FATAL: repo lacks --fape_sigma_max; git pull"; exit 6; }

echo "w_fape=$W_FAPE lr=$LR warmup=$WARMUP diff_chunk=$DIFF_CHUNK fape_chunk=$FAPE_CHUNK t_beta=$T_BETA"
echo "branch point: $BP"
env CHAIN=38 DEVICES=1 ACCUM=8 NDIFF=48 LR="$LR" PRECISION=bf16-mixed VAL_EVERY=500 WARMUP="$WARMUP" REPO="$REPO" \
    GRES=gpu:h200:1 INIT_FROM="$BP" \
    EXTRA="--name $NAME --no_lddt --n_dump 16 --diff_chunk $DIFF_CHUNK --t_beta $T_BETA --w_fape $W_FAPE --fape_chunk $FAPE_CHUNK --dataset pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828" \
    sbatch --parsable -J "$NAME" -p mit_preemptable --gres=gpu:h200:1 --time=6:00:00 --cpus-per-task=8 --mem=160G "$L"
squeue -h -u chenxiou -n "$NAME" -o "%i %j %T %P %b %l %R"
