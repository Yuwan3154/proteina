#!/bin/bash
# FRESH start of the ConFind c2c twin of c2c_cb8_tbeta (Directive B, user 2026-09-22): tbeta's exact
# recipe, ONLY the input contact map changed (ConFind instead of CB-8 A).
# User choices 2026-09-22: "1 H200 x accum 8 now" (mit_normal_gpu, 6 h chained segments; same
# effective batch and same deterministic val chains as tbeta, differs only in card type) and a
# retention ladder matching tbeta's anchors: 8,000 / 8,500 / 9,500 + the frozen tbeta step + every
# 2,000 steps after 9,500.
#
# ⛔⛔ EVERY training value is tbeta's own, from its authoritative log line:
#   "[model] ... n_diffusion_samples=48, lr=0.0003, warmup=2000, t_beta=(1.3, 2.0), diff_chunk=8,
#    smooth_lddt=False, ... p_mirror=0.0" and "[batch] per-rank 1 x accum 8 x 1 ranks = effective 8",
#   plus SIGMA_DATA=16.0, VAL_EVERY=500, n_dump 16, bf16-mixed. The sbatch DEFAULTS differ in three
#   places (NDIFF 8, LR 6e-4, WARMUP 1000) and would silently train a different model.
# ⛔ --t_beta carries a COMMA, so it travels inside EXTRA as an env var, never via --export=ALL,...
# ⛔ -J NAME == --name NAME: the single-writer lock is keyed by the SLURM job name.
# ⛔ No disk `du` here: that is heavy I/O on the login node. Free space is checked by a CPU job.
set -uo pipefail
NAME="${NAME:-c2c_confind_tbeta}"
FROZEN="${FROZEN:?FROZEN = global step of the frozen tbeta reference (branch_points/c2c_cb8_tbeta_frozen_step*.ckpt)}"
S=/orcd/scratch/orcd/011/chenxiou/c2c_store
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
L="$REPO/scratchpad/train_c2c.sbatch"
DATASET=pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828
# 6 h segments; ~18.6k steps at an H200 step rate below tbeta's RTX 94/h needs ~40 of them. Extra
# segments cost nothing: the chain can be stopped at any time.
CHAIN="${CHAIN:-60}"

bash -n "$L" || exit 1
git -C "$REPO" log --oneline -1
[ -e "$REPO/configs/datasets_config/pdb/$DATASET.yaml" ] || { echo "FATAL: $DATASET.yaml not in checkout"; exit 6; }
grep -q "keep_steps" "$REPO/scratchpad/train_c2c.py" || { echo "FATAL: checkout lacks the ladder flags"; exit 7; }
[ -e "$S/branch_points/c2c_cb8_tbeta_frozen_step${FROZEN}.ckpt" ] || { echo "FATAL: frozen tbeta step ${FROZEN} not found"; exit 8; }

# FRESH-start guards: never train into an existing run directory or next to a live lock.
[ -e "$S/$NAME/last.ckpt" ] && { echo "FATAL: $S/$NAME/last.ckpt exists -- this is a fresh-start launcher"; exit 2; }
[ -e "$S/.run.lock.$NAME" ] && { echo "FATAL: lock $S/.run.lock.$NAME exists"; exit 3; }
squeue -h -u chenxiou -n "$NAME" -o "%i %T" | grep -q . && { echo "FATAL: $NAME already queued/running"; exit 4; }

KEEP="8000,8500,9500,${FROZEN}"
echo "[launch] $NAME fresh, dataset=$DATASET, ladder keep_steps=$KEEP keep_every=2000"
env CHAIN="$CHAIN" DEVICES=1 ACCUM=8 NDIFF=48 LR=0.0003 PRECISION=bf16-mixed \
    VAL_EVERY=500 WARMUP=2000 REPO="$REPO" SIGMA_DATA=16.0 \
    GRES=gpu:h200:1 \
    EXTRA="--name $NAME --no_lddt --n_dump 16 --diff_chunk 8 --t_beta 1.3,2.0 --dataset $DATASET --keep_steps $KEEP --keep_every 2000" \
    sbatch --parsable -J "$NAME" -p mit_normal_gpu --gres=gpu:h200:1 --time=06:00:00 \
           --cpus-per-task=8 --mem=100G "$L"
squeue -h -u chenxiou -n "$NAME" -o "%i %j %T %P %b %l %R"
