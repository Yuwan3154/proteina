#!/bin/bash
# RELAUNCH c2c_cb8_sd10 -- the SIGMA_DATA correction experiment (user directive 2026-09-17).
#
# WHAT THIS RUN IS: EDM/Karras preconditioning takes a sigma_data that should equal the
# PER-COORDINATE std of the data. The model ships 16.0; the measured CB-8 value is 10.31 A
# (radius of gyration 17.85 / sqrt(3)). A wrong sigma_data mis-scales c_skip/c_out/c_in and the loss
# weight w(sigma) = 1/c_out^2, leaving training ~2.4x too weak in exactly the high-noise regime where
# global structure -- and therefore handedness -- is decided. This run changes ONLY that value,
# branched from c2c_cb8_tbeta step 5500.
#
# ⛔⛔ ITS CHAIN DIED on 2026-09-17 when the scratch quota filled (my fault: I added 6.4 GB to a
# filesystem at 99%). No training was lost -- last.ckpt survived at global_step 610.
#
# ⛔⛔ THIS IS A RESUME, NOT A FRESH START. train_c2c.py prefers last.ckpt over --init_from, so the
# run continues from 610 with its optimizer and scheduler intact. INIT_FROM is still passed so that
# if last.ckpt were ever missing the run would warm-start from the ORIGINAL branch point rather than
# train from scratch into the same directory.
# ⛔ Hyperparameters are NOT restored from the checkpoint -- the model is built from CLI args BEFORE
# loading. Every value below is copied from this run's OWN log echo (job 22882837), not guessed:
#   [edm] SIGMA_DATA=10.31
#   [batch] per-rank 2 x accum 4 x 1 ranks = effective 8
#   [model] n_diffusion_samples=48, lr=0.0003, warmup=2000, t_beta=(1.3, 2.0), diff_chunk=8,
#           smooth_lddt=False, w_chiral=0.0
#   [ckpt] interval 200 -> 10          => VAL_EVERY=200
#   n_dump=2 (default) -- verified from the dumped filenames val00_00/val00_01/val01_00/val01_01,
#           i.e. 2 BATCHES x batch_size 2, since n_dump gates on batch_idx not structure count.
# ⚠️ batch_size 2 travels in EXTRA: train_c2c.sbatch does NOT pass --batch_size itself.
set -uo pipefail
NAME=c2c_cb8_sd10
S=/orcd/scratch/orcd/011/chenxiou/c2c_store
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
L="$REPO/scratchpad/train_c2c.sbatch"
BP="$S/branch_points/c2c_cb8_tbeta_step5500.ckpt"
CK="$S/$NAME/last.ckpt"

bash -n "$L" || exit 1
git -C "$REPO" log --oneline -1
# ⛔ INVERTED guard vs the fresh-start launchers: here last.ckpt MUST exist, because the whole point
# is to continue from 610. Its absence would mean silently restarting the experiment.
[ -e "$CK" ] || { echo "FATAL: $CK missing -- this launcher RESUMES; it must not start fresh"; exit 2; }
[ -e "$BP" ] || { echo "FATAL: branch point $BP missing"; exit 3; }
[ -e "$S/.run.lock.$NAME" ] && { echo "FATAL: stale lock $S/.run.lock.$NAME"; exit 4; }
squeue -h -u chenxiou -n "$NAME" -o "%i %T" | grep -q . && { echo "FATAL: $NAME already queued"; exit 5; }

ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
TORCH_CUDA_ARCH_LIST=9.0 "$ENV_DIR/.venv/bin/python" - "$CK" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"[resume-from] global_step={ck.get('global_step')} epoch={ck.get('epoch')} ema={'ema' in ck}")
assert ck.get("global_step", 0) >= 600, f"expected step >=600, got {ck.get('global_step')}"
PY
[ $? -eq 0 ] || { echo "FATAL: checkpoint unreadable or at an unexpected step"; exit 6; }

echo "relaunching $NAME with SIGMA_DATA=10.31"
env CHAIN=38 DEVICES=1 ACCUM=4 NDIFF=48 LR=0.0003 PRECISION=bf16-mixed VAL_EVERY=200 WARMUP=2000 \
    REPO="$REPO" GRES=gpu:h200:1 INIT_FROM="$BP" SIGMA_DATA=10.31 \
    EXTRA="--name $NAME --no_lddt --batch_size 2 --diff_chunk 8 --t_beta 1.3,2.0 --dataset pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828" \
    sbatch --parsable -J "$NAME" -p mit_preemptable --gres=gpu:h200:1 --time=6:00:00 --cpus-per-task=8 --mem=160G "$L"
squeue -h -u chenxiou -n "$NAME" -o "%i %j %T %P %b %l %R"
