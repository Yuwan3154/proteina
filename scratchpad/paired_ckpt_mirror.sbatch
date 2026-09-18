#!/bin/bash
#SBATCH --job-name=pairmirror
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Paired fixed-chain mirror assay across four checkpoints. See paired_ckpt_mirror.py for the design.
#
# ⛔ mit_normal_gpu, NOT mit_preemptable: the preemptable per-user cap is gres/gpu=4 and is already
# full (tri holds 2, a genmaps array task 1, one c2c arm 1). Adding this there would preempt one of
# the very training arms it is meant to measure. mit_normal_gpu has its own gres/gpu=2 cap of which
# I hold 0, and had an idle h200 node at submit time.
# ⛔ h200 NOT l40s: this is the card the model trains on, so memory adequacy is verified rather than
# guessed. No L40S substitution to chase a faster start.
# ⛔ --time=00:30:00 is grounded, not guessed: the equivalent 16-chain sweeps 22957243 and 22962642
# measured 3m42s and 2m51s at 6.4x MORE rollout work than this job, and probe 22867000 used this
# same 30 min limit. The python writes JSONL after every checkpoint, so even a tight limit cannot
# cost more than the last checkpoint.
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name --format=csv,noheader

STORE=/orcd/scratch/orcd/011/chenxiou/c2c_store

# Cheap prechecks BEFORE any copy. Job 22851194 copied 3.2 GB and only then died on a bad config dir.
python -c "import os,sys; d=os.path.join('$REPO','configs','datasets_config','pdb'); sys.exit(0 if os.path.isdir(d) else 1)" \
    || { echo "FATAL: hydra config dir missing under $REPO"; exit 78; }
python -c "import hydra, torch, proteinfoundation" \
    || { echo "FATAL: imports unavailable in this env"; exit 79; }

ANC="$STORE/branch_points/c2c_cb8_tbeta_fape.ckpt"
RES="$STORE/c2c_cb8_tbeta/epoch=5-step=9750.ckpt"
CTRL="$STORE/c2c_cb8_tbeta/epoch=5-step=8500.ckpt"
ARM_SRC="$STORE/c2c_cb8_fapelo/last.ckpt"
ARM="$TMPDIR/pairmirror-arm-$SLURM_JOB_ID.ckpt"

# ⭐ CKPTS overrides the default list (';'-separated label=path -- NEVER commas, --export splits on
# those). An override is assumed to name only STATIC checkpoints, so it needs no snapshot and
# touches no disk at all; that is why the disk gate below runs only on the default path.
if [ -n "${CKPTS:-}" ]; then
    SPEC="$CKPTS"
    echo "[ckpts] override supplied; no snapshot needed, zero disk written"
else
    # ⛔ DISK GATE, default path only. Recorded facts: scratch quota is 1 TB, and on 2026-09-17 a
    # write at 1022 G killed three running jobs. A 3.2 GB checkpoint needs 2x its size free.
    USED_GB=$(du -sm /orcd/scratch/orcd/011/chenxiou 2>/dev/null | awk '{printf "%d", $1/1024}')
    FREE_GB=$((1024 - USED_GB))
    echo "[disk] scratch used ${USED_GB} G of 1024 G quota; ${FREE_GB} G free"
    if [ "$FREE_GB" -lt 7 ]; then
        echo "FATAL: only ${FREE_GB} G free; a 3.2 GB copy needs 7 G (2x + margin). Refusing."
        exit 76
    fi
    # ⛔ Only the ARM is copied. c2c_cb8_tbeta is CANCELLED so its directory is static, and the
    # branch point is frozen by design -- neither can be read mid-write. c2c_cb8_fapelo is queued
    # and WILL rewrite last.ckpt the moment it gets a GPU, so that one must be snapshotted.
    cp "$ARM_SRC" "$ARM" || { echo "FATAL: cannot copy $ARM_SRC"; exit 77; }
    echo "[copy] arm snapshot taken from $ARM_SRC"
    SPEC="anc8076=$ANC;resolved9750=$RES;fapelo_arm=$ARM;ctrl8500=$CTRL"
fi

# ⛔ Name the JSONL after the JOB NAME, matching what %x puts in --output. With a hardcoded prefix a
# `-J pairdose` run wrote its log to pairdose-*.out but its data to pairmirror-*.jsonl, and I went
# looking for a completed job's artefact in the wrong file. Same trap as %x, opposite direction.
python scratchpad/paired_ckpt_mirror.py \
    --ckpts "$SPEC" \
    --steps "${STEPS:-20}" --n_chains "${NCHAINS:-16}" \
    --out "$STORE/logs/${SLURM_JOB_NAME:-pairmirror}-$SLURM_JOB_ID.jsonl"
echo "PAIRMIRROR_EXIT=$?"
rm -f "$ARM"
echo "[cleanup] done"
