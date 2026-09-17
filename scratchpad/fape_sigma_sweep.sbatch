#!/bin/bash
#SBATCH --job-name=c2c_fapesig
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Second queue position for the one-step probe. The mit_normal_gpu copy (22845480) has sat at
# Reason=Priority since 14:23 despite free cards -- past backfill depth. This is the same 30 min
# inference payload on a partition where I hold only 1 of 4 GPU slots.
#
# ⛔ Deliberately invokes the SAME absolute-path script as 22845480 rather than a repo copy, so the
# two queue positions cannot diverge into running different code.
# ⛔ h200 reused because it is the card this model is trained on, so memory adequacy is verified
# rather than guessed. L40S was NOT substituted to chase a faster start.
# ⛔ Whichever copy starts first, cancel the other by JobName -- do not leave both to run.
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

# ⛔ Copy the checkpoint first -- c2c_cb8_tbeta is LIVE and rewrites its checkpoint directory at every
# validation; reading one mid-write gives a truncated file.
# ⛔ Cheap prechecks BEFORE the 3.2 GB copy. Job 22851194 copied the checkpoint and only then died on
# a missing hydra config dir, burning the allocation for nothing.
python -c "import os,sys; d=os.path.join('$REPO','configs','datasets_config','pdb'); sys.exit(0 if os.path.isdir(d) else 1)" \
    || { echo "FATAL: hydra config dir missing under $REPO"; exit 78; }
python -c "import hydra, torch, proteinfoundation" \
    || { echo "FATAL: imports unavailable in this env"; exit 79; }

SRC="${CKPT:-/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_cb8_tbeta/last.ckpt}"
CK="$TMPDIR/onestep-$SLURM_JOB_ID.ckpt"
cp "$SRC" "$CK" || { echo "FATAL: cannot copy $SRC"; exit 77; }
echo "checkpoint copied from $SRC"

python scratchpad/fape_sigma_sweep.py --ckpt "$CK" --n_batches 6
echo "FAPESIG_EXIT=$?"
rm -f "$CK"
