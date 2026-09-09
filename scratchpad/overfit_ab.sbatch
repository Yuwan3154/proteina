#!/bin/bash
#SBATCH --job-name=c2c_ov
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# A/B on the chirality of the loss, ONE structure, from scratch (user directive 2026-09-08):
#   ARM=on   vanilla baseline (smooth_lddt in the diffusion loss)
#   ARM=off  identical, but --no_lddt: the chiral MSE is the only structure loss
# usage:  ARM=off sbatch --job-name=c2c_ov_lddt_off scratchpad/overfit_ab.sbatch
#
# Everything else is the vanilla baseline: 24 blocks, 4 tri blocks, NDIFF=16, lr 6e-4, warmup 1000.
# ACCUM defaults to the baseline 16; the user chose ACCUM=1 for this A/B (2026-09-08) so a single
# structure converges in minutes rather than hours -- applied to BOTH arms. Same --seed in both arms so
# they pin the SAME chain and start from the SAME init. No chain: 48 h on preemptable is enough, and
# a preemption requeues this job id, which resumes from last.ckpt and re-pins from overfit_batch.pt.
set -uo pipefail
ARM="${ARM:?ARM must be on or off}"
case "$ARM" in on) LDDT_FLAG="";; off) LDDT_FLAG="--no_lddt";; *) echo "FATAL: ARM=$ARM"; exit 76;; esac
NAME="c2c_ov_lddt_${ARM}"
ACCUM="${ACCUM:-16}"

REPO="${REPO:-/orcd/scratch/orcd/011/chenxiou/proteina_sh}"
cd "$REPO"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR" /orcd/scratch/orcd/011/chenxiou/c2c_store/logs
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# preemptable hands out busy GPUs; fail in seconds and resubmit rather than after a 3 GB load.
TRY="${TRY:-1}"; MAXTRY=10
if ! python -c "import torch; assert torch.cuda.is_available(); torch.zeros(8).cuda(); print('[gpu]', torch.cuda.get_device_name(0))"; then
    echo "[gpu] UNUSABLE on $(hostname) (attempt $TRY/$MAXTRY)"
    if [ "$TRY" -lt "$MAXTRY" ]; then
        # Resubmit to the SAME partition and card this job was given (CLI overrides included), so a
        # retry on pi_so3 does not silently migrate to preemptable's default gres.
        GRES_NOW=$(scontrol show job "$SLURM_JOB_ID" | grep -oE "TresPerNode=[^ ]+" | cut -d= -f2)
        TRY=$((TRY + 1)) ARM="$ARM" REPO="$REPO" ACCUM="$ACCUM" sbatch --partition="$SLURM_JOB_PARTITION" \
            --gres="${GRES_NOW:-gpu:h200:1}" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            --mem="${SLURM_MEM_PER_NODE}M" --job-name="$SLURM_JOB_NAME" \
            "$REPO/scratchpad/overfit_ab.sbatch"
    else
        echo "FATAL: $MAXTRY nodes all had unusable GPUs -- not transient"
    fi
    exit 79
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git -C "$REPO" log --oneline -1
echo "[arm] $ARM  -> --name $NAME $LDDT_FLAG  (accum $ACCUM)"

torchrun --standalone --nproc_per_node=1 \
    scratchpad/train_c2c.py --devices 1 --accum "$ACCUM" --n_diff 16 --lr 0.0006 \
    --precision bf16-mixed --val_every 500 --warmup 1000 \
    --overfit --seed 0 --name "$NAME" $LDDT_FLAG
rc=$?
echo "C2C_OV_EXIT=$rc"
exit $rc
