#!/bin/bash
#SBATCH --job-name=c2c_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# How large can the per-rank batch go? MEASURE it -- do not pick it.
#
# ⭐ Why this is the right knob. c2c measured 99-100% GPU util but only 35.4 GB of 97.9 GB (36%) at
# per-rank 1 x accum 8, running 1.44 optim steps/min: compute-bound with VRAM to spare.
# ⛔ train_c2c.py:93 already states the rule -- "Effective batch = batch_size * accum * devices;
# keep it fixed when raising batch_size." So we raise batch_size and DIVIDE accum, holding effective
# batch at 8. Training dynamics are then unchanged and the fine-tune's only deliberate difference
# from its parent run stays SIGMA_DATA.
# ⛔ VRAM does NOT scale linearly with batch (model + optimizer state are fixed overhead), which is
# exactly why this is measured rather than extrapolated from 35.4 GB.
#
# Submit once per card type you intend to allow, e.g.
#   sbatch -p pi_so3        --gres=gpu:rtx_pro_6000:1 scratchpad/c2c_batch_probe.sbatch
#   sbatch -p mit_normal_gpu --gres=gpu:h200:1        scratchpad/c2c_batch_probe.sbatch
set -uo pipefail
REPO=/orcd/scratch/orcd/011/chenxiou/proteina_tri
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
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
TOTMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "CARD=$CARD  TOTAL_MEM_MiB=$TOTMEM"

# ⛔ --smoke caps max_steps at 16, so each trial is short. ⛔ A DIFFERENT --name per trial, and never
# the real run's name: train_c2c.py resumes from <store>/<name>/last.ckpt if it exists, so probing
# under the live name would resume the real run and write into its checkpoint directory.
for BS in 1 2 4; do
    ACC=$((8 / BS))            # hold effective batch at 8
    echo ""
    echo "################ per-rank batch=$BS  accum=$ACC  (effective 8) ################"
    # Sample GPU memory while the trial runs; report the PEAK. Avoids patching train_c2c.py.
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 3; done ) > "$TMPDIR/mem_$SLURM_JOB_ID.$BS" 2>/dev/null &
    SAMPLER=$!
    # ⛔⛔ MIRROR THE PARENT RUN EXACTLY, varying ONLY batch_size/accum. A first version of this probe
    # omitted --diff_chunk and --no_lddt and so ran diff_chunk=0 + smooth_lddt=True: all 48 diffusion
    # samples resident at once plus an L^2 pair-grid sigmoid. It OOM'd at 94.58 GiB on a 97.9 GB card
    # at batch 1 -- a configuration the run does not use, measuring a number that means nothing.
    # Parent settings, from c2c_cb8_tbeta-22756317.out's own [model] line:
    #   n_diffusion_samples=48, diff_chunk=8, smooth_lddt=False, t_beta=(1.3,2.0), lr=3e-4, warmup=2000
    timeout 900 python scratchpad/train_c2c.py --smoke \
        --devices 1 --batch_size "$BS" --accum "$ACC" --n_diff 48 --diff_chunk 8 \
        --no_lddt --t_beta 1.3,2.0 --lr 0.0003 --warmup 2000 \
        --name "probe_bs${BS}_${SLURM_JOB_ID}" \
        --dataset pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828
    RC=$?
    kill "$SAMPLER" 2>/dev/null
    PEAK=$(sort -n "$TMPDIR/mem_$SLURM_JOB_ID.$BS" 2>/dev/null | tail -1)
    rm -f "$TMPDIR/mem_$SLURM_JOB_ID.$BS"
    # ⛔ Report the exit code explicitly: an OOM is the ANSWER here, not a failure of the probe.
    echo "PROBE_RESULT bs=$BS accum=$ACC rc=$RC peak_mib=${PEAK:-unknown} total_mib=$TOTMEM card=$CARD"
done
echo "C2C_PROBE_DONE"
