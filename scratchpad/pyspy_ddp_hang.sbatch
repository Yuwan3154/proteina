#!/bin/bash
#SBATCH --job-name=pyspy_hang
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# FlightRecorder proved the divergence (seq 172: rank0 all_reduce_barrier[1] vs rank1
# all_reduce[21390025]) but not WHICH Python frame issues the barrier -- and `rg barrier\(`
# finds NOTHING in our code, so it comes from Lightning internals. py-spy on the live hung
# ranks gives the exact stack. The watcher MUST live inside this job: py-spy needs the same
# node and PID namespace as the training processes.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
OUT=$REPO/logs/pyspy_${SLURM_JOB_ID}
mkdir -p "$OUT"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data

module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export SKIP_CONFIND_PRECOMPUTE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
cd "$REPO"

# ── the watcher (directive 3: one-time approved on-box poller) ───────────────────────────────
# Dumps every python train.py process once the run has been in the first step long enough to
# be hung, then keeps a heartbeat file so the state is readable from login without SSH races.
(
  STATUS="$OUT/STATUS.txt"
  for i in $(seq 1 110); do
    sleep 30
    PIDS=$(pgrep -f "[t]rain.py" | tr '\n' ' ')
    echo "[$(date +%H:%M:%S)] elapsed=$((i*30))s pids=${PIDS:-none}" >> "$STATUS"
    # 300 s into the first step is well past healthy (the 1-GPU control did 200 batches in 4 min)
    if [ "$i" -eq 10 ] || [ "$i" -eq 20 ]; then
      for p in $PIDS; do
        echo "=== py-spy dump pid $p at $((i*30))s ===" >> "$OUT/pyspy_dump.txt"
        py-spy dump --pid "$p" --locals >> "$OUT/pyspy_dump.txt" 2>&1
      done
      echo "[$(date +%H:%M:%S)] py-spy dumped at $((i*30))s" >> "$STATUS"
    fi
  done
) &
WATCHER=$!

torchrun --standalone --nproc_per_node=2 proteinfoundation/train.py \
    --config_name training_contact_hier_topology_full512_v1 \
    --ngpus_per_node 2 --nnodes 1 run_name_=pyspy_hang_diag \
    opt.dist_timeout_s=900 || echo "training exited rc=$?"

kill $WATCHER 2>/dev/null || true
echo "=== dumps written to $OUT ==="
ls -l "$OUT"
