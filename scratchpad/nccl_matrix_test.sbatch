#!/bin/bash
#SBATCH --job-name=nccl_matrix
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# Controlled A/B, because "it hung, then I added a flag and it ran" is correlation, not cause --
# and in fact NCCL_P2P_DISABLE=1 did NOT save the 2x L40S run (job 20854847 still died with 3
# collective timeouts at the same SeqNum=172). Both ranks reported the SAME last-enqueued work,
# so this is transport, not rank divergence.
# Runs the SAME tiny 1-element allreduce -- the exact op that hangs -- under four configurations
# on one 2x L40S node, each with a short 60 s watchdog so a failure is fast instead of 10 minutes.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
module load miniforge cuda/13.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate cue_openfold
set -u
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
cd "$REPO"

echo "host=$(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== P2P topology as the driver sees it ==="
nvidia-smi topo -m 2>/dev/null | head -8
python - <<'PY'
import torch
n = torch.cuda.device_count()
print("device_count:", n)
for i in range(n):
    for j in range(n):
        if i != j:
            print(f"  can_device_access_peer({i},{j}) = {torch.cuda.can_device_access_peer(i, j)}")
PY

cat > /tmp/allreduce_probe.py <<'PY'
import datetime, os, sys
import torch, torch.distributed as dist

backend = os.environ.get("PROBE_BACKEND", "nccl")
dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
# The op that hangs in training: a single-element allreduce, repeated past SeqNum 172.
t = torch.ones(1, device="cuda")
for i in range(400):
    dist.all_reduce(t)
    if i % 100 == 0 and rank == 0:
        print(f"    allreduce #{i} ok, value={t.item():.0f}", flush=True)
dist.barrier()
if rank == 0:
    print(f"    PASS backend={backend} world={world} 400 allreduces completed", flush=True)
dist.destroy_process_group()
PY

run_case () {
  local label="$1"; shift
  echo "──────── CASE: $label ────────"
  ( set +e
    env "$@" NCCL_DEBUG=WARN TORCH_NCCL_BLOCKING_WAIT=1 \
      timeout 180 torchrun --standalone --nproc_per_node=2 /tmp/allreduce_probe.py 2>&1 \
      | grep -vE "^\[rank[01]\]:\[" | tail -8
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 124 ]; then echo "    RESULT $label = HUNG (timeout)"; \
    elif [ "$rc" -ne 0 ]; then echo "    RESULT $label = FAILED rc=$rc"; \
    else echo "    RESULT $label = OK"; fi )
}

run_case "1-nccl-default"                    PROBE_BACKEND=nccl
run_case "2-nccl-P2P_DISABLE"                PROBE_BACKEND=nccl NCCL_P2P_DISABLE=1
run_case "3-nccl-P2P+SHM_DISABLE"            PROBE_BACKEND=nccl NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
run_case "4-gloo-control"                    PROBE_BACKEND=gloo
