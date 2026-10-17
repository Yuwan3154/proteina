#!/bin/bash
#SBATCH --job-name=t8nccl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/t8_rosetta/nccl-%j.out
#
# 2-rank NCCL A/B on ONE node in the T8 training env (sm120 venv), after hang test 26095542 hung in the FIRST collective
# (DDP-init allgather, SeqNum 1) on node5202. Same probe as nccl_matrix_test.sbatch (cases 1-3 + gloo control), plus the
# first op DDP issues (an allgather) and NCCL_DEBUG=INFO transport lines. Node/partition/gres at submit.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$CC"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID:-nojob}_nccl
LOGS=/orcd/scratch/orcd/011/chenxiou/t8_rosetta/nccl-${SLURM_JOB_ID:-nojob}
mkdir -p "$TMPDIR" "$LOGS"

echo "host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
nvidia-smi topo -m
echo "=== inherited NCCL/torch env (case 1 must be the default) ==="
env | grep -E '^(NCCL_|TORCH_NCCL|CUDA_VISIBLE|GLOO)' || true
python -c "
import torch
n = torch.cuda.device_count()
print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), 'device_count', n)
for i in range(n):
    for j in range(n):
        if i != j:
            print(f'  can_device_access_peer({i},{j}) = {torch.cuda.can_device_access_peer(i, j)}')
"

cat > "$TMPDIR/probe.py" <<'PY'
import datetime
import os

import torch
import torch.distributed as dist

backend = os.environ.get("PROBE_BACKEND", "nccl")
rank = int(os.environ["LOCAL_RANK"])
dev = torch.device("cuda", rank)
torch.cuda.set_device(dev)
# no device_id: lazy communicator creation inside the first collective, as in the run that hung
dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=60))
world = dist.get_world_size()
if rank == 0:
    print("    init ok", flush=True)
out = [torch.zeros(1, device=dev) for _ in range(world)]
dist.all_gather(out, torch.full((1,), float(rank), device=dev))
assert [o.item() for o in out] == list(range(world)), [o.item() for o in out]
if rank == 0:
    print(f"    allgather ok: {[o.item() for o in out]}", flush=True)
t = torch.ones(1, device=dev)
for i in range(400):
    dist.all_reduce(t)
big = torch.ones(64 * 2**20, device=dev)
dist.all_reduce(big)
torch.cuda.synchronize()
assert bool(big.eq(world).all())
dist.barrier()
if rank == 0:
    print(f"    PASS backend={backend} world={world}: allgather + 400 allreduces + 256 MB allreduce (values checked)", flush=True)
dist.destroy_process_group()
PY

run_case () {
  local label="$1"; shift
  echo "──────── CASE: $label ────────"
  env "$@" TORCH_NCCL_BLOCKING_WAIT=1 timeout -k 20 180 torchrun --standalone --nproc_per_node=2 "$TMPDIR/probe.py" \
      > "$LOGS/$label.log" 2>&1
  local rc=$?
  grep -E "NCCL INFO (comm .* rank|Channel 00|Connected all)" "$LOGS/$label.log" | head -12
  grep -E "init ok|allgather ok|PASS|rror|[Tt]imed? ?out|Watchdog" "$LOGS/$label.log" | tail -15
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then echo "    RESULT $label = HUNG (timeout, rc=$rc)"
  elif [ "$rc" -ne 0 ]; then echo "    RESULT $label = FAILED rc=$rc"
  else echo "    RESULT $label = OK"; fi
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
}

run_case 1-nccl-default         PROBE_BACKEND=nccl NCCL_DEBUG=INFO
run_case 2-nccl-P2P_DISABLE     PROBE_BACKEND=nccl NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO
run_case 3-nccl-P2P+SHM_DISABLE PROBE_BACKEND=nccl NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO
run_case 4-gloo-control         PROBE_BACKEND=gloo
echo "NCCL_PROBE_DONE logs=$LOGS"
