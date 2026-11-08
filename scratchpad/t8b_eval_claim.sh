#!/bin/bash
# T8b eval (user 2026-11-08): run a pipeline stage at most ONCE across its per-partition sibling jobs. The first sibling
# to start (and pass the GPU probe) claims the stage with an atomic mkdir, cancels its pending siblings BY JOB ID
# (claims/<stage>.jobs, written at submit), and runs it; a later sibling exits 0 at once. The stage's exit code lands in
# claims/<stage>/rc -- a claim with owner but no rc whose owner job is gone = a dead stage (the orchestrator checks).
#   sbatch body: t8b_eval_claim.sh <stage> <script> [args]     (the script runs under bash in $S/proteina_xrun)
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
E=$S/t8b_eval
STAGE=${1:?stage}
shift
CL=$E/claims/$STAGE
if [ "${T8E_KIND:?}" = gpu ]; then
  # probe BEFORE claiming: an unapproved / unusable GPU must leave the stage to a sibling (user 2026-10-13 type list)
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
  echo "$name" | grep -Eq 'L40S|H100|H200|RTX PRO 6000' || { echo "[claim] GPU '$name' not approved -> exit 1, unclaimed"; exit 1; }
  ( module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
    export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
    source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate && source /home/chenxiou/cue-openfold-env-sm120/activate.sh \
    && python -c "import torch, torch_scatter; x = torch_scatter.scatter_add(torch.ones(4, device='cuda'), torch.tensor([0, 0, 1, 1], device='cuda')); assert x.tolist() == [2.0, 2.0]; print('[probe]', torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())" ) \
    || { echo "[claim] GPU probe failed on '$name' -> exit 1, unclaimed"; exit 1; }
fi
if ! mkdir "$CL" 2>/dev/null; then
  [ -d "$CL" ] || { echo "[claim] mkdir $CL failed and it does not exist -> exit 1"; exit 1; }
  for _ in 1 2 3 4 5 6; do [ -s "$CL/owner" ] && break; sleep 5; done
  [ -s "$CL/owner" ] || { echo "[claim] $CL exists without owner after 30 s -> exit 1"; exit 1; }
  echo "[claim] $STAGE already claimed by $(cat "$CL/owner") -> exit 0"
  exit 0
fi
echo "$SLURM_JOB_ID $SLURM_JOB_PARTITION $(hostname) $(git -C "$S/proteina_xrun" rev-parse --short HEAD) $(md5sum < "$1" | cut -c1-12)" > "$CL/owner"
echo "[claim] $STAGE claimed: $(cat "$CL/owner")"
for j in $(grep -v "^$SLURM_JOB_ID$" "$E/claims/$STAGE.jobs"); do scancel "$j" && echo "[claim] cancelled sibling $j"; done
bash "$@"
rc=$?
echo "$rc" > "$CL/rc"
exit $rc
