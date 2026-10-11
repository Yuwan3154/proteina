#!/bin/bash
# Run scratchpad/t8_compile_ddp_stress.py inside an existing GPU allocation (srun --jobid=<holder> --overlap ...), with
# the training env of run_tri_cb8synth_h200x4.sbatch. Usage: bash scratchpad/run_t8_compile_stress.sh NPROC OUT_DIR [args...]
# Run it twice with the same OUT_DIR for warm-cache (requeue) timings; stderr goes to OUT_DIR/run_stderr_<n>.log (n = run id).
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
NPROC="${1:?usage: run_t8_compile_stress.sh NPROC OUT_DIR [args...]}"
OUT=$(realpath -m "${2:?usage: run_t8_compile_stress.sh NPROC OUT_DIR [args...]}")
shift 2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID:-nojob}_stress
mkdir -p "$TMPDIR" "$OUT"
# Compile caches on NODE-LOCAL disk (the script adds /rank<r>): ~87k files from a partial cold run filled the 1.0M-file
# scratch quota on 2026-10-11. First run = cold; a rerun in the same allocation (same node) = warm.
CACHE="${STRESS_CACHE_ROOT:-$TMPDIR/compile_cache}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_LOGS="recompiles,graph_breaks" TORCH_NCCL_DUMP_ON_TIMEOUT=1
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
git log --oneline -1
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_arch_list())"
echo "host $(hostname) cpus $(nproc) tmp $(df -h /tmp | tail -1)"
N=$(ls "$OUT" | grep -c '_stderr_' || true)
ERR="$OUT/run_stderr_${N}.log"
torchrun --standalone --nproc_per_node="$NPROC" scratchpad/t8_compile_ddp_stress.py --out "$OUT" --run_id "$N" "$@" \
    2> >(tee "$ERR" >&2)
rc=$?
sleep 2
echo "recompile log lines: $(grep -c 'Recompiling function' "$ERR" || true); graph-break log lines: $(grep -ci 'graph break' "$ERR" || true); limit warnings: $(grep -ciE 'recompile_limit|cache_size_limit' "$ERR" || true)"
du -sh "$CACHE"/inductor "$CACHE"/triton "$CACHE"/inductor/rank* "$CACHE"/triton/rank* 2>/dev/null | tee -a "$OUT/cache_size.txt"
echo "cache files: $(find "$CACHE" -type f | wc -l)" | tee -a "$OUT/cache_size.txt"
echo "STRESS_EXIT=$rc"
exit $rc
