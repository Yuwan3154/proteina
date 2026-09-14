#!/bin/bash
#SBATCH --job-name=vocab_all
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_tri/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_tri/logs/%x-%j.err
#
# The name-filtered scan found only the MLM head (44,320). A head sized 44 with an INPUT EMBEDDING
# still sized 65 would be a genuine mismatch, so scan EVERY parameter for vocab-scale dimensions
# rather than concluding from two tensors.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=9.0
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export OMP_NUM_THREADS=2
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
cd /orcd/scratch/orcd/011/chenxiou

python - <<'PY'
import collections
import torch

CKPT = "/orcd/scratch/orcd/011/chenxiou/proteina_tri/store/tri_cb8synth_v5/checkpoints/last.ckpt"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
state = sd.get("state_dict", sd)
tensors = [(k, tuple(v.shape)) for k, v in state.items() if hasattr(v, "shape")]
print(f"{len(tensors)} parameter tensors in the checkpoint\n")

# every tensor carrying a dimension in the vocab range, whatever its name
VOCAB_RANGE = range(30, 90)
hits = [(k, s) for k, s in tensors if any(d in VOCAB_RANGE for d in s)]
print(f"{len(hits)} tensors with a dimension in 30..89 (the vocab-scale band):")
for k, s in hits:
    print(f"  {k:70} {s}")

dims = collections.Counter(d for _, s in hits for d in s if d in VOCAB_RANGE)
print(f"\nvocab-scale dimensions and how many tensors carry each: {dict(dims)}")
print("\n44 = 2 special (PAD, MASK) + 2 SSE types (H,E) x 21 length slots  <- intended")
print("65 = the LEGACY alphabet that still included loops               <- must be absent")
if 65 in dims or 66 in dims:
    print("⛔ LEGACY vocab dimension present -- the model is NOT purely on the H+E+MASK alphabet.")
elif 44 in dims:
    print("⭐ 44 present, no 65/66 anywhere -- the deployed model is consistently on H+E+MASK.")
else:
    print("⚠️ neither 44 nor 65 found in the vocab band; inspect the list above.")
PY
echo "EXIT=$?"
