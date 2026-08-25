#!/bin/bash
#SBATCH --job-name=ckptinsp
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# Verify the checkpoint's EMA layout BEFORE writing the harness against it. Two conflicting
# claims exist: the memory rule says ckpt["ema"]["params"], the EMA callback docstring says
# ckpt["optimizer_states"][0]["ema"]. Guessing wrong is how the last five attempts went.
source /etc/profile >/dev/null 2>&1
set -uo pipefail
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate cue_openfold
set -u
python - <<'PY'
import torch
P = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/store/tri_full384/checkpoints/last.ckpt"
ck = torch.load(P, map_location="cpu", weights_only=False)
print("top-level keys:", sorted(ck.keys()))
print("\nepoch:", ck.get("epoch"), " global_step:", ck.get("global_step"))
sd = ck.get("state_dict", {})
print("\nstate_dict: n=%d" % len(sd))
for k in list(sd)[:3]:
    print("   ", k, tuple(sd[k].shape))
if "ema" in ck:
    e = ck["ema"]
    print("\nck['ema'] type:", type(e), "keys:" , sorted(e.keys()) if isinstance(e, dict) else "n/a")
else:
    print("\nck['ema'] ABSENT")
os_ = ck.get("optimizer_states", [])
print("\noptimizer_states: n=%d" % len(os_))
if os_:
    o0 = os_[0]
    print("   opt0 keys:", sorted(o0.keys()) if isinstance(o0, dict) else type(o0))
    if isinstance(o0, dict) and "ema" in o0:
        ema = o0["ema"]
        print("   opt0['ema'] type:", type(ema), "len:", len(ema))
        if len(ema):
            print("   first 3 shapes:", [tuple(t.shape) for t in list(ema)[:3]])
        # do the EMA tensors line up with the trainable params in state_dict order?
        trainable = [v for k, v in sd.items() if v.dtype.is_floating_point]
        print(f"   n_ema={len(ema)}  n_float_state_dict={len(trainable)}")
PY
