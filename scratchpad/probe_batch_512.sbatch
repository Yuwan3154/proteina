#!/bin/bash
#SBATCH --job-name=bs_probe512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
#
# Largest batch that fits at L=512, measured per (architecture, card). Nothing here is assumed:
# the max512 regime raises the tri model's grid to N = 512 + 64 = 576 vs 320 at max256, and its
# dominant term is an N^3 einsum with [B,N,N,d] activations, so the 62.73 GB measured at
# B=2/L=256 does NOT scale linearly. ARCH and ENVKIND are set by the caller.
#   ARCH=tri  ENVKIND=sm120   -> ContactMapTriSiT  on RTX PRO 6000 (cu128 uv venv)
#   ARCH=hier ENVKIND=conda   -> ContactMapHierSiT on L40S        (cu126 conda env)
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
ARCH="${ARCH:?set ARCH to tri or hier}"
export PROBE_L="${PROBE_L:-512}"
ENVKIND="${ENVKIND:?set ENVKIND to sm120 or conda}"

if [ "$ENVKIND" = "sm120" ]; then
    module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
    export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=12.0
    source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
    source /home/chenxiou/cue-openfold-env-sm120/activate.sh
else
    module load miniforge cuda/13.0.1
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null
    conda activate cue_openfold
    set -u
    export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
fi

export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
ARCH="$ARCH" PROBE_L="$PROBE_L" python - <<'PY'
import os
import torch

ARCH = os.environ["ARCH"]
L, T = int(os.environ["PROBE_L"]), 64
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"card: {torch.cuda.get_device_name(0)}  {total:.1f} GB  arch={torch.cuda.get_arch_list()[-2:]}")

# Build from the REAL nn config, not hand-written kwargs: the first attempt died on
# KeyError: 'enc_pair_update_every_n' because a hand-copied kwarg list drifts from the yaml.
from omegaconf import OmegaConf

CFG = {"tri": "contact_map_tri_30M", "hier": "contact_map_hier_topology_30M"}[ARCH]
cfg = OmegaConf.load(f"configs/experiment_config/model/nn/{CFG}.yaml")
kwargs = {k: v for k, v in OmegaConf.to_container(cfg, resolve=False).items()
          if k not in ("name", "nn_class")}
kwargs["max_topology_he_len"] = T
if ARCH == "tri":
    from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT as Model
else:
    from proteinfoundation.nn.contact_map_hier import ContactMapHierSiT as Model
    kwargs.pop("max_topology_he_len", None)
    kwargs["cath_code_dir"] = os.environ["DATA_PATH"] + "/pdb_raw/"
print("building %s from %s.yaml" % (Model.__name__, CFG))
model = Model(**kwargs).cuda()

print("params: %d" % sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)


def attempt(B):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = {
        "contact_map_t": torch.rand(B, L, L, device="cuda"),
        "contact_map_sc": torch.zeros(B, L, L, device="cuda"),
        "mask": torch.ones(B, L, dtype=torch.bool, device="cuda"),
        "t": torch.rand(B, device="cuda"),
        "residue_type": torch.randint(0, 20, (B, L), device="cuda"),
        "res_seq_pdb_idx": torch.arange(L, device="cuda").expand(B, L).contiguous(),
        "chain_break_per_res": torch.zeros(B, L, device="cuda"),
        "topology_he_tokens": torch.randint(2, 60, (B, T), device="cuda"),
        "topology_he_pos_raw": torch.rand(B, T, device="cuda") * 400,
        "topology_he_feat": torch.randn(B, T, T, 10, device="cuda"),
        "topology_tokens": torch.randint(2, 60, (B, 128), device="cuda"),
        "topology_pos_raw": torch.rand(B, 128, device="cuda") * 400,
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)["contact_map_logits"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out.float(), (torch.rand(B, L, L, device="cuda") > 0.5).float())
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated() / 1024**3


best = 0
for B in (1, 2, 3, 4, 6, 8):
    try:
        peak = attempt(B)
        best = B
        print(f"  batch {B:2d}  OK    peak {peak:6.2f} GB  ({100*peak/total:.0f}% of card)")
    except torch.OutOfMemoryError:
        print(f"  batch {B:2d}  OOM")
        break
    except Exception as e:
        print(f"  batch {B:2d}  ERROR {type(e).__name__}: {str(e)[:150]}")
        break

print(f"RESULT arch={ARCH} L={L} largest_fitting_batch={best}")
if best == 0:
    print("RESULT batch 1 does NOT fit on this card.")
else:
    print(f"RESULT accumulate_grad_batches for 2 GPUs at effective batch 32 = {32 // (2 * best)}")
PY
