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
ARCH="$ARCH" python - <<'PY'
import os
import torch

ARCH = os.environ["ARCH"]
L, T = 512, 64
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"card: {torch.cuda.get_device_name(0)}  {total:.1f} GB  arch={torch.cuda.get_arch_list()[-2:]}")

if ARCH == "tri":
    from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT
    model = ContactMapTriSiT(pair_dim=320, tri_hidden=320, n_blocks=12, transition_n=4,
                             dim_cond=128, max_rel_pos=64, max_topology_he_len=T).cuda()
else:
    from proteinfoundation.nn.contact_map_hier import ContactMapHierSiT
    model = ContactMapHierSiT(
        token_dim=128, pair_repr_dim=64, dim_cond=128, n_enc_layers=4, nheads_enc=8,
        block_size=12, super_factor=4, max_seq_len=512, block_featurizer="local_attn",
        d_local=64, n_local_layers=2, nheads_local=2, d_block=192, d_super=192, nheads_dit=8,
        d_cond_dit=128, mlp_ratio=4.0, n_global_layers=8, n_block_layers=4,
        topology_cond=True, topology_vocab_size=65, max_topology_len=128, d_topo=64,
        d_single=96, n_topo_layers=4, nheads_topo=8, topology_reinject=True,
        pair_ref_features="both", contact_map_mode=True, contact_map_input_dim=1,
        non_contact_value=0, predict_coords=False).cuda()

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
