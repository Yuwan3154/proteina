"""Measure the training-time activation memory of the T8 tri (ContactMapTriV2), one fresh process per variant.

Activation memory = the unique storages autograd keeps for backward (saved_tensors_hooks; a non-reentrant
checkpoint's inputs pass through them too), minus parameters; float32 on CPU, batch 1. (RSS deltas were tried
and are not usable on macOS.) Reported in MB and in units U = one [N, N, 128] float32 tensor (N = L + T), so it
extrapolates as N^2 at a fixed T / L (stage B scales as L^2). Stored activations, not the backward peak; eager fp32
(bf16 autocast and torch.compile will change the ratios).
Variants: head = a given older model file (e.g. the global-vector conditioning); pair = the current file;
pair_plain = the current file with the AdaLN / gate recompute (checkpoint) replaced by the plain computation.
Usage: python t8_activation_memory.py L T [head_model_file]
"""

import importlib.util
import json
import os
import subprocess
import sys

import torch

CFG = dict(pair_dim=128, tri_hidden=128, n_blocks_ref=4, n_blocks_query=4, transition_hidden=512, dim_cond=128,
           max_topology_he_len=96, max_rel_pos=32, topology_vocab_size=45, n_elem_features=1, n_residue_types=22,
           pair_ref_features="both", align_head={"enabled": True}, mlm_head={"enabled": True})


def batch(L, T):
    g = torch.Generator().manual_seed(0)
    return {
        "contact_map_t": (torch.rand(1, L, L, generator=g) > 0.9).float(),
        "contact_map_sc": torch.rand(1, L, L, generator=g),
        "mask": torch.ones(1, L),
        "residue_type": torch.randint(0, 20, (1, L), generator=g),
        "t": torch.rand(1, generator=g),
        "topology_he_tokens": torch.randint(2, 45, (1, T), generator=g),
        "topology_he_pos_raw": torch.sort(torch.rand(1, T, generator=g) * L, dim=1).values,
        "topology_he_feat": torch.randn(1, T, T, 8, generator=g),
        "topology_he_elem_feat": torch.randn(1, T, 1, generator=g),
    }


def measure(variant, L, T, head_file):
    if variant == "head":
        spec = importlib.util.spec_from_file_location("cmtv2_head", head_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import proteinfoundation.nn.contact_map_tri_v2 as mod
    held = {}

    def keep(x):
        held[x.untyped_storage().data_ptr()] = x.untyped_storage().nbytes()

    if variant == "pair_plain":
        mod.checkpoint = lambda f, *a, **k: f(*a)
    torch.manual_seed(0)
    m = mod.ContactMapTriV2(**CFG)
    b = batch(L, T)

    def pack(x):
        keep(x)
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        out = m(b)
    for p in m.parameters():
        held.pop(p.untyped_storage().data_ptr(), None)
    loss = sum(v.float().sum() for k, v in out.items() if k != "contact_map_pred")
    loss.backward()
    total = sum(held.values())
    unit = (L + T) ** 2 * 128 * 4
    return {"variant": variant, "L": L, "T": T, "act_MB": round(total / 2**20, 1), "act_U": round(total / unit, 2)}


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        torch.set_num_threads(4)
        print(json.dumps(measure(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "")))
        sys.exit(0)
    L, T = int(sys.argv[1]), int(sys.argv[2])
    head = sys.argv[3] if len(sys.argv) > 3 else ""
    for v in (["head"] if head else []) + ["pair_plain", "pair"]:
        r = subprocess.run([sys.executable, os.environ["NSRUN"], __file__, "--one", v, str(L), str(T), head],
                           capture_output=True, text=True, check=True)
        print(r.stdout.strip().splitlines()[-1], flush=True)
