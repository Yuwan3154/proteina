"""ContactMapTriSiT compile + padding checks (tiny model; CPU or GPU).

1. padding invariance: outputs at real positions are identical with and without bucket padding (eager);
2. gradient invariance: parameter gradients of a masked loss are identical padded vs unpadded;
3. compiled vs eager: same outputs (grad and no-grad entry points), padded shapes repeat => no recompile;
4. state_dict keys unchanged by compile (no _orig_mod prefix), so old checkpoints load and new ones stay portable.

Usage: python scratchpad/test_tri_compile_padding.py [--device cuda] [--skip_compile]
"""

import argparse
import sys

import torch

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES
from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--skip_compile", action="store_true")
args = ap.parse_args()
dev = torch.device(args.device)
torch.manual_seed(0)

BASE = dict(pair_dim=16, tri_hidden=16, n_blocks=2, transition_n=2, dim_cond=16, max_topology_he_len=64,
            max_rel_pos=8, topology_vocab_size=44, n_residue_types=22, pair_ref_features="both",
            align_head={"enabled": True}, mlm_head={"enabled": True})
BUCKETS = dict(pad_len_buckets=[16, 32, 48], pad_topo_buckets=[8, 16])


def make_batch(B, L, T, n_real_L, n_real_T):
    mask = torch.zeros(B, L)
    tok = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        mask[b, : n_real_L[b]] = 1
        tok[b, : n_real_T[b]] = torch.randint(1, 44, (n_real_T[b],))
    cm = (torch.rand(B, L, L) < 0.1).float()
    cm = torch.triu(cm, 1); cm = cm + cm.transpose(1, 2)
    cm = cm * mask[:, :, None] * mask[:, None, :]
    return {
        "contact_map_t": cm, "contact_map_sc": torch.rand(B, L, L) * mask[:, :, None] * mask[:, None, :],
        "mask": mask, "residue_type": torch.randint(0, 20, (B, L)) * mask.long(),
        "topology_he_tokens": tok, "topology_he_pos_raw": torch.rand(B, T) * L,
        "topology_he_feat": torch.randn(B, T, T, N_PAIR_FEATURES) * (tok > 0)[:, :, None, None]
        * (tok > 0)[:, None, :, None],
        "t": torch.rand(B),
    }


def to(b):
    return {k: v.to(dev) for k, v in b.items()}


def compare(o1, o2, tag, tol):
    worst = 0.0
    for k in o1:
        if torch.is_tensor(o1[k]):
            assert o1[k].shape == o2[k].shape, f"{tag} {k}: shape {tuple(o1[k].shape)} vs {tuple(o2[k].shape)}"
            worst = max(worst, (o1[k].float() - o2[k].float()).abs().max().item())
    ok = worst <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: max |diff| over {len(o1)} outputs = {worst:.3g} (tol {tol})")
    return ok


results = []
for variant, extra in (("align+mlm", {}), ("align+mlm+ot", {"ot_align": {"enabled": True}})):
    ref = ContactMapTriSiT(**BASE, **extra).to(dev)
    with torch.no_grad():  # zero-init output layers would make every comparison trivially 0
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    pad = ContactMapTriSiT(**BASE, **extra, **BUCKETS).to(dev)
    pad.load_state_dict(ref.state_dict())
    batch = to(make_batch(2, 21, 5, [21, 13], [5, 3]))  # L=21 -> 32, T=5 -> 8
    with torch.no_grad():
        results.append(compare(ref(batch), pad(batch), f"{variant} padding invariance (eager)", 1e-5))

    def masked_loss(m, b):
        o = m(b)
        q = b["mask"][:, :, None] * b["mask"][:, None, :]
        l = (o["contact_map_logits"] * q).square().sum() + o["align_logits"].square().sum() + o["mlm_logits"].square().sum()
        return l
    for m in (ref, pad):
        m.zero_grad()
        masked_loss(m, batch).backward()
    gd = max((p1.grad - p2.grad).abs().max().item() for p1, p2 in zip(ref.parameters(), pad.parameters())
             if p1.grad is not None)
    ok = gd <= 1e-5
    print(f"[{'PASS' if ok else 'FAIL'}] {variant} gradient invariance: max |dgrad| = {gd:.3g}")
    results.append(ok)

    if not args.skip_compile:
        comp = ContactMapTriSiT(**BASE, **extra, **BUCKETS, use_torch_compile=True, use_torch_compile_sc=True).to(dev)
        comp.load_state_dict(ref.state_dict())
        keys_ok = list(comp.state_dict()) == list(ref.state_dict())
        print(f"[{'PASS' if keys_ok else 'FAIL'}] {variant} state_dict keys identical to eager ({len(ref.state_dict())} keys)")
        results.append(keys_ok)
        torch._dynamo.utils.counters.clear()
        with torch.no_grad():
            results.append(compare(ref(batch), comp(batch), f"{variant} compiled no-grad vs eager", 1e-4))
        results.append(compare(ref(batch), comp(batch), f"{variant} compiled grad vs eager", 1e-4))
        # a different real length in the SAME bucket must reuse the graphs
        b2 = to(make_batch(2, 27, 7, [27, 20], [7, 2]))  # L=27 -> 32, T=7 -> 8
        n0 = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        with torch.no_grad():
            results.append(compare(ref(b2), comp(b2), f"{variant} compiled no-grad, same bucket", 1e-4))
        n1 = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        ok = n1 == n0
        print(f"[{'PASS' if ok else 'FAIL'}] {variant} same-bucket call compiled no new graph ({n0} -> {n1})")
        results.append(ok)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
