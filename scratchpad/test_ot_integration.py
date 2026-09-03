"""ContactMapTriSiT + OT head integration checks. CPU only.

The claim being tested is the one the whole ablation rests on: turning the head ON must not change
a single output value at initialisation, because the injection is zero-initialised. If that is not
exactly true, any later difference is not attributable to the coupling.
"""

import sys

import torch

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier")

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES
from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


BASE = dict(
    pair_dim=64, tri_hidden=64, n_blocks=2, transition_n=4, dim_cond=32, max_rel_pos=64,
    topology_cond=True, max_topology_he_len=64, topology_vocab_size=65, n_residue_types=22,
    pair_ref_features="both", contact_map_mode=True, contact_map_input_dim=1,
    non_contact_value=0, predict_coords=False,
)


def make_batch(B=2, L=24, T=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        mask[b, : L - 3 * b] = True
    cm = (torch.rand(B, L, L, generator=g) > 0.8).float()
    cm = ((cm + cm.transpose(1, 2)) > 0).float()
    he_tokens = torch.randint(1, 60, (B, T), generator=g)
    he_tokens[1, -2:] = 0                                   # ragged reference
    return {
        "contact_map_t": cm,
        "contact_map_sc": cm.clone(),
        "mask": mask,
        "residue_type": torch.randint(0, 21, (B, L), generator=g),
        "topology_he_tokens": he_tokens,
        "topology_he_pos_raw": torch.sort(torch.rand(B, T, generator=g) * 200, dim=1).values,
        "topology_he_feat": torch.rand(B, T, T, N_PAIR_FEATURES, generator=g),
        "t": torch.rand(B, generator=g),
    }


batch = make_batch()

# 1. Disabled head -> the model must be exactly the model we have been training.
torch.manual_seed(1234)
base = ContactMapTriSiT(**BASE).eval()
with torch.no_grad():
    out_base = base(batch)["contact_map_logits"]
check("baseline forward runs", torch.isfinite(out_base).all().item(),
      f"shape {tuple(out_base.shape)}")
check("baseline has no OT head", base.ot_align is None)

# 2. Enabled head at init -> bit-identical output.
for mode in ("sinkhorn", "fgw"):
    torch.manual_seed(1234)
    cfg = dict(BASE)
    cfg["ot_align"] = dict(enabled=True, mode=mode, eps=0.1, n_iter=10, n_outer=3)
    m = ContactMapTriSiT(**cfg).eval()
    # Same trunk weights as the baseline: the head's extra params are drawn after, so re-seeding
    # alone does not guarantee it. Copy the baseline trunk in explicitly.
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    check(f"{mode}: trunk weights copied, only OT params extra",
          len(unexpected) == 0 and all(k.startswith("ot_align.") for k in missing),
          f"{len(missing)} OT params, {len(unexpected)} unexpected")
    with torch.no_grad():
        out = m(batch)["contact_map_logits"]
    delta = (out - out_base).abs().max().item()
    check(f"{mode}: enabled-at-init output is BIT-IDENTICAL", delta == 0.0,
          f"max |delta| = {delta:.3e}")

# 3. Backward works through the in-place query-reference injection.
for mode in ("sinkhorn", "fgw"):
    cfg = dict(BASE)
    cfg["ot_align"] = dict(enabled=True, mode=mode, eps=0.1, n_iter=10, n_outer=3)
    m = ContactMapTriSiT(**cfg)
    torch.nn.init.normal_(m.ot_align.project.weight, std=0.05)
    m(batch)["contact_map_logits"].square().mean().backward()
    gp = m.ot_align.project.weight.grad
    gq = m.ot_align.q_proj.weight.grad
    check(f"{mode}: backward through the injection", gp is not None and bool((gp.abs() > 0).any()),
          f"|grad project| = {gp.abs().max():.3e}" if gp is not None else "None")
    check(f"{mode}: gradient reaches the cost projection",
          gq is not None and bool((gq.abs() > 0).any()),
          f"|grad q_proj| = {gq.abs().max():.3e}" if gq is not None else "None")

# 4. Degenerate case: no topology reference at all must not produce NaN.
nb = make_batch(seed=3)
for k in ("topology_he_tokens", "topology_he_pos_raw", "topology_he_feat"):
    nb.pop(k)
cfg = dict(BASE)
cfg["ot_align"] = dict(enabled=True, mode="sinkhorn", eps=0.1, n_iter=10)
m = ContactMapTriSiT(**cfg).eval()
with torch.no_grad():
    out = m(nb)["contact_map_logits"]
check("no-topology batch stays finite", bool(torch.isfinite(out).all()))

# 5. All-masked reference elements (every token 0) must not produce NaN either.
nb2 = make_batch(seed=4)
nb2["topology_he_tokens"] = torch.zeros_like(nb2["topology_he_tokens"])
with torch.no_grad():
    out = m(nb2)["contact_map_logits"]
check("all-invalid reference stays finite", bool(torch.isfinite(out).all()))

print()
print("ALL PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
