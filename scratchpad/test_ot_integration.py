"""ContactMapTriSiT + OT head integration checks. CPU only.

The claim being tested is the one the whole ablation rests on: turning the head ON must not change
a single output value at initialisation, because the injection is zero-initialised. If that is not
exactly true, any later difference is not attributable to the coupling.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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
#
# ⛔ This CANNOT be measured on a freshly built model. Every residual branch in TriBlock ends in
# OpenFold's zero-initialised output projection (contact_map_tri.py:80-82) and `out` is zero-init
# too, so at initialisation each block is exactly the identity: nothing travels from the
# query-reference block into the query-query block that `out` reads, and every upstream gradient
# is exactly 0. Perturbing the zero-inits is what a single optimiser step would do anyway.
def unfreeze_(model, std=0.02):
    """Break every all-zero parameter, so the trunk conducts the way a trained one does."""
    n = 0
    with torch.no_grad():
        for _, prm in model.named_parameters():
            if float(prm.abs().max()) == 0.0:
                prm.normal_(0.0, std)
                n += 1
    return n


for mode in ("sinkhorn", "fgw"):
    cfg = dict(BASE)
    cfg["ot_align"] = dict(enabled=True, mode=mode, eps=0.1, n_iter=10, n_outer=3)
    m = ContactMapTriSiT(**cfg)
    n_unfrozen = unfreeze_(m)
    m(batch)["contact_map_logits"].square().mean().backward()
    gp = m.ot_align.project.weight.grad
    gq = m.ot_align.q_proj.weight.grad
    check(f"{mode}: backward through the injection ({n_unfrozen} zero-inits broken)",
          gp is not None and bool((gp.abs() > 0).any()),
          f"|grad project| = {gp.abs().max():.3e}" if gp is not None else "None")
    check(f"{mode}: gradient reaches the cost projection",
          gq is not None and bool((gq.abs() > 0).any()),
          f"|grad q_proj| = {gq.abs().max():.3e}" if gq is not None else "None")
    if mode == "fgw":
        ga = m.ot_align.alpha_logit.grad
        check("fgw: gradient reaches learned alpha", ga is not None and bool(ga.abs() > 0),
              f"|grad alpha| = {ga.abs().item():.3e}" if ga is not None else "None")

# 3a. The substantive wiring claim: in a conducting trunk the injection must actually MOVE the
# output, i.e. the query-reference block really does reach the query-query block `out` reads.
cfg = dict(BASE)
cfg["ot_align"] = dict(enabled=True, mode="sinkhorn", eps=0.1, n_iter=10)
torch.manual_seed(77)
m = ContactMapTriSiT(**cfg).eval()
unfreeze_(m)
with torch.no_grad():
    torch.nn.init.zeros_(m.ot_align.project.weight)
    torch.nn.init.zeros_(m.ot_align.project.bias)
    off = m(batch)["contact_map_logits"].clone()
    torch.nn.init.normal_(m.ot_align.project.weight, std=0.05)
    on = m(batch)["contact_map_logits"]
moved = (on - off).abs().max().item()
check("injection propagates QT -> QQ and changes the output", moved > 1e-6,
      f"max |delta| = {moved:.3e}")

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

# 6. The shipped YAML must actually build the head -- this is what catches a key renamed in one
# place and not the other, which no dict-literal test can see.
import yaml

cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs",
                        "experiment_config", "model", "nn", "contact_map_tri_30M_ot.yaml")
raw = yaml.safe_load(open(cfg_path))
raw.pop("name", None)
raw.pop("nn_class", None)
raw["n_blocks"] = 2
raw["pair_dim"] = raw["tri_hidden"] = 64
raw["dim_cond"] = 32
m = ContactMapTriSiT(**raw).eval()
check("shipped YAML builds the OT head", m.ot_align is not None,
      f"mode={getattr(m.ot_align, 'mode', None)} eps={getattr(m.ot_align, 'eps', None)}")
with torch.no_grad():
    out = m(batch)["contact_map_logits"]
check("shipped YAML forward is finite", bool(torch.isfinite(out).all()))

print()
print("ALL PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
