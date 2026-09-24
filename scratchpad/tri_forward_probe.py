"""One deterministic mask-arm forward of ContactMapTriSiT under WHICHEVER checkout is on PYTHONPATH.

The mask arm (no topology keys -> a single MASK token, zero reference features) touches neither the CA feature
columns nor the vocab rows the ConFind surgery changes, so the old code + old weights and the new code + surgered
weights must give identical contact logits. Saves {contact_map_logits} to OUT.

With a 4th arg "ref", the batch also carries helix/strand reference elements encoded in the model's OWN alphabet
(vocab 65: types loop,helix,strand; vocab 44: helix,strand) and pair features filled by NAME, CA-distance channels
at 0 (their standardised mean) -- this exercises the surgered vocab rows too.

Usage: python scratchpad/tri_forward_probe.py NN_YAML CKPT OUT.pt [ref]
"""

import sys

import torch
import yaml

from proteinfoundation.datasets.sse_topology import PAIR_FEATURE_NAMES
from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT

nn_yaml, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
kw = yaml.safe_load(open(nn_yaml))
kw.pop("name", None)
kw.pop("nn_class", None)
model = ContactMapTriSiT(**kw)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
res = model.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith("nn.")}, strict=False)
print(f"[probe] missing={res.missing_keys} unexpected={res.unexpected_keys}")
model.eval()

g = torch.Generator().manual_seed(0)
B, L = 2, 48
mask = torch.ones(B, L)
mask[1, 40:] = 0
cm = (torch.rand(B, L, L, generator=g) < 0.08).float()
cm = torch.triu(cm, 1)
cm = (cm + cm.transpose(1, 2)) * mask[:, :, None] * mask[:, None, :]
batch = {"contact_map_t": cm, "contact_map_sc": torch.rand(B, L, L, generator=g) * mask[:, :, None] * mask[:, None, :],
         "mask": mask, "residue_type": torch.randint(0, 20, (B, L), generator=g), "t": torch.tensor([0.3, 0.7])}
if len(sys.argv) > 4 and sys.argv[4] == "ref":
    T = 6
    types = (0, 1, 2) if model.topology_vocab_size == 65 else (1, 2)   # DSSP loop=0, helix=1, strand=2
    elem = [(1, 3), (2, 5), (1, 12), (2, 1), (1, 20), (2, 7)]          # (dssp type, slot)
    tok = torch.tensor([[2 + types.index(t) * 21 + sl for t, sl in elem]] * B)
    tok[1, 4:] = 0                                                     # second sample: 4 real elements
    shared = torch.randn(T, T, 8, generator=g)                         # values for the 8 names both alphabets share
    shared_names = [n for n in PAIR_FEATURE_NAMES if n not in ("min_ca_dist", "mean_ca_dist")]
    feat = torch.zeros(B, T, T, len(PAIR_FEATURE_NAMES))
    for j, n in enumerate(shared_names):
        feat[:, :, :, PAIR_FEATURE_NAMES.index(n)] = shared[:, :, j]
    feat = feat * (tok > 0)[:, :, None, None] * (tok > 0)[:, None, :, None]
    batch.update({"topology_he_tokens": tok, "topology_he_pos_raw": torch.tensor([[5., 14., 22., 30., 37., 44.]] * B),
                  "topology_he_feat": feat})
with torch.no_grad():
    o = model(batch)
torch.save({"contact_map_logits": o["contact_map_logits"]}, out)
print(f"[probe] saved {out}; logits abs-mean {o['contact_map_logits'].abs().mean().item():.6f}")
