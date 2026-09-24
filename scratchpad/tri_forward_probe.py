"""One deterministic mask-arm forward of ContactMapTriSiT under WHICHEVER checkout is on PYTHONPATH.

The mask arm (no topology keys -> a single MASK token, zero reference features) touches neither the CA feature
columns nor the vocab rows the ConFind surgery changes, so the old code + old weights and the new code + surgered
weights must give identical contact logits. Saves {contact_map_logits} to OUT.

Usage: python scratchpad/tri_forward_probe.py NN_YAML CKPT OUT.pt
"""

import sys

import torch
import yaml

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
with torch.no_grad():
    o = model(batch)
torch.save({"contact_map_logits": o["contact_map_logits"]}, out)
print(f"[probe] saved {out}; logits abs-mean {o['contact_map_logits'].abs().mean().item():.6f}")
