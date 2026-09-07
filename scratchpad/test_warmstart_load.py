"""Does the A+B warm start actually load? Run this BEFORE the training slot opens.

The real run starts at ~16:10 unattended and carries CHAIN=40. If the scoped-strict=False
assertion is wrong, the job dies at startup and every one of the 40 chained successors dies the
same way, with nobody watching. This exercises exactly that code path -- build the post-fix-A
model, load the pre-fix-A EMA, check the assertion -- on CPU in seconds, with no GPU and no
training.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

CKPT = ("/orcd/scratch/orcd/011/chenxiou/c2c_store/"
        "c2c_v1_qknorm_15500_22168866/last.ckpt")
MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4, n_diffusion_samples=16,
)
ALLOWED = ("atom_enc.dist_proj", "atom_enc.valid_proj", "atom_enc.pair_mlp")

fails = []
model = ContactToCoordTrainer(model_cfg=MODEL_CFG)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"checkpoint: {CKPT}\n  global_step={sd.get('global_step')}  has_ema={'ema' in sd}")

missing, unexpected = model.model.load_state_dict(sd["ema"]["params"], strict=False)
bad = [k for k in missing if not any(k.startswith(a) for a in ALLOWED)]
mods = sorted(set(k.rsplit(".", 1)[0] for k in missing))

print(f"\n  missing keys    : {len(missing)}")
for m in mods:
    print(f"      {m}")
print(f"  unexpected keys : {len(unexpected)}  {list(unexpected)[:6]}")

if bad:
    fails.append(f"PRE-EXISTING params would be left uninitialised: {bad[:8]}")
if unexpected:
    fails.append(f"checkpoint carries params the model lacks: {list(unexpected)[:8]}")
if not missing:
    fails.append("no missing keys at all -- fix A's modules are absent from the model")

# The loaded weights must be the EMA ones, not random. Compare a tensor that exists in both.
k = next(k for k in sd["ema"]["params"] if k.endswith("weight") and "atom_enc.ref_proj" in k)
same = torch.allclose(dict(model.model.named_parameters())[k].detach(),
                      sd["ema"]["params"][k].float())
if not same:
    fails.append(f"{k} does not match the checkpoint after load")
print(f"\n  spot-check {k}: {'matches checkpoint' if same else 'MISMATCH'}")

# And the newly-added modules must be finite and non-degenerate, not zeros or NaN.
for nm in ALLOWED:
    ps = [p for n, p in model.model.named_parameters() if n.startswith(nm)]
    if not ps:
        fails.append(f"{nm} has no parameters")
        continue
    tot = sum(float(p.abs().sum()) for p in ps)
    ok = all(torch.isfinite(p).all() for p in ps) and tot > 0
    print(f"  {nm:<26} {len(ps)} tensors, |sum|={tot:.3f}  {'ok' if ok else 'DEGENERATE'}")
    if not ok:
        fails.append(f"{nm} is degenerate after init")

print("\n" + ("WARM START OK -- the 16:10 job will load" if not fails else "FAILED:"))
for f in fails:
    print("  " + f)
sys.exit(1 if fails else 0)
