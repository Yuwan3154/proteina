"""Regression test: can a PRE-p_mirror checkpoint be STRICTLY resumed?

⛔⛔ THE BUG THIS GUARDS. `to_hand_s` was created unconditionally when the p_mirror feature landed,
so it appeared in every state_dict. Lightning resumes with strict=True, so restarting any run whose
checkpoint predates the feature died with:
    RuntimeError: Error(s) in loading state_dict for ContactToCoordTrainer:
        Missing key(s) in state_dict: "model.to_hand_s.weight".
That killed the c2c_cb8_tbeta restart from step 10,290 (jobs 23021859 / 23021923, 2026-09-18) and
would have consumed the entire 36-deep chain at ~1 min per failure.

⛔ Test 3 is the one that matters: it loads the REAL checkpoint strictly, which is exactly what
Lightning does on resume. Tests 1 and 2 alone would pass even if the real resume still broke.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)
CKPT = "/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_cb8_tbeta/last.ckpt"

fails = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")
    if not ok:
        fails.append(name)


print("1. p_mirror=0 must NOT create to_hand_s")
# ⛔ p_mirror is NOT a trainer kwarg -- it travels inside model_cfg, exactly as
# train_c2c.py does it (`MODEL_CFG["p_mirror"] = args.p_mirror`, default 0.0).
m0 = ContactToCoordTrainer(model_cfg={**MODEL_CFG, "p_mirror": 0.0})
k0 = [k for k in m0.state_dict() if "to_hand_s" in k]
check("no to_hand_s key at p_mirror=0", k0 == [], f"found {k0}")

print("2. p_mirror>0 must still create it (feature intact)")
m1 = ContactToCoordTrainer(model_cfg={**MODEL_CFG, "p_mirror": 0.02})
k1 = [k for k in m1.state_dict() if "to_hand_s" in k]
check("to_hand_s present at p_mirror=0.02", len(k1) == 1, f"found {k1}")

print("3. THE REGRESSION: strict load of the real pre-p_mirror checkpoint")
if not os.path.exists(CKPT):
    check("checkpoint reachable", False, f"missing {CKPT}")
else:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    has_hand = any("to_hand_s" in k for k in sd)
    print(f"     checkpoint global_step={ck.get('global_step')}, "
          f"contains to_hand_s: {has_hand}")
    try:
        m0.load_state_dict(sd, strict=True)          # exactly what Lightning does on resume
        check("STRICT load succeeds (resume will work)", True)
    except RuntimeError as e:
        check("STRICT load succeeds (resume will work)", False, str(e).split("\n")[1].strip())

print()
if fails:
    print(f"FAILED: {len(fails)} -> {fails}")
    sys.exit(1)
print("ALL RESUME REGRESSION TESTS PASS")
