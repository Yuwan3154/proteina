"""Does a checkpoint's `state_dict` differ from its `ema.params`, and by how much?

⛔⛔ WHY. The raw-weights ladder (job 22990055) showed the branch point at step 8,076 as a dramatic
mirror OUTLIER (refl-sign 0.516) against its own neighbours at 8,000 (0.250) and 8,500 (0.016).
Before treating that as a real transient, check the alternative explanation: the branch point was
FROZEN FROM EMA, so its `state_dict` may hold EMA weights. If so, a "raw" ladder that includes it is
silently mixing one EMA checkpoint with four raw ones -- and since EMA LAGS, that alone manufactures
the spike. `anc8076` returning byte-identical numbers under --weights ema and --weights raw is the
hint that prompted this.

Reports, per checkpoint: how many tensors are shared, how many are bit-identical, and the max
relative deviation. Bit-identical throughout => the two fields hold the same weights.
"""

import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
args = ap.parse_args()

print(f"{'checkpoint':>34} {'step':>7} {'shared':>7} {'identical':>10} {'max rel dev':>12} verdict")
for path in args.ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    gs = ck.get("global_step")
    if "ema" not in ck or "state_dict" not in ck:
        print(f"{os.path.basename(path)[:34]:>34} {str(gs):>7} "
              f"{'-':>7} {'-':>10} {'-':>12} MISSING ema or state_dict")
        continue
    ema = ck["ema"]["params"]
    sd = ck["state_dict"]
    # state_dict keys are prefixed for the LightningModule; ema keys address model.model
    shared, identical, worst = 0, 0, 0.0
    for k, v in ema.items():
        cand = [k, f"model.{k}"]
        hit = next((c for c in cand if c in sd), None)
        if hit is None or not torch.is_tensor(v) or not torch.is_tensor(sd[hit]):
            continue
        a, b = v.float(), sd[hit].float()
        if a.shape != b.shape:
            continue
        shared += 1
        if torch.equal(v, sd[hit]):
            identical += 1
        denom = a.abs().max().clamp_min(1e-12)
        worst = max(worst, float(((a - b).abs().max() / denom)))
    verdict = ("state_dict IS the EMA (all identical)" if shared and identical == shared
               else f"GENUINELY DIFFERENT ({shared - identical} of {shared} differ)")
    print(f"{os.path.basename(path)[:34]:>34} {str(gs):>7} {shared:>7} {identical:>10} "
          f"{worst:>12.3e} {verdict}")
    del ck, ema, sd
