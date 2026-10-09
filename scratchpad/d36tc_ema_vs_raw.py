"""D36tc: file-based EMA-ness test for the frozen tri. A raw checkpoint at the SAME step must DIFFER from the -EMA
file (EMA != raw). Reports identical / differing tensor counts and the max relative deviation; also md5-pins the
run's last-EMA.ckpt to the frozen copy. Identical state_dicts are reported, never read as proof (D10: a ckpt written
at validation end can carry EMA weights as its state_dict).
"""
import argparse
import hashlib

import torch


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--ema", required=True)
ap.add_argument("--run-last-ema", required=True)
ap.add_argument("--step", type=int, required=True)
args = ap.parse_args()

m_frozen, m_run = md5(args.ema), md5(args.run_last_ema)
print(f"md5 frozen {m_frozen}  run last-EMA {m_run}  equal={m_frozen == m_run}")
assert m_frozen == m_run, "the run's last-EMA.ckpt is not the frozen copy"
raw = torch.load(args.raw, map_location="cpu", weights_only=False, mmap=True)
ema = torch.load(args.ema, map_location="cpu", weights_only=False, mmap=True)
print(f"global_step raw {raw.get('global_step')} ema {ema.get('global_step')} (want {args.step})")
assert raw.get("global_step") == args.step == ema.get("global_step"), "not the same step"
a, b = raw["state_dict"], ema["state_dict"]
assert set(a) == set(b), f"key sets differ: {len(set(a) ^ set(b))}"
same, diff, maxrel, worst = 0, 0, 0.0, ""
for k in a:
    if torch.equal(a[k], b[k]):
        same += 1
        continue
    diff += 1
    if a[k].is_floating_point():
        r = float((a[k].float() - b[k].float()).norm() / b[k].float().norm().clamp_min(1e-12))
        if r > maxrel:
            maxrel, worst = r, k
print(f"state_dict raw vs EMA: {len(a)} tensors, identical {same}, differ {diff}, max rel dev {maxrel:.4g} ({worst})")
print("VERDICT: " + ("EMA != raw (consistent with an EMA file)" if diff else "IDENTICAL -- inconclusive, report"))
