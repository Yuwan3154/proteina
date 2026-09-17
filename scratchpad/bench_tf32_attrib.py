"""WHERE does the TF32 speedup live -- and does it survive bf16 autocast at all?

`bench_tf32` measured TF32 at +37.5%. That cannot come from where I claimed the fp32 work is:
`weighted_rigid_align` is O(atoms) with a batched 3x3 SVD and cannot be 37% of a 24-block L^2 trunk.

⛔⛔ THE LIKELY CULPRIT IS THE BENCHMARK ITSELF. It calls `model._step()` DIRECTLY, outside
Lightning's Trainer. Lightning's bf16-mixed plugin installs the autocast context around
`training_step`; calling `_step` directly bypasses it, so the whole TRUNK ran in **fp32** -- exactly
the regime TF32 accelerates. Real training runs the trunk in **bf16**, where TF32 should do nothing.
If that is right, +37.5% is an artefact of the harness and does NOT transfer to production.

This settles it by crossing two factors that were previously confounded:
  autocast  in {OFF (what the benchmark did), BF16 (what production does)}
  matmul    in {highest (fp32 cores), high (TF32 cores)}
and additionally splits TRUNK from LOSS, so the remaining speedup can be attributed.

⛔ The loss disables autocast internally regardless, so the BF16 arm still has an fp32 loss -- which
is the point: it is the only fp32 region production actually has.
"""

import argparse
import contextlib
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import hydra
from omegaconf import OmegaConf

from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--n_diff", type=int, default=48)
ap.add_argument("--diff_chunk", type=int, default=8)
args = ap.parse_args()

print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)

CFG_DIR = os.path.join(REPO, "configs", "datasets_config", "pdb")
with hydra.initialize_config_dir(CFG_DIR, version_base=hydra.__version__):
    cfg = hydra.compose(config_name=args.dataset)
OmegaConf.set_struct(cfg, False)
cfg.datamodule.batch_size = 1
cfg.datamodule.num_workers = 0
cfg.datamodule.prefetch_factor = None
dm = hydra.utils.instantiate(cfg.datamodule)
dm.setup("fit")

MODEL_CFG["n_diffusion_samples"] = args.n_diff
MODEL_CFG["diff_chunk"] = args.diff_chunk
MODEL_CFG["t_beta"] = (1.3, 2.0)
model = ContactToCoordTrainer(model_cfg=MODEL_CFG, use_smooth_lddt=False).to("cuda")
model.train()
batch = next(iter(dm.train_dataloader())).to("cuda")
b = model._prepare(batch, True)


def ctx(use_ac):
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_ac else contextlib.nullcontext()


def timeit(fn):
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / args.steps


# Fixed tensors for the LOSS-only arm, captured once so both matmul modes see identical input.
with torch.no_grad(), ctx(True):
    _o = model.model(b)
XD = _o["x_denoised"].detach().float().clone()
XG = _o["x_gt_rep"].detach().float().clone()
SIG = _o["sigma"].detach().float().clone()
AM = _o["atom_mask_rep"].detach().float().clone()
del _o
torch.cuda.empty_cache()


def full_step(use_ac):
    def f():
        with ctx(use_ac):
            loss, _ = model._step(batch, True)
        loss.backward()
        model.zero_grad(set_to_none=True)
    return f


def trunk_only(use_ac):
    def f():
        with ctx(use_ac):
            out = model.model(b)
        s = out["x_denoised"].float().sum()
        s.backward()
        model.zero_grad(set_to_none=True)
    return f


def loss_only():
    def f():
        xd = XD.clone().requires_grad_(True)
        dl, _ = diffusion_loss(xd, XG, SIG, AM, use_smooth_lddt=False)
        dl.mean().backward()
    return f


print(f"\n{'component':>12} {'autocast':>9} {'highest':>9} {'high(TF32)':>11} {'speedup':>9}")
rows = []
for label, mk, acs in (("full step", full_step, (False, True)),
                       ("trunk only", trunk_only, (False, True)),
                       ("loss only", lambda _ac: loss_only(), (None,))):
    for ac in acs:
        t = {}
        for mode in ("highest", "high"):
            torch.set_float32_matmul_precision(mode)
            t[mode] = timeit(mk(ac))
        sp = t["highest"] / t["high"]
        acs_lbl = {False: "OFF", True: "BF16", None: "n/a (fp32)"}[ac]
        print(f"{label:>12} {acs_lbl:>9} {t['highest']:>9.3f} {t['high']:>11.3f} {sp:>8.3f}x",
              flush=True)
        rows.append((label, acs_lbl, t["highest"], t["high"], sp))

print("\n=== reading ===")
fs_off = next(r for r in rows if r[0] == "full step" and r[1] == "OFF")
fs_bf = next(r for r in rows if r[0] == "full step" and r[1] == "BF16")
print(f"  full step, autocast OFF  (the old benchmark): {fs_off[4]:.3f}x from TF32")
print(f"  full step, autocast BF16 (PRODUCTION)       : {fs_bf[4]:.3f}x from TF32")
if fs_bf[4] < 1.03:
    print("\n⛔⛔ VERDICT: the +37.5% was a HARNESS ARTEFACT. Under bf16 autocast -- which is what")
    print("   training actually uses -- TF32 buys essentially nothing, because the trunk's matmuls")
    print("   are already bf16 and the only fp32 left is the small loss. DO NOT adopt it for speed.")
else:
    print(f"\n⭐ VERDICT: TF32 still gives {fs_bf[4]:.3f}x under production's bf16 autocast, so the")
    print("   effect is real in the configuration that matters. Adoption is then a numerics question.")
print("\n⚠️ Also note the absolute times: if 'full step, autocast BF16' is much faster than")
print("   'autocast OFF', the old benchmark was mis-measuring the step cost itself, not just TF32.")
