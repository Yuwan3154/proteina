"""Does TF32 for fp32 matmuls buy speed, and does it move the loss?

PyTorch warns in every one of our runs that `torch.set_float32_matmul_precision` is unset, so the
fp32 matmuls run on CUDA cores instead of TF32 tensor cores. The trunk is bf16 under autocast, but
the LOSS is deliberately fp32 with autocast disabled, so those matmuls are exactly what the warning
is about.

⛔⛔ SPEED IS THE EASY HALF. The hard half is whether TF32 is SAFE here: `weighted_rigid_align` is a
Kabsch SVD whose docstring says it "wants full precision regardless", and TF32 carries a 10-bit
mantissa against fp32's 23. A speedup that quietly perturbs the alignment would corrupt the training
signal while every dashboard still looked healthy. So this measures BOTH, on the SAME batch with the
SAME noise draw, and reports the loss delta beside the timing.

⛔ set_float32_matmul_precision is GLOBAL and sticky, so each arm sets it explicitly rather than
assuming the previous state.

Usage: bench_tf32.py --steps 6
"""

import argparse
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

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--n_diff", type=int, default=48)
ap.add_argument("--diff_chunk", type=int, default=8)
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()

print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)

CFG_DIR = os.path.join(REPO, "configs", "datasets_config", "pdb")
assert os.path.isdir(CFG_DIR), f"config dir missing: {CFG_DIR}"
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

# ⛔ ONE fixed batch, held on the device, so the two arms see identical inputs. Pulling a fresh
# batch per arm would let data variation masquerade as a precision effect.
batch = next(iter(dm.train_dataloader())).to("cuda")

# ⛔⛔ ORDER CONTROL. The first arm pays one-time costs the second inherits for free: cuDNN autotune
# caches, CUDA allocator growth, lazy module init. Running highest-then-high ONCE credited TF32 with
# +37.6%, which is implausible for a loss region that is O(atoms) with a 3x3 SVD. Run each arm TWICE
# in alternating order and compare the SECOND occurrences, when both are equally warm.
results = {}
rounds = {}
for rnd, mode in enumerate(("highest", "high", "highest", "high")):
    torch.set_float32_matmul_precision(mode)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    # numerics first, on an identical noise draw
    torch.manual_seed(args.seed)
    with torch.no_grad():
        loss_val, metrics = model._step(batch, True)
    loss_val = float(loss_val)
    mse_val = float(metrics["mse"])
    rmsd_val = float(metrics["rmsd"])

    # then timing, fwd+bwd, no optimizer so we time the model and loss only
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        loss, _ = model._step(batch, True)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.steps
    results[mode] = (dt, loss_val, mse_val, rmsd_val, tf32)
    rounds.setdefault(mode, []).append(dt)
    print(f"  round {rnd} {mode:>8}  allow_tf32={tf32!s:>5}  {dt:>7.3f} s/pass   "
          f"loss={loss_val:.8f}  mse={mse_val:.8f}  rmsd={rmsd_val:.6f}", flush=True)

_, l_hi, m_hi, r_hi, _ = results["highest"]
_, l_tf, m_tf, r_tf, _ = results["high"]
# ⭐ SECOND occurrence of each -- both fully warm, so the comparison is not a warmup artefact.
dt_hi, dt_tf = rounds["highest"][-1], rounds["high"][-1]

print(f"\n=== speed ===")
print(f"  highest rounds: {['%.3f' % v for v in rounds['highest']]}")
print(f"  high    rounds: {['%.3f' % v for v in rounds['high']]}")
print(f"  ⭐ comparing the SECOND of each (both warm):")
print(f"  highest (fp32 cores) : {dt_hi:.3f} s/pass")
print(f"  high    (TF32 cores) : {dt_tf:.3f} s/pass")
print(f"  speedup              : {dt_hi/dt_tf:.4f}x  ({100*(dt_hi-dt_tf)/dt_hi:+.2f}% time)")
warm_drop = 100 * (rounds['highest'][0] - rounds['highest'][1]) / rounds['highest'][0]
print(f"  ⚠️ warmup artefact size: 'highest' alone dropped {warm_drop:+.1f}% between its two rounds")

print(f"\n=== numerics on an IDENTICAL batch and noise draw ===")
for nm, a, b in (("loss", l_hi, l_tf), ("mse", m_hi, m_tf), ("rmsd", r_hi, r_tf)):
    rel = abs(a - b) / max(abs(a), 1e-12)
    print(f"  {nm:>5}: highest={a:.8f}  high={b:.8f}  rel diff={rel:.3e}")

rel_loss = abs(l_hi - l_tf) / max(abs(l_hi), 1e-12)
print(f"\nVERDICT:")
if dt_hi / dt_tf < 1.02:
    print("  SPEED: negligible (<2%). The fp32 region is O(atoms) with a 3x3 SVD, not O(L^2),")
    print("         so there is little fp32 matmul for TF32 to accelerate. Not worth a numerics risk.")
else:
    print(f"  SPEED: {100*(dt_hi-dt_tf)/dt_hi:.1f}% faster -- worth considering.")
print(f"  NUMERICS: relative loss change {rel_loss:.3e}")
print("  ⚠️ Judge this against the EDM weight's ~5-order-of-magnitude span, not against zero:")
print("     a small relative change in the loss can still be a large change in the gradient at")
print("     the high-sigma tail. If speed is negligible, do not take the risk at all.")
