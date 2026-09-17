"""Is the DATALOADER the bottleneck, not the GPU?

An H200 has far more bf16 tensor throughput than an RTX PRO 6000, yet our h200 run measured SLOWER.
`nvidia-smi` reported 99-100% GPU utilisation on both, but that metric only says "a kernel was
resident", not "the GPU was fed" -- it reads high even when the device stalls briefly between steps.

⛔ The arithmetic says the loader SHOULD have headroom: at bs=1 a micro-step is ~4.7 s, and with 8
workers each worker gets ~37 s per sample. So if loading is nevertheless rate-limiting, the cause is
not the worker COUNT -- it is per-sample cost or contention. Two suspects:
  - the processed .pt files live under /orcd/data/so3/... (the COLD tier), not scratch;
  - the h200 node is SHARED with other users' jobs, so its 8 CPUs and its NFS path are contended.

This measures the loader ALONE -- no model, no GPU work -- so any starvation shows up directly, and
sweeps worker counts to separate "too few workers" from "each sample is just slow".

Usage: bench_dataloader.py --batches 40 --workers 0,4,8,16
"""

import argparse
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import hydra
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--workers", default="0,4,8,16")
ap.add_argument("--batch_size", type=int, default=1)
args = ap.parse_args()

WORKERS = [int(v) for v in args.workers.split(",")]

# The compute side, measured on the same card class (job 22906126): 7.53 s per fwd+bwd at bs=1,
# and production tbeta sustains ~4.7 s per micro-step. The loader must beat that to not be limiting.
COMPUTE_S_PER_BATCH = 4.7

CFG_DIR = os.path.join(REPO, "configs", "datasets_config", "pdb")
assert os.path.isdir(CFG_DIR), f"config dir missing: {CFG_DIR}"

print(f"[node] {os.uname().nodename}", flush=True)
print(f"[reference] production micro-step ~{COMPUTE_S_PER_BATCH:.1f} s at bs=1 -- the loader must "
      f"deliver a batch faster than this", flush=True)
print(f"\n{'workers':>8} {'s/batch':>9} {'median':>9} {'p90':>9} {'batches/s':>10} {'verdict':>28}")

for nw in WORKERS:
    with hydra.initialize_config_dir(CFG_DIR, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.batch_size = args.batch_size
    cfg.datamodule.num_workers = nw
    # ⛔ prefetch_factor must be None when num_workers=0; PyTorch raises otherwise.
    cfg.datamodule.prefetch_factor = 2 if nw > 0 else None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    it = iter(dm.train_dataloader())
    times = []
    for i in range(args.warmup + args.batches):
        t0 = time.time()
        b = next(it)
        # touch the tensors so lazy work cannot hide behind the timer
        _ = b["coords"].shape, b["contact_map"].shape
        dt = time.time() - t0
        if i >= args.warmup:
            times.append(dt)
    mean = sum(times) / len(times)
    med = statistics.median(times)
    p90 = sorted(times)[int(0.9 * len(times)) - 1]
    verdict = "LIMITING" if mean > COMPUTE_S_PER_BATCH else \
              ("marginal" if mean > 0.5 * COMPUTE_S_PER_BATCH else "not limiting")
    print(f"{nw:>8} {mean:>9.3f} {med:>9.3f} {p90:>9.3f} {1/mean:>10.2f} {verdict:>28}", flush=True)
    del dm, it

print("\n⭐ Read the P90, not just the mean: a loader that is fine on average but stalls on 10% of")
print("   batches still leaves the GPU idle for those, and that is exactly what a contended NFS")
print("   path looks like. A flat profile across worker counts means per-sample cost (I/O or")
print("   transform), not worker starvation.")
