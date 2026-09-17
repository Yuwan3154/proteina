"""How fast is a c2c training step, and how large a batch actually fits?

Two questions, one harness, because both need the same thing: real training steps timed with real
memory accounting.

⛔⛔ WHY THIS EXISTS. I compared c2c_cb8_tbeta (rtx_pro_6000, pi_so3) against c2c_cb8_fape (h200,
mit_preemptable) and reported the RTX as ~12% faster. That comparison conflated THREE variables:
  - the h200 node was SHARED with two other users' GPU jobs; the pi_so3 node was exclusive;
  - the h200 run also computes the FAPE loss, which the rtx run does not;
  - the two cards run at 2415 vs 1980 MHz SM clock.
⇒ It was not a hardware measurement. Run this on both card types on the SAME partition, with
identical settings, to get one.

⛔ batch_size=1 is an INHERITED DEFAULT, not a measurement for this model. The dataset yaml's
batch_size=1 is documented in train_c2c.py as "a real measurement for THAT model (tri_sm120,
71.8 GB at L=384). It says nothing about this one." c2c uses ~35 GB of 144 GB at bs=1, so there is
large headroom and the sweep below finds where it actually ends.

⭐ Raising batch_size while lowering accum keeps the EFFECTIVE batch identical, so the experiment is
unchanged -- it only replaces sequential micro-steps with parallel ones.

Usage: bench_bs_gpu.py --batch_sizes 1,2,4 --steps 6
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

from proteinfoundation.proteinflow.contact2coord_trainer import GRAD_CLIP, ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--batch_sizes", default="1,2,4")
ap.add_argument("--steps", type=int, default=6, help="timed optimizer-free fwd+bwd passes per size")
ap.add_argument("--warmup", type=int, default=2, help="untimed passes first (cuDNN autotune, alloc)")
ap.add_argument("--n_diff", type=int, default=48)
ap.add_argument("--diff_chunk", type=int, default=8)
ap.add_argument("--w_fape", type=float, default=0.0)
args = ap.parse_args()

SIZES = [int(v) for v in args.batch_sizes.split(",")]

name = torch.cuda.get_device_name(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"[gpu] {name}  {total_gb:.1f} GiB", flush=True)

CFG_DIR = os.path.join(REPO, "configs", "datasets_config", "pdb")
assert os.path.isdir(CFG_DIR), f"config dir missing: {CFG_DIR}"

MODEL_CFG["n_diffusion_samples"] = args.n_diff
MODEL_CFG["diff_chunk"] = args.diff_chunk
MODEL_CFG["t_beta"] = (1.3, 2.0)

print(f"\n{'bs':>4} {'status':>10} {'s/pass':>9} {'struct/s':>10} {'peak GiB':>10} {'of total':>9}")
results = []
for bs in SIZES:
    # ⛔ Rebuild the datamodule per size: batch_size is baked into the loader at construction.
    with hydra.initialize_config_dir(CFG_DIR, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.batch_size = bs
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    model = ContactToCoordTrainer(model_cfg=MODEL_CFG, w_fape=args.w_fape,
                                  fape_chunk=8, use_smooth_lddt=False).to("cuda")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)   # tiny lr: we time, we do not train
    torch.cuda.reset_peak_memory_stats()
    it = iter(dm.train_dataloader())

    try:
        for i in range(args.warmup + args.steps):
            if i == args.warmup:
                torch.cuda.synchronize()
                t0 = time.time()
            batch = next(it).to("cuda")
            # ⛔⛔ AUTOCAST IS MANDATORY HERE. Calling _step() directly bypasses Lightning's
            # bf16-mixed plugin, which installs this context around training_step. Without it the
            # trunk runs in fp32 -- a DIFFERENT and much slower model than production. The first
            # version of this benchmark omitted it, reported 7.53 s/pass instead of production's
            # ~4.70, and separately credited TF32 with a +37.5% speedup that is pure fp32-vs-tensor-
            # core and does not exist under bf16 (job 22911040).
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model._step(batch, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.steps
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{bs:>4} {'ok':>10} {dt:>9.2f} {bs/dt:>10.3f} {peak:>10.1f} {100*peak/total_gb:>8.0f}%",
              flush=True)
        results.append((bs, dt, bs / dt, peak))
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:>4} {'OOM':>10} {'-':>9} {'-':>10} {'-':>10} {'-':>9}", flush=True)
        results.append((bs, None, None, None))
    del model, opt, dm, it
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

ok = [r for r in results if r[1] is not None]
if len(ok) > 1:
    b0, _, thr0, _ = ok[0]
    print(f"\nthroughput relative to bs={b0}:")
    for bs, dt, thr, peak in ok:
        print(f"  bs={bs:<3} {thr/thr0:>5.2f}x  ({thr:.3f} structures/s, peak {peak:.1f} GiB)")
    best = max(ok, key=lambda r: r[2])
    print(f"\n⭐ best throughput at bs={best[0]}: {best[2]:.3f} structures/s, "
          f"peak {best[3]:.1f} GiB of {total_gb:.1f}")
    print("⚠️ Keep the EFFECTIVE batch fixed when adopting this: bs x accum must stay at 8.")
