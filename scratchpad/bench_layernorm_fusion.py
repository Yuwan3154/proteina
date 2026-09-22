"""Is c2c's LayerNorm cost recoverable by fusion?

WHY: the H200 op profile (job 23434465) measured LayerNorm fwd+bwd at **~81% of all CUDA time**
(native_layer_norm_backward 54.72% + native_layer_norm 26.24%), against ~1.2% for attention. So the
only optimisation with real leverage on this model targets LayerNorm. This asks whether a fused
implementation actually delivers, at the shapes the model really uses.

⛔ SHAPES ARE MEASURED, NOT ASSUMED. The earlier profile ran with record_shapes=False, so the
LayerNorm input shapes are unknown. Guessing them would benchmark a different problem: LayerNorm
throughput depends strongly on the normalised width and the row count. Phase 1 below re-profiles
ONE step with record_shapes=True and extracts the actual (shape -> call count) distribution; phase 2
benchmarks only those shapes, weighted by how often they occur.

⛔ AUTOCAST IS MANDATORY, identical to bench_bs_gpu.py -- calling _step() directly bypasses
Lightning's bf16-mixed plugin, and without it the trunk runs fp32, a different and much slower
model. A previous benchmark omitted it and reported 7.53 s/pass against production's ~4.70.

⛔ This does NOT change the training config. It measures a candidate; adopting it would be a
separate, explicit decision.
"""

import argparse
import collections
import os
import time

import hydra
import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "configs", "datasets_config", "pdb")
GRAD_CLIP = 10.0

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--iters", type=int, default=50, help="timed reps per shape")
ap.add_argument("--top_shapes", type=int, default=6)
args = ap.parse_args()

print(f"[gpu] {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}",
      flush=True)
print(f"[torch] {torch.__version__}", flush=True)

# ── PHASE 1: what shapes does LayerNorm actually see? ─────────────────────────────────────────
MODEL_CFG = dict(c_s=384, c_z=128, n_blocks=24, n_heads=12,
                 n_diffusion_samples=48, diff_chunk=8, t_beta=(1.3, 2.0))
with hydra.initialize_config_dir(CFG_DIR, version_base=hydra.__version__):
    cfg = hydra.compose(config_name=args.dataset)
OmegaConf.set_struct(cfg, False)
cfg.datamodule.batch_size = args.bs
cfg.datamodule.num_workers = 0
cfg.datamodule.prefetch_factor = None
dm = hydra.utils.instantiate(cfg.datamodule)
dm.setup("fit")

model = ContactToCoordTrainer(model_cfg=MODEL_CFG, w_fape=0.0,
                              fape_chunk=8, use_smooth_lddt=False).to("cuda")
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
it = iter(dm.train_dataloader())


def one_step():
    batch = next(it).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model._step(batch, True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    opt.zero_grad(set_to_none=True)


one_step()  # warmup
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    one_step()
    torch.cuda.synchronize()

shapes = collections.Counter()
for e in prof.key_averages(group_by_input_shape=True):
    if "native_layer_norm" in e.key and e.input_shapes:
        inp = e.input_shapes[0] if isinstance(e.input_shapes[0], list) else e.input_shapes
        if inp and len(inp) >= 2:
            shapes[tuple(inp)] += e.count
print("\n===== MEASURED LayerNorm input shapes (one step) =====", flush=True)
for s, c in shapes.most_common(args.top_shapes * 2):
    print(f"  {c:7d} calls   {s}", flush=True)
if not shapes:
    raise SystemExit("no LayerNorm shapes captured; cannot benchmark shapes I have not measured")

del model, opt, dm, it
torch.cuda.empty_cache()

# ── PHASE 2: native vs compiled, on the measured shapes, weighted by frequency ────────────────
print(f"\n===== NATIVE vs torch.compile LayerNorm (fwd+bwd, {args.iters} reps) =====", flush=True)
print(f"{'shape':>26} {'calls':>8} {'native ms':>10} {'compiled ms':>12} {'speedup':>8}", flush=True)


def timed(fn, x, w, b, iters):
    for _ in range(5):
        y = fn(x, w, b)
        y.sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        y = fn(x, w, b)
        y.sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def native(x, w, b):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b)


compiled = torch.compile(native, mode="max-autotune-no-cudagraphs")

tot_n = tot_c = 0.0
for s, c in shapes.most_common(args.top_shapes):
    width = s[-1]
    x = torch.randn(*s, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.ones(width, device="cuda", dtype=torch.bfloat16)
    bb = torch.zeros(width, device="cuda", dtype=torch.bfloat16)
    tn = timed(native, x, w, bb, args.iters)
    try:
        tc = timed(compiled, x, w, bb, args.iters)
    except Exception as ex:  # a compile failure is a RESULT, not a crash
        print(f"{str(s):>26} {c:8d} {tn:10.3f} {'COMPILE FAILED':>12} {type(ex).__name__:>8}", flush=True)
        continue
    tot_n += tn * c
    tot_c += tc * c
    print(f"{str(s):>26} {c:8d} {tn:10.3f} {tc:12.3f} {tn/tc:8.2f}x", flush=True)

if tot_c > 0:
    print(f"\n  CALL-WEIGHTED total over the top {args.top_shapes} shapes: "
          f"native {tot_n:.0f} ms vs compiled {tot_c:.0f} ms  =>  {tot_n/tot_c:.2f}x", flush=True)
    print("  ⚠️ This is the LayerNorm-only speedup. LayerNorm is ~81% of step time, so the "
          "END-TO-END gain is bounded by Amdahl: 1/(0.19 + 0.81/s).", flush=True)
    s_ln = tot_n / tot_c
    print(f"  => predicted end-to-end speedup ceiling: {1/(0.19 + 0.81/s_ln):.2f}x", flush=True)
print("LNBENCH_EXIT=0", flush=True)
