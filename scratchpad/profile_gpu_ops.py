"""Op-level CUDA profile of one c2c training step, to explain the RTX-vs-H200 gap.

MEASURED so far (clean A/B, jobs 22906126/22906127, both on pi_so3):
  RTX PRO 6000  0.133 struct/s, peak 45.5 GiB, 2430 MHz, OOM at bs=6
  H200 NVL      0.178 struct/s, peak 31.8 GiB, 1785 MHz, fine at bs=6
⇒ The H200 is 1.34x faster -- but it also uses 30% LESS memory for IDENTICAL work. Same model, same
batch, same code. That asymmetry is the clue: if the RTX is falling back to a memory-heavier
attention path (no fused/flash kernel on sm_120 in this build), part of the 1.34x is a SOFTWARE
gap that could be recovered, not a hardware fact.

This prints the self-CUDA-time breakdown by op and the actual KERNEL names, so the two cards can be
compared directly. It answers: are they running the same kernels?

⛔⛔ AUTOCAST IS MANDATORY and is kept identical to bench_bs_gpu.py. Calling `_step()` directly
bypasses Lightning's bf16-mixed plugin; without autocast the trunk runs fp32 -- a different and
much slower model than production. A previous benchmark omitted it and reported 7.53 s/pass against
production's ~4.70, and separately credited TF32 with a +37.5% speedup that does not exist under
bf16 (job 22911040).

⛔ This measures ONE card per run. Compare two runs; do not infer a difference from one.
"""

import argparse
import os

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
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--n_diff", type=int, default=48)
ap.add_argument("--diff_chunk", type=int, default=8)
ap.add_argument("--top", type=int, default=25)
args = ap.parse_args()

MODEL_CFG = dict(
    c_s=384, c_z=128, n_blocks=24, n_heads=12,
    n_diffusion_samples=args.n_diff, diff_chunk=args.diff_chunk, t_beta=(1.3, 2.0),
)

name = torch.cuda.get_device_name(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
cc = torch.cuda.get_device_capability(0)
print(f"[gpu] {name}  {total_gb:.1f} GiB  sm_{cc[0]}{cc[1]}", flush=True)
print(f"[torch] {torch.__version__}  arch_list={torch.cuda.get_arch_list()}", flush=True)
# Which attention backends does this build actually offer on THIS card?
for flag in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled",
             "cudnn_sdp_enabled"):
    fn = getattr(torch.backends.cuda, flag, None)
    if fn is not None:
        print(f"[sdp] {flag}: {fn()}", flush=True)

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
torch.cuda.reset_peak_memory_stats()


def one_step():
    batch = next(it).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model._step(batch, True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    opt.zero_grad(set_to_none=True)


for _ in range(args.warmup):
    one_step()
torch.cuda.synchronize()
warm_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"[mem] peak after warmup: {warm_peak:.1f} GiB of {total_gb:.1f} "
      f"({100*warm_peak/total_gb:.0f}%)", flush=True)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, profile_memory=True) as prof:
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()

print(f"\n===== TOP {args.top} OPS BY SELF CUDA TIME ({args.steps} steps) =====", flush=True)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.top), flush=True)

print(f"\n===== TOP {args.top} BY SELF CUDA MEMORY =====", flush=True)
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=args.top), flush=True)

# ⭐ The question is whether the two cards run the SAME kernels. Surface attention/matmul
# kernel names explicitly rather than leaving them buried in the table.
print("\n===== ATTENTION / MATMUL KERNELS ACTUALLY DISPATCHED =====", flush=True)
seen = {}
for e in prof.key_averages():
    n = e.key.lower()
    if any(k in n for k in ("attention", "sdpa", "flash", "gemm", "matmul", "bmm", "addmm",
                            "efficient", "cutlass", "triton")):
        seen[e.key] = (e.self_cuda_time_total, getattr(e, "self_cuda_memory_usage", 0))
for k, (t, m) in sorted(seen.items(), key=lambda kv: -kv[1][0])[:args.top]:
    print(f"  {t/1000:10.1f} ms  {m/1024**2:9.1f} MiB  {k}", flush=True)

print(f"\n[mem] final peak: {torch.cuda.max_memory_allocated()/1024**3:.1f} GiB", flush=True)
print("PROFILE_EXIT=0", flush=True)
