"""T8 tri (ContactMapTriV2) under torch.compile + DDP, driven the way Lightning + model_trainer_base drive it, to prove
it does not hang or thrash, and to time compilation per shape.

Mirrors production:
  * torch._dynamo.config.optimize_ddp = False and recompile_limit = 32, as train.py sets at import;
  * a wrapper module (the LightningModule's role) is what DDP wraps; its forward = training_step: with probability
    p_sc a self-conditioning forward of the INNER model under no_grad, torch.clear_autocast_cache() (PyTorch #105211),
    then the training forward + loss -- all inside ONE bf16 autocast region; backward outside it;
  * gradient accumulation: forward AND backward of non-final micro-batches under ddp.no_sync();
  * a per-micro-batch all_reduce (the sync_dist=True logs in training_step);
  * validation at step 0 (Lightning's sanity check) and every --val_every steps: an EMA-style weight swap (copy_),
    model.eval(), torch.inference_mode(), autocast; each val forward does SC with p_sc then the main forward; then a
    sampling loop of --sample_steps forwards at batch --sample_bs; an all_reduce; swap back; model.train();
  * batches carry seq_dropped like SequenceDropoutTransform does.
Per-rank random (L, T) puts ranks in different compile buckets at different times (the cross-rank asymmetry that can
hang an allreduce). Hang guard: faulthandler dumps all stacks if a step (or the final barrier) exceeds --stall_s.

Output in --out: {tag}_run{id}_rank{r}.jsonl, one row per timed unit -- kind in {train_step (fwd+bwd of one micro-batch, incl. its
SC), sc, val_sc, val, sample}, L, T, buckets, B, entry ("grad" / "nograd" / "inference"), variant (dtype the model
feeds contact_map_sc to the network: bf16 since the model casts it), first_call for that (entry, Lb, Tb, B, variant),
seconds, peak GB of THAT unit; {tag}_run{id}_rank{r}_stats.json with Dynamo cache entries per code object, graph breaks, and the
expected entry counts. Each rank uses its own inductor cache dir (TORCHINDUCTOR_CACHE_DIR/rank{r}) so first calls are
cold per rank; rerun with the same --out to measure warm-cache (requeue) times.
Usage: torchrun --nproc_per_node=N scratchpad/t8_compile_ddp_stress.py --out DIR [--steps 300] [--compile True]
(--compile False = the eager baseline, with the same bucket padding; --sc defaults to --compile)
"""

import argparse
import contextlib
import datetime
import faulthandler
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch._dynamo  # noqa: E402

torch._dynamo.config.optimize_ddp = False  # as train.py
torch._dynamo.config.recompile_limit = 32  # as train.py (configure_compile raises it further)

from proteinfoundation.nn.contact_map_tri_v2 import ContactMapTriV2  # noqa: E402

# ContactMapTriV2.forward casts contact_map_sc to bf16 (user 2026-10-11), so every call is one variant per bucket
SC_DTYPE = "torch.bfloat16"
LEN_EDGES = [64, 128, 192, 256, 320, 384, 448, 512]
TOPO_EDGES = [16, 32, 48, 64, 80, 96]
CFG = dict(pair_dim=128, tri_hidden=128, n_blocks_ref=4, n_blocks_query=4, transition_hidden=512, dim_cond=128,
           max_topology_he_len=96, max_rel_pos=32, topology_vocab_size=45, n_elem_features=1, n_residue_types=22,
           pair_ref_features="both", align_head={"enabled": True}, mlm_head={"enabled": True},
           pad_len_buckets=LEN_EDGES, pad_topo_buckets=TOPO_EDGES)


def make_batch(B, L, T, dev, g):
    cm = (torch.rand(B, L, L, generator=g) > 0.9).float()
    b = {
        "contact_map_t": 0.5 * (cm + cm.transpose(1, 2)),
        "contact_map_sc": torch.zeros(B, L, L),
        "mask": torch.ones(B, L),
        "residue_type": torch.randint(0, 20, (B, L), generator=g),
        "seq_dropped": torch.randint(0, 2, (B,), generator=g),
        "t": torch.rand(B, generator=g),
        "topology_he_tokens": torch.randint(2, 45, (B, T), generator=g),
        "topology_he_pos_raw": torch.sort(torch.rand(B, T, generator=g) * L, dim=1).values,
        "topology_he_feat": torch.randn(B, T, T, 8, generator=g),
        "topology_he_elem_feat": torch.randn(B, T, 1, generator=g),
    }
    return {k: v.to(dev) for k, v in b.items()}


def bucket(x, edges):
    return next((e for e in edges if e >= x), x)


def loss_of(out):
    return sum(v.float().pow(2).mean() for k, v in out.items() if k != "contact_map_pred")


class Step(torch.nn.Module):
    """The LightningModule's role: DDP wraps this; forward = training_step (SC under no_grad, then the training forward)."""

    def __init__(self, model, timer):
        super().__init__()
        self.model, self.timer = model, timer

    def forward(self, b, do_sc, L, T):
        if do_sc:
            with torch.no_grad():
                sc = self.timer("sc", "nograd", lambda: self.model(b), L, T, 1, SC_DTYPE,
                                sync_only=True)
            torch.clear_autocast_cache()
            b = {**b, "contact_map_sc": sc["contact_map_pred"].detach()}
        return loss_of(self.model(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--compile", type=lambda s: s == "True", default=True, help="use_torch_compile")
    ap.add_argument("--sc", type=lambda s: s == "True", default=None, help="use_torch_compile_sc (default = --compile)")
    ap.add_argument("--p_sc", type=float, default=0.5)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--n_val", type=int, default=4)
    ap.add_argument("--sample_bs", type=int, default=4)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--stall_s", type=int, default=1800)
    ap.add_argument("--run_id", type=int, default=0, help="0 = first (cold-cache) run on this --out; 1+ = reruns (warm)")
    ap.add_argument("--pg_timeout_s", type=int, default=3600)
    args = ap.parse_args()
    sc_compile = args.compile if args.sc is None else args.sc

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    n_gpu = torch.cuda.device_count()
    dev = torch.device("cuda", local % n_gpu)
    torch.cuda.set_device(dev)
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], f"rank{rank}")
    if os.environ.get("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["TRITON_CACHE_DIR"], f"rank{rank}")
    backend = "nccl" if n_gpu >= world else "gloo"  # gloo lets 2 ranks share one GPU for a hang test
    dist.init_process_group(backend, timeout=datetime.timedelta(seconds=args.pg_timeout_s))
    os.makedirs(args.out, exist_ok=True)
    tag = "compiled" if args.compile else "eager"
    log = open(os.path.join(args.out, f"{tag}_run{args.run_id}_rank{rank}.jsonl"), "w")
    print(f"[rank {rank}] backend {backend}, {torch.cuda.get_device_name(dev)}, world {world}, compile {args.compile}, "
          f"compile_sc {sc_compile}, cpus {len(os.sched_getaffinity(0))}", flush=True)

    torch.manual_seed(0)
    model = ContactMapTriV2(**{**CFG, "use_torch_compile": args.compile, "use_torch_compile_sc": sc_compile}).to(dev)
    seen = set()

    def timer(kind, entry, fn, L, T, B, variant, sync_only=False):
        # variant = dtype of contact_map_sc as the network sees it (a dynamic=False guard; bf16 since the model casts
        # it), so each (entry, bucket) compiles once
        key = (entry, bucket(L, LEN_EDGES), bucket(T, TOPO_EDGES), B, variant)
        torch.cuda.synchronize(dev)
        if not sync_only:
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize(dev)
        dt = time.time() - t0
        first = key not in seen
        seen.add(key)
        log.write(json.dumps({"rank": rank, "kind": kind, "entry": entry, "L": L, "T": T, "Lb": key[1], "Tb": key[2],
                              "B": B, "variant": variant, "run": args.run_id, "s": round(dt, 4), "first_call": first,
                              "peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)}) + "\n")
        log.flush()
        return out

    step_mod = Step(model, timer)
    ddp = DDP(step_mod, device_ids=[dev.index] if backend == "nccl" else None, find_unused_parameters=False,
              gradient_as_bucket_view=True, static_graph=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = [p.detach().clone() for p in model.parameters()]
    g = torch.Generator().manual_seed(1000 + rank)
    rng = random.Random(1000 + rank)

    def validate():
        live = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p, e in zip(model.parameters(), ema):
                p.copy_(e)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            acc = torch.zeros((), device=dev)
            for _ in range(args.n_val):
                L, T = rng.randint(20, args.max_len), rng.randint(1, 96)
                vb = make_batch(1, L, T, dev, g)
                if rng.random() < args.p_sc:
                    sc = timer("val_sc", "inference", lambda: model(vb), L, T, 1, SC_DTYPE)
                    vb = {**vb, "contact_map_sc": sc["contact_map_pred"]}
                acc = acc + loss_of(timer("val", "inference", lambda: model(vb), L, T, 1,
                                          SC_DTYPE))
            L, T = rng.randint(20, args.max_len), rng.randint(1, 96)
            sb = make_batch(args.sample_bs, L, T, dev, g)
            for _ in range(args.sample_steps):
                o = timer("sample", "inference", lambda: model(sb), L, T, args.sample_bs, SC_DTYPE)
                sb = {**sb, "contact_map_t": o["contact_map_pred"].float(), "contact_map_sc": o["contact_map_pred"]}
            dist.all_reduce(acc)
        torch.clear_autocast_cache()
        model.train()
        with torch.no_grad():
            for p, l in zip(model.parameters(), live):
                p.copy_(l)

    t_start = time.time()
    faulthandler.dump_traceback_later(args.stall_s, repeat=True, file=sys.stderr)
    validate()  # Lightning's sanity check, before step 0
    for step in range(args.steps):
        faulthandler.dump_traceback_later(args.stall_s, repeat=True, file=sys.stderr)
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            L, T = rng.randint(20, args.max_len), rng.randint(1, 96)
            b = make_batch(1, L, T, dev, g)
            do_sc = rng.random() < args.p_sc
            ctx = contextlib.nullcontext() if micro == args.accum - 1 else ddp.no_sync()

            def micro_step():
                with ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = ddp(b, do_sc, L, T) / args.accum
                    loss.backward()
                return loss

            loss = timer("train_step", "grad", micro_step, L, T, 1, SC_DTYPE)
            red = loss.detach().clone()
            dist.all_reduce(red)  # the sync_dist=True logs of training_step
        opt.step()
        with torch.no_grad():
            for p, e in zip(model.parameters(), ema):
                e.mul_(0.999).add_(p.detach(), alpha=0.001)
        if (step + 1) % args.val_every == 0:
            validate()
        if rank == 0 and (step + 1) % 10 == 0:
            print(f"[rank 0] step {step + 1}/{args.steps} {time.time() - t_start:.0f}s", flush=True)
    faulthandler.dump_traceback_later(args.stall_s, repeat=False, file=sys.stderr)
    dist.barrier()
    faulthandler.cancel_dump_traceback_later()

    get_entries = getattr(torch._C._dynamo.eval_frame, "_debug_get_cache_entry_list", None)
    entries = "UNAVAILABLE (torch._C._dynamo.eval_frame._debug_get_cache_entry_list missing)"
    if get_entries is not None:
        entries = {}
        for name in ("_forward_impl", "_forward_impl_sc"):
            entries[name] = len(get_entries(getattr(type(model), name).__code__))
    c = torch._dynamo.utils.counters
    # compare _forward_impl with "grad", _forward_impl_sc with nograd + inference (fewer = shared guards, more = thrash)
    exp = {e: len({k for k in seen if k[0] == e}) for e in ("grad", "nograd", "inference")}
    stats = {"rank": rank, "tag": tag, "steps": args.steps, "wall_s": round(time.time() - t_start, 1),
             "cache_entries": entries, "expected_shapes": exp,
             "graph_breaks": {str(k): int(v) for k, v in c["graph_break"].items()},
             "unique_graphs": int(c["stats"]["unique_graphs"])}
    with open(os.path.join(args.out, f"{tag}_run{args.run_id}_rank{rank}_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"[rank {rank}] DONE {json.dumps(stats)}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
