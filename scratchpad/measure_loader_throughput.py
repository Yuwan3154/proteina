"""Train-loader throughput (samples/s) vs DataLoader workers, through the real datamodule (CPU only).

Builds the datamodule as train.py composes it for --config_name (epoch_batch_plan OFF: at batch 1 the per-sample work is
the same, and the plan sampler needs a Trainer), then:
  * sweep: for each worker count, forward then reverse order, a fresh loader drawing a DIFFERENT epoch's members
    (ClusterSampler.set_epoch), so no run reads records an earlier run pulled into the page cache; --warmup batches
    skipped, the next --n timed (up to prefetch_factor x workers batches were already queued at t0: <= ~3% at n=1000);
  * node: --procs concurrent processes (forked after setup), each with --proc_workers workers and its own epoch, as
    training runs one loader per rank; the summed samples/s is the node's loader capacity.
Prints the CPU model. Usage (PYTHONPATH=repo): python scratchpad/measure_loader_throughput.py --config_name C
"""

import argparse
import multiprocessing as mp
import subprocess
import time

from scratchpad.test_plan_datamodule_integration import compose


def run_one(dm, nw, epoch, n, warmup, out=None):
    dm.num_workers = nw
    dl = dm.train_dataloader()
    dl.sampler.set_epoch(epoch)
    it = iter(dl)
    for _ in range(warmup):
        next(it)
    t0 = time.time()
    k = 0
    for _ in range(n):
        k += int(next(it)["coord_mask"].shape[0])
    dt = time.time() - t0
    del it, dl
    if out is not None:
        out.put((k, dt))
    return k, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--proc_workers", type=int, default=4)
    args = ap.parse_args()
    cpu = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, capture_output=True, text=True).stdout
    print(cpu.strip(), flush=True)
    _, dm = compose(args.config_name, plan_on=False, num_workers=args.workers[0])
    dm.setup("fit")
    epoch = 100
    for order in ("forward", "reverse"):
        for nw in (args.workers if order == "forward" else list(reversed(args.workers))):
            epoch += 1
            k, dt = run_one(dm, nw, epoch, args.n, args.warmup)
            print(f"[{order}] 1 process x {nw:2d} workers: {k / dt:6.1f} samples/s ({1000 * dt / k:5.1f} ms/sample)",
                  flush=True)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    ps = [ctx.Process(target=run_one, args=(dm, args.proc_workers, 200 + i, args.n, args.warmup, q))
          for i in range(args.procs)]
    t0 = time.time()
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    wall = time.time() - t0
    per = [k / dt for k, dt in res]
    print(f"[node] {args.procs} concurrent processes x {args.proc_workers} workers: "
          f"{sum(per):.1f} samples/s total (upper bound: windows not fully overlapping; per process "
          f"{', '.join(f'{x:.1f}' for x in per)}); lower bound incl. start-up and warm-up "
          f"{sum(k for k, _ in res) / wall:.1f} samples/s over {wall:.0f} s wall",
          flush=True)


if __name__ == "__main__":
    main()
