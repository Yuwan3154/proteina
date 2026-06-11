import gc
import os
from collections import Counter

from lightning.pytorch.callbacks import Callback


def _rss_gb() -> float:
    # Current resident set size of THIS process (Linux /proc), GB. No psutil dep.
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6  # kB -> GB
    return 0.0


def _cgroup_mem_gb() -> float:
    # Total RAM of THIS job's cgroup (main process + ALL dataloader workers) -- the
    # metric SLURM OOM-kills on. cgroup v2: /proc/self/cgroup is "0::/<path>".
    rel = ""
    if os.path.exists("/proc/self/cgroup"):
        with open("/proc/self/cgroup") as f:
            txt = f.read().strip()
        if "::" in txt:
            rel = txt.split("::", 1)[-1].splitlines()[0]
    for cand in (f"/sys/fs/cgroup{rel}/memory.current", "/sys/fs/cgroup/memory.current"):
        if os.path.exists(cand):
            with open(cand) as f:
                return int(f.read().strip()) / 1e9
    return -1.0


class MemTracker(Callback):
    """Debug-only host-RAM leak tracker (enabled by env MEM_DEBUG=1, off otherwise).

    Every ``every_n_steps`` optim steps logs: epoch, main-process RSS (+delta), the
    cgroup TOTAL RAM (+delta; covers all dataloader workers -- the SLURM-OOM metric),
    the gc object TYPES whose live count grew most since the previous sample, and
    lengths of known list accumulators. Tracking the cgroup total across epoch
    boundaries pinpoints a per-epoch worker-recreation leak (main flat, cgroup climbs).
    """

    _ACCUMS = (
        "validation_output_data",
        "_val_traj_buffer",
        "_validation_tmscore_results",
        "_validation_contact_results",
    )

    def __init__(self, every_n_steps: int = 50):
        self.every = int(every_n_steps)
        self.prev = None
        self.base = None
        self.cg_base = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.every <= 0 or step % self.every != 0:
            return
        gc.collect()
        rss = _rss_gb()
        cg = _cgroup_mem_gb()
        if self.base is None:
            self.base = rss
            self.cg_base = cg
        objs = gc.get_objects()
        counts = Counter(type(o).__name__ for o in objs)
        ep = int(getattr(trainer, "current_epoch", -1))
        msg = (
            f"[MEM] ep={ep} step={step} rss={rss:.2f}GB(d{rss - self.base:+.2f}) "
            f"cgroup={cg:.2f}GB(d{cg - self.cg_base:+.2f}) live_objs={len(objs)}"
        )
        if self.prev is not None:
            grow = sorted(((counts[k] - self.prev.get(k, 0), k) for k in counts), reverse=True)[:8]
            msg += " | grow:" + " ".join(f"{k}+{d}" for d, k in grow if d > 0)
        self.prev = counts
        accums = {
            n: len(getattr(pl_module, n))
            for n in self._ACCUMS
            if isinstance(getattr(pl_module, n, None), list)
        }
        if accums:
            msg += " | accums:" + " ".join(f"{n}={v}" for n, v in accums.items())
        print(msg, flush=True)
