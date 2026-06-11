import gc
from collections import Counter

from lightning.pytorch.callbacks import Callback


def _rss_gb() -> float:
    # Current resident set size of THIS process (Linux /proc), GB. No psutil dep.
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6  # kB -> GB
    return 0.0


class MemTracker(Callback):
    """Debug-only host-RAM leak tracker (enabled by env MEM_DEBUG=1, off otherwise).

    Every ``every_n_steps`` optim steps logs: current main-process RSS, its delta vs
    the first sample, total live gc object count, the object TYPES whose live count
    grew the most since the previous sample (the leak fingerprint), and the lengths of
    known list accumulators on the LightningModule. Tracks the long-lived training
    process where the leak accumulates (workers were ruled out: COW resets per epoch).
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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.every <= 0 or step % self.every != 0:
            return
        gc.collect()
        rss = _rss_gb()
        if self.base is None:
            self.base = rss
        objs = gc.get_objects()
        counts = Counter(type(o).__name__ for o in objs)
        msg = f"[MEM] step={step} rss={rss:.2f}GB d={rss - self.base:+.2f}GB live_objs={len(objs)}"
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
