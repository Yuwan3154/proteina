import time

from lightning.pytorch.callbacks import Callback


class DataloaderTimer(Callback):
    """Logs dataloader-wait vs compute time per training micro-batch (rank 0 only).

    `on_train_batch_start` fires right after the dataloader yields a batch, so the
    gap from the previous `on_train_batch_end` is the time the training loop spent
    WAITING for the next batch (~0 if prefetch/workers keep up). `end - start` is the
    compute (forward+backward, plus optimizer on accumulation boundaries). A high
    `wait_frac` => the GPU is starved by the input pipeline (add workers / prefetch);
    `wait_frac ~ 0` => compute/sync-bound (more workers won't help).

    The first logged window includes torch.compile warmup, so read steady-state from
    the second window onward. Enable with env DIAG_DATALOADER=1.
    """

    def __init__(self, every_n_steps: int = 100):
        super().__init__()
        self.every_n_steps = every_n_steps
        self._last_end = None
        self._t0 = None
        self._waits = []
        self._computes = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._last_end is not None:
            self._waits.append(now - self._last_end)
        self._t0 = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._t0 is not None:
            self._computes.append(now - self._t0)
        self._last_end = now
        if len(self._computes) >= self.every_n_steps:
            n = len(self._computes)
            w = sum(self._waits) / max(len(self._waits), 1)
            c = sum(self._computes) / n
            frac = w / (w + c) if (w + c) > 0 else 0.0
            if trainer.is_global_zero:
                print(
                    f"[DLTIMER] micro_batches={n} dataloader_wait={w * 1000:.0f}ms "
                    f"compute={c * 1000:.0f}ms wait_frac={frac * 100:.1f}%",
                    flush=True,
                )
            self._waits.clear()
            self._computes.clear()
