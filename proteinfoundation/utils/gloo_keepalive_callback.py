import time

import torch.distributed as dist
from lightning.pytorch.callbacks import Callback


class GlooObjectCollectiveKeepaliveCallback(Callback):
    """Periodically exercises gloo's object-collective channel to keep it from being
    dropped by network infrastructure during long idle gaps.

    Root cause (found 2026-07-05, SuperCloud multi-node run): gloo's tensor all-reduce
    (used every training step) and its object-collective path (`broadcast_object_list`,
    used only rarely -- e.g. once per validation epoch by ModelCheckpoint's cross-rank
    file-existence check) use separate underlying connections. The tensor-collective
    connection stays constantly warm from per-step gradient sync; the object-collective
    connection sits idle between uses (~15-20 min at this project's validation cadence)
    and gets silently dropped -- confirmed NOT an OS-level TCP keepalive issue (this
    cluster's tcp_keepalive_time is 7200s, far longer than the observed failure window),
    so it's almost certainly a network-infrastructure idle-connection policy in between.
    A trivial periodic broadcast_object_list keeps that channel exercised often enough
    to never go idle past whatever the network's reap threshold actually is.
    """

    def __init__(self, every_n_seconds: float = 120.0):
        super().__init__()
        self.every_n_seconds = float(every_n_seconds)
        self._last_keepalive = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not (dist.is_available() and dist.is_initialized()):
            return
        now = time.time()
        if self._last_keepalive is None:
            self._last_keepalive = now
            return
        if now - self._last_keepalive < self.every_n_seconds:
            return
        self._last_keepalive = now
        dist.broadcast_object_list([None], src=0)
