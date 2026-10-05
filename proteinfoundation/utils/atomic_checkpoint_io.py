"""Checkpoint writes that are atomic without relying on $TMPDIR.

Lightning's TorchCheckpointIO writes through an fsspec transaction whose temp file lives in $TMPDIR, so the final
rename is only atomic when $TMPDIR shares the checkpoint's filesystem; that is why the launchers had pinned $TMPDIR
to scratch (07dbb58, 919b65b). But DataLoader workers hand every batch over through AF_UNIX sockets under $TMPDIR,
so each scratch (fstor018) NFS stall froze all batch handoffs until the 300 s DataLoader timeout killed the run
(2026-09-28, 10-02, 10-04; measured 2 socket connects per batch under $TMPDIR). This plugin writes the temp file
next to the checkpoint and os.replace()s it, so $TMPDIR can be node-local.
"""

import os

import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO


class SameDirAtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        assert storage_options is None, "storage_options is not supported"
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"  # not *.ckpt, so ModelCheckpoint never mistakes a partial file for a checkpoint
        with open(tmp, "wb") as fh:
            torch.save(checkpoint, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
