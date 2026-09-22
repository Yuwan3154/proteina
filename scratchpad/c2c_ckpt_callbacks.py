"""Checkpoint callbacks for train_c2c.py that never store EMA-swapped weights as the raw state_dict.

⛔⛔ WHY. ContactToCoordTrainer swaps EMA into the model at on_validation_start and restores the raw
weights only in its OWN on_validation_end, and Lightning 2.5.0 runs callback hooks BEFORE the
module's (evaluation_loop.py:346-352). So every ModelCheckpoint save made at on_validation_end
writes EMA as `state_dict`. MEASURED in the production env (job 23498231, toy model): the best files
AND last.ckpt written at validation end were all EMA. That is how the 8,076 branch point became an
EMA checkpoint, and a resume from such a last.ckpt loads EMA as the raw weights (user: fix it,
2026-09-22).
⇒ last.ckpt and the ladder save ONLY from on_train_batch_end, where the raw weights are in place.
  best_cb still saves at validation end (that is the only point with a fresh val/loss), so its
  files are EMA by construction and are NAMED so: "ema-epoch=E-step=S.ckpt".
"""

import os

from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class TrainStepCheckpoint(ModelCheckpoint):
    """A ModelCheckpoint that saves only from on_train_batch_end, never at validation end."""

    def on_validation_end(self, trainer, pl_module):
        return


class StepLadder(Callback):
    """Keep checkpoints at fixed optimizer steps: every step in `keep_steps`, plus every
    `keep_every` steps after the largest of them. Raw state_dict (and the trainer's ema) per file.

    ⛔ Step-triggered on purpose (see the module docstring). ⛔ Its own filename prefix, so best_cb
    (which deletes the files it evicts by path) can never touch a ladder file.
    """

    def __init__(self, dirpath, keep_steps=(), keep_every=0):
        self.dirpath = dirpath
        self.keep_steps = sorted({int(s) for s in keep_steps})
        self.keep_every = int(keep_every)
        self.after = self.keep_steps[-1] if self.keep_steps else 0
        self._start = 0
        self._last = None

    def wants(self, step):
        return step in self.keep_steps or (
            self.keep_every > 0 and step > self.after and step % self.keep_every == 0)

    def path(self, step):
        return os.path.join(self.dirpath, f"ladder-step{step:07d}.ckpt")

    def on_train_start(self, trainer, pl_module):
        self._start = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = trainer.global_step
        # global_step repeats across the micro-batches of one accumulation window: save once.
        if s == self._last or s < self._start or not self.wants(s):
            return
        # A resume lands exactly on a saved step; do not rewrite a file that already exists.
        if s == self._start and os.path.exists(self.path(s)):
            return
        self._last = s
        trainer.save_checkpoint(self.path(s))


def build_ckpt_callbacks(dirpath, overfit, ckpt_every, keep_steps=(), keep_every=0):
    """The callback list train_c2c.py trains with (and the unit test exercises)."""
    if overfit:
        # Overfit keeps EVERY checkpoint on purpose (the mirror-rate trajectory), on its step
        # interval, raw -- the pre-d8fd3bb behaviour, which d8fd3bb had silently turned into EMA
        # files by moving them to a validation-end trigger.
        cbs = [TrainStepCheckpoint(
            dirpath=dirpath, monitor=None, save_top_k=-1,
            filename="step{step:07d}", auto_insert_metric_name=False,
            save_last=True, every_n_train_steps=ckpt_every, enable_version_counter=False)]
    else:
        # last_cb: resume anchor; save_top_k=0 so it writes ONLY last.ckpt, every ckpt_every steps.
        # best_cb: genuine top-k on a FRESH val/loss (no step trigger, see train_c2c.py); EMA files.
        cbs = [
            TrainStepCheckpoint(dirpath=dirpath, monitor=None, save_top_k=0, save_last=True,
                                every_n_train_steps=ckpt_every, enable_version_counter=False),
            ModelCheckpoint(dirpath=dirpath, monitor="val/loss", mode="min", save_top_k=3,
                            filename="ema-{epoch}-{step}", save_last=False,
                            enable_version_counter=False),
        ]
    if keep_steps or keep_every:
        cbs.append(StepLadder(dirpath, keep_steps=keep_steps, keep_every=keep_every))
    return cbs
