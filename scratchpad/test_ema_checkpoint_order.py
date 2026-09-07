"""Does ModelCheckpoint serialise the EMA weights as if they were the raw model?

The trainer swaps EMA weights in at `on_validation_start` and restores at `on_validation_end`.
Lightning calls the CALLBACK hooks (ModelCheckpoint.on_validation_end, which writes the file) BEFORE
the LightningModule's own on_validation_end. If that ordering holds, every checkpoint's
`state_dict` is the EMA weights rather than the raw ones -- and since the chain resumes with
`ckpt_path=`, each handoff would silently reload averaged weights as the live model and the two
would converge together instead of the EMA trailing.

⛔ Verified by ORDERING, not by reading Lightning's source: a probe callback records what the model
actually held at the moment ModelCheckpoint would have written.
"""

import os
import sys
import tempfile

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


class Tiny(L.LightningModule):
    """Mirrors the c2c trainer's EMA hook structure exactly."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self._ema = {"weight": torch.full((1, 2), 99.0)}     # unmistakably distinct
        self._cached = None

    def training_step(self, b, i):
        return self.model(b[0]).sum() * 0.0 + self.model.weight.sum() * 0.0 + torch.tensor(0.0, requires_grad=True)

    def validation_step(self, b, i):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def on_validation_start(self):
        if self._ema is None or self._cached is not None:
            return
        self._cached = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict({k: v.to(self._cached[k].dtype) for k, v in self._ema.items()})

    def on_validation_end(self):
        if self._cached is not None:
            self.model.load_state_dict(self._cached)
            self._cached = None


class Probe(Callback):
    """Records the live weight at the moment a checkpoint callback would write."""

    def __init__(self):
        self.at_callback_time = None

    def on_validation_end(self, trainer, pl_module):
        self.at_callback_time = float(pl_module.model.weight.flatten()[0])


ds = TensorDataset(torch.randn(4, 2))
probe = Probe()
with tempfile.TemporaryDirectory() as d:
    ck = ModelCheckpoint(dirpath=d, save_last=True, every_n_epochs=1)
    tr = L.Trainer(max_epochs=1, callbacks=[probe, ck], logger=False, enable_progress_bar=False,
                   enable_model_summary=False, num_sanity_val_steps=0, accelerator="cpu")
    m = Tiny()
    tr.fit(m, DataLoader(ds, batch_size=2), DataLoader(ds, batch_size=2))

    print("\n=== what the model held when the checkpoint callback ran ===")
    print(f"  raw weight = 1.0, EMA weight = 99.0")
    print(f"  observed at callback time: {probe.at_callback_time}")
    leaked = probe.at_callback_time is not None and abs(probe.at_callback_time - 99.0) < 1e-6
    # ⛔ INFORMATIONAL, not a gate. Callbacks DO observe the EMA weights during validation -- the
    # ordering hazard is real -- but Lightning collects the state_dict for the checkpoint at a
    # different point, so the WRITTEN file is unaffected. Measured, not assumed. The gate below is
    # the invariant that actually matters; asserting on the intermediate would be a false alarm.
    print(f"  (callbacks observe EMA weights: {leaked} -- a latent hazard for any callback that")
    print("   reads weights at validation end, but NOT a checkpoint-corruption bug)")

    last = os.path.join(d, "last.ckpt")
    if os.path.exists(last):
        sd = torch.load(last, map_location="cpu", weights_only=False)["state_dict"]
        w = float(list(sd.values())[0].flatten()[0])
        print(f"  state_dict actually written: {w}")
        check("the WRITTEN state_dict holds the raw weights, not the EMA",
              abs(w - 1.0) < 1e-6, f"got {w}")

    print(f"\n  after validation, live model = {float(m.model.weight.flatten()[0])} (must be 1.0)")
    check("EMA is restored after validation", abs(float(m.model.weight.flatten()[0]) - 1.0) < 1e-6)

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    print("\n⛔ A FAILURE HERE means the written state_dict holds EMA weights, so every chained")
    print("   resume via ckpt_path reloads averaged weights as the live model. Fix: restore in")
    print("   on_validation_epoch_end, which runs BEFORE the callback hooks (OpenFold3 does this).")
sys.exit(1 if FAIL else 0)
