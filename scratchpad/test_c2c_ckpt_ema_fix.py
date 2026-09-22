"""The c2c checkpoint callbacks must never store EMA-swapped weights where raw weights are meant.

Toy model with the trainer's exact swap pattern: every weight becomes 7.0 at on_validation_start and
is restored in the module's own on_validation_end (which Lightning runs AFTER the callbacks). A saved
file is "SWAPPED" iff its weights are all 7.0. Validations land on steps that are NOT multiples of
the last.ckpt interval -- the case that wrote an EMA last.ckpt before the fix (job 23498231).
Built through the SAME build_ckpt_callbacks() train_c2c.py calls, with gradient accumulation on.

Run: python scratchpad/test_c2c_ckpt_ema_fix.py
"""

import glob
import os
import sys
import tempfile

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2c_ckpt_callbacks import build_ckpt_callbacks  # noqa: E402

SWAP = 7.0
ACCUM = 2
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


class M(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self._cached = None

    def training_step(self, b, i):
        return self.layer(b[0]).pow(2).mean()

    def validation_step(self, b, i):
        self.log("val/loss", self.layer(b[0]).pow(2).mean())

    def on_validation_start(self):
        self._cached = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self.load_state_dict({k: torch.full_like(v, SWAP) for k, v in self._cached.items()})

    def on_validation_end(self):
        self.load_state_dict(self._cached)
        self._cached = None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def fit(d, cbs, max_steps, val_every_steps, ckpt=None):
    data = TensorDataset(torch.randn(200, 4))
    tr = L.Trainer(accelerator="cpu", max_steps=max_steps, accumulate_grad_batches=ACCUM,
                   val_check_interval=val_every_steps * ACCUM, limit_val_batches=2,
                   num_sanity_val_steps=0, callbacks=cbs, logger=False,
                   enable_progress_bar=False, enable_model_summary=False)
    tr.fit(M(), DataLoader(data, batch_size=1), DataLoader(data, batch_size=1), ckpt_path=ckpt)


def read(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return int(ck["global_step"]), bool(torch.all(ck["state_dict"]["layer.weight"] == SWAP))


torch.manual_seed(0)
print(f"lightning {L.__version__}")

# ── 1. production: last.ckpt every 3 steps, validation at steps 5 and 10, ladder {4, 7} + every 4 ──
print("\n== 1. production callbacks ==")
d = tempfile.mkdtemp()
fit(d, build_ckpt_callbacks(d, overfit=False, ckpt_every=3, keep_steps=(4, 7), keep_every=4),
    max_steps=11, val_every_steps=5)
gs, sw = read(os.path.join(d, "last.ckpt"))
check("last.ckpt holds RAW weights", not sw, f"global_step={gs}")
check("last.ckpt is the last train-step save (step 9), not the step-10 validation", gs == 9, f"{gs}")
best = sorted(glob.glob(os.path.join(d, "ema-*.ckpt")))
check("best files carry the ema- label", len(best) == 2, f"{[os.path.basename(p) for p in best]}")
check("best files are EMA, as labelled", all(read(p)[1] for p in best))
lad = sorted(glob.glob(os.path.join(d, "ladder-step*.ckpt")))
steps = [read(p)[0] for p in lad]
check("ladder = {4, 7} + every 4 after 7 -> [4, 7, 8]", steps == [4, 7, 8], f"{steps}")
check("ladder files are RAW", not any(read(p)[1] for p in lad))
check("ladder filename step == stored global_step",
      all(int(os.path.basename(p)[len("ladder-step"):-5]) == read(p)[0] for p in lad))
check("no stray epoch=*.ckpt files", not glob.glob(os.path.join(d, "epoch=*.ckpt")))

# ── 2. resume: first segment to step 6, second to 11; ladder {4, 7}, no interval ──
print("\n== 2. resume across a segment ==")
d = tempfile.mkdtemp()
fit(d, build_ckpt_callbacks(d, overfit=False, ckpt_every=3, keep_steps=(4, 7)), max_steps=6,
    val_every_steps=5)
m4 = os.path.getmtime(os.path.join(d, "ladder-step0000004.ckpt"))
fit(d, build_ckpt_callbacks(d, overfit=False, ckpt_every=3, keep_steps=(4, 7)), max_steps=11,
    val_every_steps=5, ckpt=os.path.join(d, "last.ckpt"))
steps = sorted(read(p)[0] for p in glob.glob(os.path.join(d, "ladder-step*.ckpt")))
check("second segment adds step 7 and keeps step 4", steps == [4, 7], f"{steps}")
check("step-4 file not rewritten on resume",
      os.path.getmtime(os.path.join(d, "ladder-step0000004.ckpt")) == m4)
check("last.ckpt RAW after the resumed segment", not read(os.path.join(d, "last.ckpt"))[1])

# ── 3. overfit: keep-all on the step interval, raw ──
print("\n== 3. overfit callbacks ==")
d = tempfile.mkdtemp()
fit(d, build_ckpt_callbacks(d, overfit=True, ckpt_every=5), max_steps=11, val_every_steps=5)
kept = sorted(glob.glob(os.path.join(d, "step*.ckpt")))
check("overfit keeps step 5 and step 10", [read(p)[0] for p in kept] == [5, 10],
      f"{[os.path.basename(p) for p in kept]}")
check("overfit keep-all files are RAW (they were EMA after d8fd3bb)",
      not any(read(p)[1] for p in kept))
check("overfit last.ckpt RAW", not read(os.path.join(d, "last.ckpt"))[1])

print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
