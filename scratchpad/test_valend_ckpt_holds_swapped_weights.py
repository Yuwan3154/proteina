"""Does a ModelCheckpoint that fires at on_validation_end save the EMA-SWAPPED weights?

Lightning 2.5.0 evaluation_loop._on_evaluation_end runs callback hooks BEFORE the LightningModule's
on_validation_end, and contact2coord_trainer swaps EMA in at on_validation_start / restores raw at
on_validation_end. If so, (a) best_cb (no step trigger) writes state_dict == EMA, and (b) last_cb
writes an EMA last.ckpt whenever a validation lands on a step that is not a multiple of its interval.
Toy model: "EMA" = every weight 7.0, so a saved file is swapped iff all its weights equal 7.0.
"""

import glob
import os
import sys
import tempfile

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

SWAP = 7.0


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


def main():
    torch.manual_seed(0)
    d = tempfile.mkdtemp()
    data = TensorDataset(torch.randn(20, 4))
    last_cb = ModelCheckpoint(dirpath=os.path.join(d, "last"), monitor=None, save_top_k=0,
                              save_last=True, every_n_train_steps=3, enable_version_counter=False)
    best_cb = ModelCheckpoint(dirpath=os.path.join(d, "best"), monitor="val/loss", mode="min",
                              save_top_k=-1, save_last=False, enable_version_counter=False)
    # validation at global steps 5 and 10 -- neither a multiple of last_cb's interval (3)
    tr = L.Trainer(accelerator="cpu", max_steps=11, val_check_interval=5, limit_val_batches=2,
                   num_sanity_val_steps=0, callbacks=[last_cb, best_cb], logger=False,
                   enable_progress_bar=False)
    tr.fit(M(), DataLoader(data, batch_size=1), DataLoader(data, batch_size=1))
    print(f"lightning {L.__version__}")
    n_files = 0
    for p in sorted(glob.glob(os.path.join(d, "*", "*.ckpt"))):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        w = ck["state_dict"]["layer.weight"]
        swapped = bool(torch.all(w == SWAP))
        n_files += 1
        print(f"{os.path.relpath(p, d):40s} global_step={ck['global_step']:3d} "
              f"state_dict={'SWAPPED (EMA)' if swapped else 'raw'}")
    # sample-size guard: expect 2 best files (steps 5, 10) + 1 last.ckpt
    print(f"files inspected: {n_files} (expected 3)")
    sys.exit(0 if n_files == 3 else 1)


if __name__ == "__main__":
    main()
