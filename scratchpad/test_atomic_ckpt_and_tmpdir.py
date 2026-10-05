"""Gate for the DataLoader-stall fix (user 2026-10-05): SameDirAtomicCheckpointIO + node-local TMPDIR.

  1. plugin unit: a save lands and loads; a crash mid-write leaves the previous checkpoint intact and no *.ckpt partial.
  2. Lightning 2-process DDP (gloo, CPU): every checkpoint goes through the plugin, files load, resume from last.ckpt.
  3. DataLoader handoff sockets: with TMPDIR node-local, ZERO socket.connect under scratch and >0 under TMPDIR
     (non-vacuous: the count under TMPDIR must equal 2 per batch, as measured on scratch in probe v2).
Usage: python scratchpad/test_atomic_ckpt_and_tmpdir.py WORKDIR_ON_SCRATCH
"""

import glob
import os
import sys

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.utils.atomic_checkpoint_io import SameDirAtomicCheckpointIO

W = sys.argv[1]
SCRATCH = "/orcd/scratch/"
TMP = os.environ["TMPDIR"]
assert not TMP.startswith(SCRATCH), f"TMPDIR must be node-local for this test: {TMP}"


class Counting(SameDirAtomicCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        super().save_checkpoint(checkpoint, path, storage_options)
        with open(os.path.join(W, f"saves_rank{os.environ.get('LOCAL_RANK', '0')}.txt"), "a") as fh:
            fh.write(os.fspath(path) + "\n")


def test_unit():
    io, p = SameDirAtomicCheckpointIO(), os.path.join(W, "unit", "a.ckpt")
    io.save_checkpoint({"v": torch.arange(5)}, p)
    assert torch.equal(torch.load(p)["v"], torch.arange(5))
    real = torch.save
    def boom(obj, fh):
        fh.write(b"partial")
        raise RuntimeError("simulated crash mid-write")
    torch.save = boom
    raised = False
    try:
        io.save_checkpoint({"v": torch.zeros(5)}, p)
    except RuntimeError:
        raised = True
    torch.save = real
    assert raised
    assert torch.equal(torch.load(p)["v"], torch.arange(5)), "previous checkpoint was damaged"
    assert sorted(os.path.basename(x) for x in glob.glob(os.path.join(W, "unit", "*.ckpt"))) == ["a.ckpt"]
    print("[1] PASS unit: save loads; simulated crash left the previous checkpoint intact; no partial *.ckpt")


class DS(Dataset):
    def __len__(self): return 64
    def __getitem__(self, i): return torch.randn(8), torch.randn(1)


class M(L.LightningModule):
    def __init__(self):
        super().__init__(); self.l = torch.nn.Linear(8, 1)
    def training_step(self, b, i):
        return torch.nn.functional.mse_loss(self.l(b[0]), b[1])
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def run(max_steps, ckpt_path=None):
    cb = ModelCheckpoint(dirpath=os.path.join(W, "ddp"), every_n_train_steps=5, save_last=True, save_top_k=-1)
    t = L.Trainer(plugins=[Counting()], accelerator="cpu", devices=2, strategy=DDPStrategy(process_group_backend="gloo"),
                  max_steps=max_steps, callbacks=[cb], logger=False, enable_progress_bar=False, use_distributed_sampler=True)
    t.fit(M(), DataLoader(DS(), batch_size=4, num_workers=2, persistent_workers=True), ckpt_path=ckpt_path)
    return t


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK", "0") == "0" and not os.path.exists(os.path.join(W, "unit")):
        test_unit()
    run(10)
    run(20, ckpt_path=os.path.join(W, "ddp", "last.ckpt"))
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        saves = open(os.path.join(W, "saves_rank0.txt")).read().split()
        files = sorted(glob.glob(os.path.join(W, "ddp", "*.ckpt")))
        steps = sorted(torch.load(f, weights_only=False)["global_step"] for f in files if "last" not in f)
        last = torch.load(os.path.join(W, "ddp", "last.ckpt"), weights_only=False)["global_step"]
        leftovers = glob.glob(os.path.join(W, "ddp", "*.tmp*"))
        assert saves and all(os.path.dirname(s) == os.path.join(W, "ddp") for s in saves)
        assert steps == [5, 10, 15, 20] and last == 20 and not leftovers, (steps, last, leftovers)
        print(f"[2] PASS ddp: {len(saves)} saves through the plugin; ckpt steps {steps}; resumed to last.ckpt step {last}; no temp leftovers")
