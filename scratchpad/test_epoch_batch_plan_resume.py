"""Mid-epoch resume under Lightning (CPU, toy model). Run with the training env's python (Lightning 2.5.0).

Reference run C trains 2 epochs uninterrupted and records the dataset indices of every trained batch. Run A is the
same run killed right after the step-K checkpoint, K = 6 (mid epoch 0) and K = 12 (epoch 0's last step), accumulate
2; run B resumes from that checkpoint. Also on 2 CPU DDP processes (gloo), where training is stopped by max_steps at
the same checkpoint (the saved state is the same) and each rank's sequence is compared.
For each sampler, A(up to the checkpoint) + B must reproduce C's batch sequence exactly.
  baseline: a seeded per-epoch batch sampler with no checkpoint state (what ClusterSampler is today);
  plan:     EpochBatchPlanSampler, its plan + consumed count saved through the datamodule's state_dict.
Also prints Lightning's batch_progress counters at save time (which one equals the batches already trained).
Run: python scratchpad/test_epoch_batch_plan_resume.py; PLAN_TEST_WORLD=2 python scratchpad/test_epoch_batch_plan_resume.py
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "epoch_batch_plan", os.path.join(HERE, "..", "proteinfoundation", "datasets", "epoch_batch_plan.py"))
ebp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ebp)

BS, ACCUM = 2, 2
N_CLUSTERS = 48 * int(os.environ.get("PLAN_TEST_WORLD", "1"))  # 24 batches = 12 optimizer steps per rank per epoch
CLUSTERS = [[3 * c, 3 * c + 1, 3 * c + 2] for c in range(N_CLUSTERS)]
LENGTHS = [40 + (7 * i) % 200 for i in range(3 * N_CLUSTERS)]
EDGES = [64, 128, 192, 256]


class Idx(Dataset):
    def __len__(self):
        return 3 * N_CLUSTERS

    def __getitem__(self, i):
        return torch.tensor([float(i)])


class SeededOrder(torch.utils.data.Sampler):
    """Baseline: a deterministic (seed, epoch) index order with no state, used as dataloader.sampler (so Lightning
    calls its set_epoch), exactly like ClusterSampler today."""

    def __init__(self, rank, world):
        self.epoch, self.rank, self.world = 0, rank, world

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        return iter([i for b in ebp.build_epoch_plan(CLUSTERS, LENGTHS, BS, EDGES, 0, self.epoch, self.world)[self.rank]
                     for i in b])

    def __len__(self):
        return BS * len(ebp.build_epoch_plan(CLUSTERS, LENGTHS, BS, EDGES, 0, self.epoch, self.world)[self.rank])


class DM(L.LightningDataModule):
    def __init__(self, kind):
        super().__init__()
        self.kind, self.restore, self.sampler = kind, None, None

    def train_dataloader(self):
        rank, world = self.trainer.global_rank, self.trainer.world_size
        if self.kind == "baseline":
            self.sampler = SeededOrder(rank, world)
            return DataLoader(Idx(), sampler=self.sampler, batch_size=BS, num_workers=2, persistent_workers=True)
        tr = self.trainer
        self.sampler = ebp.EpochBatchPlanSampler(
            CLUSTERS, LENGTHS, BS, EDGES, 0, rank, world, restore=self.restore,
            position_fn=lambda: (tr.current_epoch, tr.fit_loop.epoch_loop.batch_progress.current.processed))
        return DataLoader(Idx(), batch_sampler=self.sampler, num_workers=2, persistent_workers=True)

    def state_dict(self):
        if self.kind == "baseline" or self.sampler is None:
            return {}
        bp = self.trainer.fit_loop.epoch_loop.batch_progress.current
        st = self.sampler.state_dict(consumed=bp.processed)
        assert st["epoch"] == self.trainer.current_epoch, (st["epoch"], self.trainer.current_epoch)
        return {"plan_state": st}

    def load_state_dict(self, sd):
        self.restore = sd.get("plan_state")


class Toy(L.LightningModule):
    def __init__(self, log_path):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.log_path = log_path

    def training_step(self, batch, batch_idx):
        with open(f"{self.log_path}.r{self.global_rank}", "a") as f:
            f.write(json.dumps({"epoch": self.current_epoch, "step": self.global_step, "batch_idx": batch_idx,
                                "ids": batch[:, 0].long().tolist()}) + "\n")
        return self.lin(batch).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class Killed(Exception):
    pass


class Kill(Callback):
    def __init__(self, at):
        self.at = at

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > self.at:
            raise Killed


class Counters(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        c = trainer.fit_loop.epoch_loop.batch_progress.current
        print(f"  [save] global_step={trainer.global_step} ready={c.ready} started={c.started} "
              f"processed={c.processed} completed={c.completed}")


def run(kind, d, log, kill=None, ckpt=None, devices=1):
    kind = "plan" if kind == "legacy" else kind
    cb = [ModelCheckpoint(dirpath=d, every_n_train_steps=3, save_top_k=-1, save_last=False), Counters()]
    max_steps = -1
    if kill is not None and devices == 1:
        cb.append(Kill(kill))
    elif kill is not None:
        max_steps = kill
    tr = L.Trainer(max_epochs=2, max_steps=max_steps, accumulate_grad_batches=ACCUM, callbacks=cb, logger=False,
                   enable_progress_bar=False, enable_model_summary=False, use_distributed_sampler=False,
                   accelerator="cpu", devices=devices, strategy="ddp_spawn" if devices > 1 else "auto",
                   num_sanity_val_steps=0)
    m, dm = Toy(log), DM(kind)
    try:
        tr.fit(m, datamodule=dm, ckpt_path=ckpt)
    except Killed:
        pass


def read(log):
    if not os.path.exists(log):
        return []
    with open(log) as f:
        return [json.loads(x) for x in f]


def main():
    ok = True
    for devices in (int(os.environ.get("PLAN_TEST_WORLD", "1")),):
        for kind in ("baseline", "plan"):
            for kill_at in (6, 12):
                d = tempfile.mkdtemp()
                run(kind, os.path.join(d, "c"), os.path.join(d, "c.jsonl"), devices=devices)
                run(kind, os.path.join(d, "a"), os.path.join(d, "a.jsonl"), kill=kill_at, devices=devices)
                ck = [f for f in os.listdir(os.path.join(d, "a")) if f"step={kill_at}" in f]
                assert len(ck) == 1, os.listdir(os.path.join(d, "a"))
                run(kind, os.path.join(d, "b"), os.path.join(d, "b.jsonl"), ckpt=os.path.join(d, "a", ck[0]),
                    devices=devices)
                same_all = True
                for r in range(devices):
                    C = [(x["epoch"], tuple(x["ids"])) for x in read(os.path.join(d, f"c.jsonl.r{r}"))]
                    A = [(x["epoch"], tuple(x["ids"])) for x in read(os.path.join(d, f"a.jsonl.r{r}"))
                         if x["step"] < kill_at]
                    B = [(x["epoch"], tuple(x["ids"])) for x in read(os.path.join(d, f"b.jsonl.r{r}"))]
                    got = A + B
                    for ep in (0, 1):
                        seen = [ids for e, ids in got if e == ep]
                        ref = [ids for e, ids in C if e == ep]
                        print(f"{kind} gpus={devices} rank {r} kill@{kill_at} epoch {ep}: reference {len(ref)} batches;"
                              f" with resume {len(seen)}, repeated {len(seen) - len(set(seen))}, never trained "
                              f"{len(set(ref) - set(seen))}")
                    ep_differ = [ids for e, ids in C if e == 0] != [ids for e, ids in C if e == 1]
                    same_all = same_all and got == C and len(C) > 0 and ep_differ
                print(f"{kind} gpus={devices} kill@{kill_at}: every rank's 2-epoch batch sequence identical to the "
                      f"uninterrupted run: {same_all}")
                if kind == "plan":
                    ok = ok and same_all
                shutil.rmtree(d)
        # legacy: a checkpoint written by the stateless sampler (no plan saved) resumed onto the plan, then killed
        # and resumed again mid-epoch 1; the plan's epoch-0 remainder is batches k..end, and the second resume exact
        d = tempfile.mkdtemp()
        run("baseline", os.path.join(d, "a"), os.path.join(d, "a.jsonl"), kill=6, devices=devices)
        ck_a = os.path.join(d, "a", [f for f in os.listdir(os.path.join(d, "a")) if "step=6" in f][0])
        run("legacy", os.path.join(d, "b"), os.path.join(d, "b.jsonl"), ckpt=ck_a, devices=devices)
        run("legacy", os.path.join(d, "b2"), os.path.join(d, "b2.jsonl"), ckpt=ck_a, kill=18, devices=devices)
        ck_b = os.path.join(d, "b2", [f for f in os.listdir(os.path.join(d, "b2")) if "step=18" in f][0])
        run("legacy", os.path.join(d, "c"), os.path.join(d, "c.jsonl"), ckpt=ck_b, devices=devices)
        legacy_ok = True
        for r in range(devices):
            B = [(x["epoch"], tuple(x["ids"])) for x in read(os.path.join(d, f"b.jsonl.r{r}"))]
            B2 = [(x["epoch"], tuple(x["ids"])) for x in read(os.path.join(d, f"b2.jsonl.r{r}")) if x["step"] < 18]
            C = [(x["epoch"], tuple(x["ids"])) for x in read(os.path.join(d, f"c.jsonl.r{r}"))]
            ep0 = [ids for e, ids in B if e == 0]
            want0 = [tuple(b) for b in ebp.build_epoch_plan(CLUSTERS, LENGTHS, BS, EDGES, 0, 0, devices)[r][12:]]
            r_ok = ep0 == want0 and B2 + C == B and len(B) > 0
            print(f"legacy gpus={devices} rank {r}: epoch-0 remainder = plan batches 12..23: {ep0 == want0}; "
                  f"kill@18 + resume reproduces the uninterrupted legacy run: {B2 + C == B}")
            legacy_ok = legacy_ok and r_ok
        ok = ok and legacy_ok
        shutil.rmtree(d)
    print("PLAN RESUME EXACT" if ok else "PLAN RESUME NOT EXACT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
