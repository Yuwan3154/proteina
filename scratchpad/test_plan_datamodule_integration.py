"""Epoch batch plan through the REAL datamodule (real data, DataLoader workers), CPU, with a dummy model.

Builds the datamodule exactly as train.py composes it for --config_name, then runs Lightning (1 process, CPU) with a
model that only records what it is fed. Phases (each a fresh Trainer, as a resumed SLURM segment would be):
  legacy : epoch_batch_plan forced OFF (today's ClusterSampler), killed after the step-S checkpoint in epoch 0;
  plan_a : epoch_batch_plan ON, resumed from the legacy checkpoint (no plan saved), killed after step 2S;
  plan_b : resumed from plan_a's step-2S checkpoint, run to the end of epoch 1.
Checks: served ids are the planned ids (no retry substitution), served residue counts equal processed_length, the
legacy -> plan transition trains epoch-0 plan batches k..end, plan_a + plan_b reproduce the saved epoch-0 plan with no
repeat or gap, epoch 1 is a full fresh plan (every cluster once, different from epoch 0), and the plan saved in each
checkpoint has the trainer's epoch.
Usage (PYTHONPATH=repo): python scratchpad/test_plan_datamodule_integration.py --config_name C --accum A --workdir DIR [--steps S]
"""

import argparse
import json
import os
import sys

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import OmegaConf

from proteinfoundation.datasets import epoch_batch_plan as ebp


class Killed(Exception):
    pass


class Kill(Callback):
    def __init__(self, at):
        self.at = at

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > self.at:
            raise Killed


class Recorder(L.LightningModule):
    def __init__(self, log_path):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.log_path = log_path

    def training_step(self, batch, batch_idx):
        ids = batch["id"] if "id" in batch else None
        nres = batch["coord_mask"].any(-1).sum(-1).long().tolist()  # = processed_length's definition
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"epoch": self.current_epoch, "step": self.global_step, "batch_idx": batch_idx,
                                "ids": list(ids) if ids is not None else None, "nres": nres}) + "\n")
        return (self.w ** 2).sum()

    def configure_optimizers(self):
        dl = self.trainer.datamodule.train_dataloader()  # as the real model's LR-horizon path does (an extra loader)
        print(f"[horizon] extra loader len {len(dl)}, all-rank batches "
              f"{getattr(dl.batch_sampler, 'n_batches', len(dl))}")
        return torch.optim.SGD(self.parameters(), lr=1e-3)


def compose(config_name, plan_on, num_workers):
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=config_name)
    sub = cfg_exp.get("dataset_config_subdir", None)
    path = f"../configs/datasets_config/{sub}" if sub else "../configs/datasets_config/"
    with hydra.initialize(path, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp["dataset"])
    exp_dm = OmegaConf.select(cfg_exp, "datamodule")
    OmegaConf.set_struct(cfg_data.datamodule, False)
    if exp_dm is not None:
        cfg_data.datamodule.merge_with(exp_dm)
    cfg_data.datamodule.num_workers = num_workers
    if cfg_exp.opt.get("batch_size") is not None:
        cfg_data.datamodule.batch_size = cfg_exp.opt.batch_size
    cfg_data.datamodule.epoch_batch_plan = bool(plan_on)
    return cfg_exp, hydra.utils.instantiate(cfg_data.datamodule)


def run(phase, args, plan_on, kill=None, ckpt=None):
    cfg_exp, dm = compose(args.config_name, plan_on, args.num_workers)
    d = os.path.join(args.workdir, phase)
    cb = [ModelCheckpoint(dirpath=d, every_n_train_steps=args.steps, save_top_k=-1, save_last=False)]
    if kill is not None:
        cb.append(Kill(kill))
    tr = L.Trainer(max_epochs=2, accumulate_grad_batches=args.accum, callbacks=cb,
                   logger=False, enable_progress_bar=False, enable_model_summary=False, use_distributed_sampler=False,
                   accelerator="cpu", devices=1, num_sanity_val_steps=0, limit_val_batches=0)
    m = Recorder(os.path.join(args.workdir, f"{phase}.jsonl"))
    try:
        tr.fit(m, datamodule=dm, ckpt_path=ckpt)
    except Killed:
        pass
    return dm


def read(p):
    with open(p) as f:
        return [json.loads(x) for x in f]


def ckpt_at(args, phase, step):
    d = os.path.join(args.workdir, phase)
    f = [x for x in os.listdir(d) if f"step={step}.ckpt" in x]
    assert len(f) == 1, (phase, step, os.listdir(d))
    return os.path.join(d, f[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--accum", type=int, required=True, help="the live chain's accumulate_grad_batches (OVERRIDES)")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=False)
    S = args.steps

    run("legacy", args, plan_on=False, kill=S)
    ck_legacy = ckpt_at(args, "legacy", S)
    dm = run("plan_a", args, plan_on=True, kill=2 * S, ckpt=ck_legacy)
    ck_a = ckpt_at(args, "plan_a", 2 * S)
    run("plan_b", args, plan_on=True, ckpt=ck_a)

    names = dm.train_ds.file_names
    lengths = dm._dataset_lengths(dm.train_ds)
    name_len = dict(zip(names, lengths))
    sd_a = torch.load(ck_a, map_location="cpu", weights_only=False)
    plan_state = None
    for k, v in sd_a.items():
        if isinstance(v, dict) and "train_epoch_plan" in v:
            plan_state = v["train_epoch_plan"]
    assert plan_state is not None, f"no train_epoch_plan in checkpoint keys {list(sd_a)}"
    assert plan_state["epoch"] == 0, plan_state["epoch"]
    plan0 = [[names[i] for i in b] for b in plan_state["plan"][0]]

    L_ = read(os.path.join(args.workdir, "legacy.jsonl"))
    A = read(os.path.join(args.workdir, "plan_a.jsonl"))
    B = read(os.path.join(args.workdir, "plan_b.jsonl"))
    legacy_done = [r for r in L_ if r["step"] < S]
    k = len(legacy_done)
    a_done = [r for r in A if r["step"] < 2 * S]
    ep0 = [r["ids"] for r in a_done if r["epoch"] == 0] + [r["ids"] for r in B if r["epoch"] == 0]
    ep1 = [r["ids"] for r in B if r["epoch"] == 1]
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")

    check("legacy -> plan: epoch-0 remainder is plan batches k..end", ep0 == plan0[k:],
          f"(k={k}, remainder {len(ep0)} vs {len(plan0) - k})")
    check("epoch 0 batches total = one plan", k + len(ep0) == len(plan0), f"({k} + {len(ep0)} vs {len(plan0)})")
    served = [i for r in L_ + A + B for i in r["ids"]]
    nres = [n for r in L_ + A + B for n in r["nres"]]
    assert len(served) == len(nres) > 0, (len(served), len(nres))
    check("served residue count == processed_length", all(name_len[i] == n for i, n in zip(served, nres)),
          f"({sum(name_len[i] != n for i, n in zip(served, nres))} of {len(served)} samples differ)")
    ep1_flat = [i for b in ep1 for i in b]
    cl_of = {}
    for c, seqs in dm.clusterid_to_seqid_mappings["train"].items():
        for s_ in seqs:
            cl_of[s_] = c
    n_cl = len(dm._train_plan_sampler.cluster_members)
    cl1 = [cl_of[i] for i in ep1_flat]
    check("epoch 1: every cluster once (plus padding repeats)", len(set(cl1)) == n_cl and len(cl1) - n_cl < 2,
          f"({len(set(cl1))} of {n_cl} clusters, {len(cl1)} draws)")
    check("epoch 1 differs from epoch 0", ep1 != plan0)
    sp = dm._train_plan_sampler
    plan1 = [[names[i] for i in b] for b in ebp.build_epoch_plan(sp.cluster_members, sp.lengths, sp.batch_size,
                                                                  sp.len_edges, sp.seed, 1, 1)[0]]
    check("epoch 1 == its (seed, epoch) plan, served ids == planned ids", ep1 == plan1,
          f"({len(ep1)} vs {len(plan1)} batches)")
    print("INTEGRATION " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
