"""Sampling-step (dt) sweep via EMA weights + trainer.validate -- NOT a training resume.

Why this shape:
  * `ckpt_path=` is a TRAINING-RESUME mechanism. It restores optimizer state and epoch, and
    on this repo's `last.ckpt` that fails with `KeyError: 'opt'` because EMAOptimizer expects
    the wrapper's {'opt', 'ema'} dict while last.ckpt holds a PLAIN optimizer state.
  * The EMA weights are NOT inside last.ckpt at all. `EmaModelCheckpoint._save_checkpoint`
    writes a companion `last-EMA.ckpt` whose `state_dict` IS the EMA weights (verified
    2026-08-24: last.ckpt has no 'ema' key and optimizer_states[0] has only
    ['param_groups','state']).
  * So: build the model, load the -EMA companion's state_dict, and call trainer.validate()
    with ckpt_path=None. No resume, no optimizer state, no epoch reconciliation.

Sampling steps = ceil(1/dt). 200 was the inherited default and was cut to 50 purely for cost
while local_attn was the arm being tuned -- tri's step count has never been validated.
"""

import argparse
import os
import sys

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.proteina import Proteina

STEPS_TO_DT = {50: 0.02, 100: 0.01, 150: 1.0 / 150, 200: 0.005, 400: 0.0025}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--ema_ckpt", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 100, 150, 200, 400])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False          # read results from stdout, not wandb
    cfg_exp.log.checkpoint = False

    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)

    model = Proteina(cfg_exp, store_dir="/tmp/dtsweep_store")
    ck = torch.load(args.ema_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[EMA load] {args.ema_ckpt}")
    print(f"[EMA load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"[EMA load] first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"[EMA load] first unexpected: {unexpected[:5]}")

    for steps in args.steps:
        dt = STEPS_TO_DT[steps]
        cfg_exp.validation_sampling.dt = dt
        model.cfg_exp = cfg_exp
        model._fixed_val_batches_cache = None      # rebuild per dt (cheap, keeps state clean)
        model._validation_contact_results = []
        trainer = L.Trainer(
            accelerator="gpu", devices=1, num_nodes=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            limit_val_batches=cfg_exp.opt.get("limit_val_batches", 64),
        )
        trainer.validate(model, datamodule=datamodule, ckpt_path=None, verbose=False)
        res = getattr(model, "_validation_contact_results", []) or []
        vals = [r["contact_precision_at_L"] for r in res if "contact_precision_at_L" in r]
        if vals:
            import numpy as np
            a = np.array(vals, dtype=float)
            print(f"RESULT {args.tag} steps={steps:4d} dt={dt:.6f} "
                  f"n={len(a):4d} mean={a.mean():.4f} median={np.median(a):.4f}", flush=True)
        else:
            print(f"RESULT {args.tag} steps={steps:4d} dt={dt:.6f} NO SAMPLES ACCUMULATED", flush=True)


if __name__ == "__main__":
    main()
