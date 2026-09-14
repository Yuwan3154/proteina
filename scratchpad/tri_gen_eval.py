"""Generative (sampling-path) contact P@L for ONE checkpoint under a CHOSEN topology arm.

⭐ Why this exists, and why `nonself_val_eval.py` could not be reused. That harness deliberately
switches the sampling trajectory OFF (`tmscore_every_n_val_epochs = 10**9`) because it measures the
LOSS path: `contact_precision_at_L_single_step_*`. The headline number people quote for these models
is a DIFFERENT metric -- `validation_sampling/contact_precision_at_L_median`, produced by a full
reverse-diffusion trajectory. Comparing one to the other is the exact confusion that already cost
this project a retracted top-k finding, so this script drives the SAMPLING path and reports only
`validation_sampling/*`.

⛔ THE ARM IS THE WHOLE POINT. `validation_sampling.topology_nonself` defaults to False, which calls
`transform.self_reference(...)` and hands the model THE CORRECT ANSWER -- a CEILING, not the task.
The old `tri_full384` run's config has no such key, so every historical number from it is a
self-conditioned ceiling; the new cb8synth run sets it True and measures the realistic task. Any
old-vs-new comparison MUST fix this arm on both sides or it compares two different tasks.

⛔ Run each model in ITS OWN checkout. The SSE alphabet changed (65 -> 44 tokens: loops dropped), so
the same integer means different things to the two models. Pointing this script at the old
checkpoint from the new repo would feed vocab-44 ids into a vocab-65 embedding and read as a
quiet accuracy drop rather than an error. Correct handling = old ckpt + old repo + old dataset cfg.

Usage (one model, one arm):
    python scratchpad/tri_gen_eval.py --config_name <cfg> --ema_ckpt <path> --arm self|nonself
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

REPORT = (
    ("validation_sampling/contact_precision_at_L_median", "P@L (median)  <- headline"),
    ("validation_sampling/contact_precision_at_L_mean", "P@L (mean)"),
    ("validation_sampling/contact_precision_at_L5_median", "P@L/5 (median)"),
    ("validation_sampling/contact_precision_at_L_long_median", "P@L long-range (median)"),
    ("validation_sampling/contact_f1_median", "F1 (median)"),
    ("validation_sampling/contact_recall_median", "recall (median)"),
    ("validation_sampling/contact_n_samples", "n chains scored"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--ema_ckpt", required=True)
    ap.add_argument("--arm", choices=["self", "nonself"], required=True)
    ap.add_argument("--limit_val_batches", type=int, default=200)
    ap.add_argument("--store", default="/tmp/tri_gen_eval_store")
    ap.add_argument("--nonself_seed", type=int, default=0)
    args = ap.parse_args()

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False
    # Fire the trajectory on THIS validation rather than waiting for its normal cadence.
    cfg_exp.validation_sampling.tmscore_every_n_val_epochs = 1
    cfg_exp.validation_sampling.force_trajectory_at_step0 = True
    cfg_exp.validation_sampling.topology_nonself = (args.arm == "nonself")
    cfg_exp.validation_sampling.topology_nonself_seed = args.nonself_seed
    print(f"[arm] topology_nonself={cfg_exp.validation_sampling.topology_nonself} "
          f"(self => the model is handed the correct topology; a CEILING)")
    print(f"[chains] fixed_chain_list={cfg_exp.validation_sampling.get('fixed_chain_list')}")

    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)
    OmegaConf.set_struct(cfg_data, False)
    print(f"[data] {cfg_exp.dataset}")

    model = Proteina(cfg_exp, store_dir=args.store)
    ck = torch.load(args.ema_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[EMA load] {args.ema_ckpt}\n[EMA load] missing={len(missing)} unexpected={len(unexpected)}")
    # ⛔ A vocab mismatch shows up HERE as a shape-mismatch skip, not as a crash. Name the tensors.
    for n in list(missing)[:5]:
        print(f"[EMA load] missing: {n}")
    for n in list(unexpected)[:5]:
        print(f"[EMA load] unexpected: {n}")

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    trainer = L.Trainer(accelerator="gpu", devices=1, num_nodes=1, logger=False,
                        enable_progress_bar=False, limit_val_batches=args.limit_val_batches)
    trainer.validate(model, datamodule=dm)

    cm = {k: float(v) for k, v in trainer.callback_metrics.items()}
    print(f"\n===== config={args.config_name}  arm={args.arm} =====")
    for key, label in REPORT:
        hits = [v for k, v in cm.items() if k.endswith(key)]
        print(f"  {label:<34s} {hits[0]:.4f}" if hits else f"  {label:<34s} (absent)")
    print("\n--- every validation_sampling/contact_* key logged ---")
    for k in sorted(cm):
        if "validation_sampling/contact" in k:
            print(f"  {k:<64s} {cm[k]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
