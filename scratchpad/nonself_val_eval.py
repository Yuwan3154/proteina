"""Stratified validation of the tri contact model by WHAT IT WAS CONDITIONED ON.

⭐ Why this exists. The headline `validation_sampling/contact_precision_at_L_*` is **100%
self-topology-reference by construction** -- `_build_self_reference_topology` calls
`transform.self_reference(stem, L)`, un-augmented and never dropped. It is a CEILING, not a
measurement of the task we care about (thread a template that is NOT the answer). The loss-path
metric uses the real dataloader transform instead, whose mixture is ~25% MASK / ~17-20% self /
rest genuine cross-chain -- so it blends three different tasks into one number.

This harness runs `trainer.validate` on loaded EMA weights and reports
`contact_precision_at_L_single_step_ref_{self,retrieved,mask}` plus the `contact_ref_frac_*`
companions, under two arms:

    --arm mixed      the dataloader's natural mixture (baseline; shows the blend)
    --arm nonself    require_nonself=True -> the transform NEVER returns the query's own
                     topology, so `retrieved` is the whole population

⛔ NOT a training resume. `ckpt_path=` restores optimizer state and fails on this repo's
last.ckpt; and for the tri model the EMA weights are the `state_dict` of the companion
`last-EMA.ckpt` -- NOT `ckpt["ema"]["params"]`, which is the c2c format. Verified in
dt_sweep_eval.py's header (2026-08-24).

⛔ devices=1, deliberately. The per-class keys are DATA-DEPENDENT (a batch may contain no `mask`
samples), and this repo has a documented 2-GPU DDP deadlock caused by exactly that: ranks issuing
different numbers of collectives. The metrics are per-rank and must not be sync_dist'd.
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
    ("contact_precision_at_L_single_step", "all"),
    ("contact_precision_at_L_single_step_ref_self", "self (ceiling)"),
    ("contact_precision_at_L_single_step_ref_retrieved", "RETRIEVED (the real task)"),
    ("contact_precision_at_L_single_step_ref_mask", "mask (unconditioned floor)"),
    ("contact_precision_at_L_noisy_floor", "noisy-input floor"),
)
FRACS = (
    ("contact_ref_frac_self", "frac self"),
    ("contact_ref_frac_retrieved", "frac retrieved"),
    ("contact_ref_frac_mask", "frac mask"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", default="training_contact_tri_full384_v1")
    ap.add_argument("--ema_ckpt", required=True)
    ap.add_argument("--arm", choices=["mixed", "nonself"], default="nonself")
    ap.add_argument("--limit_val_batches", type=int, default=200)
    args = ap.parse_args()

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False
    # Fire the per-sample metric on EVERY validation batch; the default of 100 would give us a
    # handful of samples out of a 200-batch pass and the per-class means would be noise.
    cfg_exp.log.single_step_metrics_every_n_steps = 1
    # The sampling arm is the 100%-self ceiling and costs a full reverse diffusion per chain.
    # This harness is about the LOSS path, so switch the trajectory off entirely.
    cfg_exp.validation_sampling.tmscore_every_n_val_epochs = 10**9
    cfg_exp.validation_sampling.force_trajectory_at_step0 = False

    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)
    OmegaConf.set_struct(cfg_data, False)

    if args.arm == "nonself":
        # Flip require_nonself on the TopologyReferenceTransform entry of the transform list.
        n_set = 0
        for tr in cfg_data.datamodule.transforms:
            if "TopologyReferenceTransform" in str(tr.get("_target_", "")):
                tr.require_nonself = True
                n_set += 1
        # ⛔ A non-matching config edit is a SILENT no-op and the arm would quietly be `mixed`.
        assert n_set == 1, f"expected exactly 1 TopologyReferenceTransform, patched {n_set}"
        print("[arm] nonself: require_nonself=True (self-reference can never be returned)")
    else:
        print("[arm] mixed: the dataloader's natural conditioning mixture")

    model = Proteina(cfg_exp, store_dir="/tmp/nonself_store")
    ck = torch.load(args.ema_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[EMA load] {args.ema_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"[EMA load] first missing: {missing[:3]}")

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    trainer = L.Trainer(accelerator="gpu", devices=1, num_nodes=1, logger=False,
                        enable_progress_bar=False, limit_val_batches=args.limit_val_batches)
    trainer.validate(model, datamodule=dm)

    cm = {k: float(v) for k, v in trainer.callback_metrics.items()}
    print(f"\n===== arm={args.arm}  ({args.limit_val_batches} val batches) =====")
    for key, label in REPORT:
        hits = [v for k, v in cm.items() if k.endswith(key)]
        print(f"  {label:<28s} {hits[0]:.4f}" if hits else f"  {label:<28s} (absent)")
    print("  --- mixture ---")
    for key, label in FRACS:
        hits = [v for k, v in cm.items() if k.endswith(key)]
        print(f"  {label:<28s} {hits[0]:.3f}" if hits else f"  {label:<28s} (absent)")
    print("\n⚠️ Means over whatever the sampler drew. Read the fracs: a class with a tiny frac has")
    print("   a mean over few samples and should not be compared against a well-populated one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
