"""Sweep SAMPLING STEPS x {self, nonself} on the tri contact model.

Answers two things at once:
  1. Is 200 sampling steps overkill? (the user's suspicion)
  2. Does the step count matter DIFFERENTLY for the ceiling arm vs the realistic arm?

⭐ Why 2 must be asked with 1. Picking a step count on the SELF arm alone would tune the sampler on
the ceiling task -- conditioning on the correct answer -- and then apply it to the retrieved-template
task the model is actually bad at. If the arms disagree, the cheap setting chosen on the ceiling
could be exactly wrong for the real one.

⛔ Both arms use a FIXED reference seed, so every step count sees the SAME (query, reference) pairs.
Without that the template is re-drawn per point and a step-count difference is confounded with a
reference-quality difference.

⛔ NOT a training resume: builds the model and loads the -EMA companion's state_dict, ckpt_path=None.
⛔ devices=1 -- the per-class keys are data-dependent and this repo has a documented 2-GPU DDP
   deadlock from exactly that shape.
"""

import argparse
import os
import sys

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.proteina import Proteina

# Sampling steps -> dt, matching dt_sweep_eval.py's table (steps = ceil(1/dt)).
STEPS_TO_DT = {25: 0.04, 50: 0.02, 100: 0.01, 200: 0.005, 400: 0.0025}

REPORT = (
    ("validation_sampling/contact_precision_at_L_mean", "prec@L mean"),
    ("validation_sampling/contact_precision_at_L_median", "prec@L median"),
    ("validation_sampling/contact_precision_at_L5_mean", "prec@L/5 mean"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", default="training_contact_tri_full384_v1")
    ap.add_argument("--ema_ckpt", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--arms", nargs="+", default=["self", "nonself"])
    ap.add_argument("--ref_seed", type=int, default=0)
    # ⛔ The nonself arm MUST run on a chain list where every chain has a
    # different-sequence cluster-mate. 10 of the standard 32 do not, and
    # _build_self_reference_topology returns None for the WHOLE BATCH when any one
    # chain fails -- so a single mate-less chain strips conditioning from all 4 and
    # the arm silently measures the UNCONDITIONED floor. That is exactly what the
    # first sweep produced (0.29/0.28/0.22, n_distinct_refs=1 throughout).
    ap.add_argument("--nonself_chain_list", default="")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for steps in args.steps:
            with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
                cfg_exp = hydra.compose(config_name=args.config_name)
            OmegaConf.set_struct(cfg_exp, False)
            cfg_exp.hardware.ngpus_per_node_ = 1
            cfg_exp.hardware.nnodes_ = 1
            cfg_exp.log.log_wandb = False
            cfg_exp.log.checkpoint = False
            # The trajectory is gated on _val_pass_idx % every_n; force 1 so it runs for ANY index.
            cfg_exp.validation_sampling.tmscore_every_n_val_epochs = 1
            # trainer.validate() runs at global_step 0, where the trajectory would otherwise be
            # skipped before emitting anything.
            cfg_exp.validation_sampling.force_trajectory_at_step0 = True
            cfg_exp.validation_sampling.dt = STEPS_TO_DT[steps]
            cfg_exp.validation_sampling.topology_nonself = (arm == "nonself")
            cfg_exp.validation_sampling.topology_nonself_seed = args.ref_seed
            if arm == "nonself" and args.nonself_chain_list:
                cfg_exp.validation_sampling.fixed_chain_list = args.nonself_chain_list

            ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
            with hydra.initialize(ds_dir, version_base=hydra.__version__):
                cfg_data = hydra.compose(config_name=cfg_exp.dataset)

            model = Proteina(cfg_exp, store_dir="/tmp/sweep_store")
            ck = torch.load(args.ema_ckpt, map_location="cpu", weights_only=False)
            sd = ck["state_dict"] if "state_dict" in ck else ck
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if steps == args.steps[0] and arm == args.arms[0]:
                print(f"[EMA load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

            # A datamodule instance cannot be reused across repeated validate() calls (re-setup
            # trips "The truth value of a DataFrame is ambiguous"); build one per point.
            dm = hydra.utils.instantiate(cfg_data.datamodule)
            # A logger is MANDATORY, not cosmetic. _run_validation_trajectory returns early on
            # `self.logger is None or not hasattr(self.logger, "experiment")`, and that guard sits
            # AFTER the qualitative_only check so it gates the METRIC path too. With logger=False
            # this sweep ran 43 minutes and produced all-NaN with n_distinct_refs=0 -- sampling
            # never executed. CSVLogger does not work either: it has `experiment`, but the payload
            # path calls `self.logger.experiment.log(...)` and CSVLogger exposes log_metrics.
            wl = WandbLogger(project="stepsweep_probe", name=f"{arm}_{steps}",
                             save_dir="/tmp/stepsweep_wandb", offline=True)
            model._fixed_val_batches_cache = None
            model._val_pass_idx = 0
            trainer = L.Trainer(accelerator="gpu", devices=1, num_nodes=1, logger=wl,
                                enable_checkpointing=False, enable_progress_bar=False,
                                limit_val_batches=cfg_exp.opt.get("limit_val_batches", 64))
            trainer.validate(model, datamodule=dm, ckpt_path=None, verbose=False)
            cm = {k: float(v) for k, v in trainer.callback_metrics.items()}
            got = {lab: next((v for k, v in cm.items() if k.endswith(key)), float("nan"))
                   for key, lab in REPORT}
            refs = getattr(model, "_last_sampling_ref_ids", [])
            rows.append((arm, steps, got, len(set(refs))))
            print(f"[{arm:>7} steps={steps:>3}] " +
                  "  ".join(f"{lab}={val:.4f}" for lab, val in got.items()) +
                  f"  (n_distinct_refs={len(set(refs))})", flush=True)
            del model, dm, trainer
            torch.cuda.empty_cache()

    print("\n===== SUMMARY =====")
    print(f"{'arm':>8} {'steps':>6} " + " ".join(f"{lab:>15}" for _, lab in REPORT))
    for arm, steps, got, _ in rows:
        print(f"{arm:>8} {steps:>6} " + " ".join(f"{got[lab]:>15.4f}" for _, lab in REPORT))
    print("\n⚠️ Compare WITHIN an arm for the step effect. Across arms the conditioning differs by")
    print("   construction, so a self-vs-nonself gap is the point, not a confound.")
    print("⚠️ n_distinct_refs>0 on the nonself arm confirms real templates were used; 0 would mean")
    print("   the arm silently fell through to unconditioned sampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
