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
import statistics
import sys

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils import sampletrace

STEPS_TO_DT = {50: 0.02, 100: 0.01, 150: 1.0 / 150, 200: 0.005, 400: 0.0025}

# Reported for every sweep point. precision_at_L alone hid the separation-resolved split, which
# is the part that actually distinguished the arms once conditioning was fixed.
REPORT_KEYS = (
    ("validation_sampling/contact_precision_at_L_mean", "prec@L"),
    ("validation_sampling/contact_precision_at_L5_mean", "prec@L/5"),
    ("validation_sampling/contact_medium_range_precision_at_L5_mean", "medium@L/5"),
    ("validation_sampling/contact_long_range_precision_at_L5_mean", "long@L/5"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--ema_ckpt", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 100, 150, 200, 400])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trace_dir", default="/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves/traces")
    ap.add_argument("--sampling_mode", default=None,
                    help="sc (SDE, default from config) or vf (pure ODE: c_t + v*dt, no score term)")
    ap.add_argument("--sc_scale_noise", type=float, default=None,
                    help="0 removes the injected noise but KEEPS the gt*score drift")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Independent validate passes per point. SDE (sampling_mode=sc) draws "
                         "fresh noise and is NOT seeded here, so a single pass carries ~0.02-0.04 "
                         "of spread on a 32-chain mean -- comparable to the effects being swept. "
                         "ODE (vf) is deterministic and needs only 1.")
    args = ap.parse_args()

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False          # read results from stdout, not wandb
    cfg_exp.log.checkpoint = False
    # The trajectory is gated by `self._val_pass_idx % every_n != 0 -> return`. Rather than
    # reason about when _val_pass_idx is incremented (setting it to -1 fails: -1 % 5 == 4 in
    # Python, so the gate closes), force every_n = 1 so the condition holds for ANY pass index.
    cfg_exp.validation_sampling.tmscore_every_n_val_epochs = 1
    # trainer.validate() runs at global_step == 0, where validation_step_data returns
    # BEFORE any diag line. This opt-in is what lets the trajectory run for loaded weights.
    cfg_exp.validation_sampling.force_trajectory_at_step0 = True
    if args.sampling_mode is not None:
        cfg_exp.validation_sampling.sampling_mode = args.sampling_mode
    if args.sc_scale_noise is not None:
        cfg_exp.validation_sampling.sc_scale_noise = args.sc_scale_noise
    print(f"[sampler] mode={cfg_exp.validation_sampling.sampling_mode} "
          f"sc_scale_noise={cfg_exp.validation_sampling.sc_scale_noise}", flush=True)

    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    def fresh_datamodule():
        # A datamodule instance cannot be reused across repeated trainer.validate() calls:
        # re-setup trips "The truth value of a DataFrame is ambiguous". Build one per variant.
        return hydra.utils.instantiate(cfg_data.datamodule)

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
        per_key = {k: [] for k, _ in REPORT_KEYS}
        for rep in range(max(1, args.repeats)):
            cfg_exp.validation_sampling.dt = dt
            model.cfg_exp = cfg_exp
            model._fixed_val_batches_cache = None  # rebuild per pass (cheap, keeps state clean)
            model._val_pass_idx = 0
            # A logger is MANDATORY here, not cosmetic: _run_validation_trajectory returns
            # early via `if is_rank0 and (self.logger is None or not hasattr(self.logger,
            # "experiment"))`, and that guard sits AFTER the qualitative_only check, so it gates
            # the metric path too. CSVLogger satisfies hasattr(.,"experiment") but its writer has
            # log_metrics, not log(), and on_validation_epoch_end_data calls
            # `self.logger.experiment.log(payload)`. Offline WandB gives the right interface with
            # no network and no run clutter.
            wl = WandbLogger(project="dtsweep_probe", name=f"{args.tag}_{steps}_r{rep}",
                             save_dir="/tmp/dtsweep_wandb", offline=True)
            trainer = L.Trainer(
                accelerator="gpu", devices=1, num_nodes=1, logger=wl,
                enable_checkpointing=False, enable_progress_bar=False,
                limit_val_batches=cfg_exp.opt.get("limit_val_batches", 64),
            )
            sampletrace.reset()
            trainer.validate(model, datamodule=fresh_datamodule(), ckpt_path=None, verbose=False)
            if sampletrace.enabled():
                out = os.path.join(args.trace_dir, f"trace_{args.tag}_{steps}" + (f"_r{rep}" if args.repeats > 1 else ""))
                os.makedirs(args.trace_dir, exist_ok=True)
                sampletrace.dump(out)
                sm = sampletrace.summary()
                st = sm["steps"]
                # Runtime evidence, printed so it is in the job log even if the npz is lost.
                print(f"[TRACE {args.tag} steps={steps}] effective args: {sm['args']}", flush=True)
                if st:
                    n_sc = sum(1 for r in st if r["sc_present"])
                    n_prev = sum(1 for r in st if r["sc_is_prev_pred"])
                    print(f"[TRACE {args.tag} steps={steps}] recorded={len(st)} "
                          f"sc_present={n_sc} sc_is_prev_pred={n_prev}", flush=True)
                    for r in st[:3] + st[-2:]:
                        print(f"   step={r['step']:4d} t={r['t']:.4f} sc={r['sc_present']} "
                              f"sc_norm={r['sc_norm']:.3f} is_prev={r['sc_is_prev_pred']} "
                              f"pred_mean={r['pred_mean']:.4f} frac>0.5={r['pred_frac_gt_half']:.4f}",
                              flush=True)
            # Read from callback_metrics, NOT from _validation_contact_results:
            # on_validation_epoch_end_data CLEARS that list at the end of the epoch, so it is always
            # empty by the time validate() returns. The logged metrics persist.
            cm = {k: float(v) for k, v in trainer.callback_metrics.items()}
            mean = cm.get("validation_sampling/contact_precision_at_L_mean")
            med = cm.get("validation_sampling/contact_precision_at_L_median")
            nsamp = cm.get("validation_sampling/contact_n_samples")
            if mean is not None:
                for k, _ in REPORT_KEYS:
                    if cm.get(k) is not None:
                        per_key[k].append(cm[k])
                print(f"RESULT {args.tag} steps={steps:4d} dt={dt:.6f} rep={rep} "
                      f"n={nsamp} mean={mean:.4f} median={med:.4f}", flush=True)
            else:
                keys = sorted(k for k in cm if "contact" in k)
                print(f"RESULT {args.tag} steps={steps:4d} dt={dt:.6f} rep={rep} NO METRIC; "
                      f"contact keys present: {keys[:6]}", flush=True)

        # One line per sweep point carrying its own spread, so a dt-to-dt difference can be read
        # against the measurement noise instead of against nothing.
        for k, label in REPORT_KEYS:
            vals = per_key[k]
            if not vals:
                continue
            m = statistics.mean(vals)
            sd_ = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            print(f"AGG {args.tag} steps={steps:4d} {label:<11s} "
                  f"mean={m:.4f} sd={sd_:.4f} n={len(vals)} vals={[round(v, 4) for v in vals]}",
                  flush=True)


if __name__ == "__main__":
    main()
