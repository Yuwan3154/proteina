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

Directive-B additions (2026-09-22), all opt-in; without them the harness behaves exactly as before:
  --seed S             L.seed_everything(S) before validate: two passes differing only in weights
                       then see the SAME diffusion noise (paired design).
  --dump_dir D         per-sample maps + samples.jsonl (per-chain metrics); one fresh dir per pass.
  --zero_ca_features   zero cell_in's min_ca_dist / mean_ca_dist columns (OLD checkout only). The
                       features are standardised and reach the net only through that Linear, so this
                       is an exact removal of their linear term (bias kept).
  --mask_regime variable   mask arm under the NEW model's own training convention (a variable-length
                       fully masked reference); `single` = one MASK token = the OLD convention.
"""

import argparse
import json
import os
import sys

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.proteina import Proteina


class _NullExperiment:
    def log(self, *args, **kwargs):
        pass


class _NullLogger(Logger):
    """⛔ The sampling trajectory REFUSES TO RUN without a logger exposing `.experiment`.

    `_run_validation_trajectory` opens with
        `if is_rank0 and (self.logger is None or not hasattr(self.logger, "experiment")): return`
    so a `Trainer(logger=False)` yields ZERO `validation_sampling/*` metrics -- silently, with a
    cheerful "traj done" in the diagnostics. That is exactly what job 22757754 produced: the fixed
    32-chain set loaded, all 8 batches "ran" in ~0 ms, and every metric came back `(absent)`.

    A real wandb run is not wanted for an offline eval, so this satisfies the gate and swallows the
    payload. The numbers still reach `trainer.callback_metrics`, because the model logs them via
    `self.log(..., logger=False)` independently of this object.
    """

    @property
    def name(self):
        return "null"

    @property
    def version(self):
        return "0"

    @property
    def experiment(self):
        return _NullExperiment()

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        pass

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        pass

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
    ap.add_argument("--arm", choices=["self", "nonself", "mask"], required=True)
    ap.add_argument("--limit_val_batches", type=int, default=200)
    ap.add_argument("--store", default="/tmp/tri_gen_eval_store")
    ap.add_argument("--nonself_seed", type=int, default=0)
    # ⛔ SAME POPULATION before comparing arms. On the default 32-chain list the OLD model's nonself
    # pass sampled 7 chains UNCONDITIONED (no cluster-mate -> no topology at all, not a template),
    # while the other three passes had zero. Those three arms were therefore measured on 32 chains
    # and the fourth on an unconditioned-contaminated mixture. Pass val_fixed_nonself.txt to restrict
    # every arm to the 22 chains that actually have a different-sequence mate.
    ap.add_argument("--fixed_chain_list", default=None)
    ap.add_argument("--seed", type=int, default=None, help="seed_everything before validate")
    ap.add_argument("--dump_dir", default=None, help="write per-sample maps + samples.jsonl here")
    ap.add_argument("--zero_ca_features", action="store_true",
                    help="zero cell_in's min_ca_dist/mean_ca_dist columns (old checkout only)")
    ap.add_argument("--mask_regime", choices=["single", "variable"], default="single")
    args = ap.parse_args()
    assert args.mask_regime == "single" or args.arm == "mask", "--mask_regime applies to --arm mask"
    if args.dump_dir is not None:
        # samples.jsonl APPENDS and the per-stem counter restarts per process: a reused dir would
        # silently mix passes.
        assert not (os.path.isdir(args.dump_dir) and os.listdir(args.dump_dir)), \
            f"dump dir {args.dump_dir} is not empty"

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
    if args.fixed_chain_list is not None:
        cfg_exp.validation_sampling.fixed_chain_list = args.fixed_chain_list
    if args.dump_dir is not None:
        cfg_exp.validation_sampling.contact_dump_dir = args.dump_dir
    if args.arm == "mask" and args.mask_regime == "variable":
        from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform
        # ⛔ The OLD checkout lacks this path and would silently run the SELF arm instead.
        assert hasattr(TopologyReferenceTransform, "masked_reference"), \
            "this checkout has no masked_reference -- variable mask regime unavailable"
        cfg_exp.validation_sampling.topology_masked = True
        print("[arm] mask (variable): every chain gets this model's training-time dropped reference")
    elif args.arm == "mask":
        # ⭐ The UNCONDITIONAL arm, and the discriminator the 2x2 cannot supply on its own: a model
        # whose score barely moves between `self` and `nonself` is either robust to the reference or
        # IGNORING it, and only the no-reference floor tells those apart.
        # ⛔ Not done by setting `model.nn.topology_cond=False` -- that changes what gets BUILT, so
        # the checkpoint would no longer match. A reference map covering no chain leaves the
        # architecture intact and drives the resolver's documented "sampling UNCONDITIONED" path.
        os.makedirs(args.store, exist_ok=True)
        sentinel = os.path.join(args.store, "empty_ref_map.json")
        with open(sentinel, "w") as fh:
            json.dump({"__no_such_chain__": "__no_such_chain__"}, fh)
        cfg_exp.validation_sampling.topology_reference_map = sentinel
        print("[arm] mask: reference map covers no chain -> every sample is UNCONDITIONED")
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
    if args.zero_ca_features:
        from proteinfoundation.datasets.sse_topology import PAIR_FEATURE_NAMES
        ca = ("min_ca_dist", "mean_ca_dist")
        assert all(n in PAIR_FEATURE_NAMES for n in ca), "this checkout has no CA pair features"
        # nn_sc would be a deep copy that keeps the old weights (proteina.py load_state_dict)
        assert getattr(model, "nn_sc", None) is None, "nn_sc copy present; zero it too"
        idx = [int(i) for i in model.nn.pair_feat_idx]
        cols = [2 + idx.index(PAIR_FEATURE_NAMES.index(n)) for n in ca]
        w = model.nn.cell_in.weight
        assert w.shape[1] == 2 + len(idx), f"cell_in width {w.shape[1]} != 2 + {len(idx)}"
        print(f"[zero_ca] cols={cols} |W| before={[round(float(w[:, c].norm()), 4) for c in cols]}")
        with torch.no_grad():
            w[:, cols] = 0.0
        print(f"[zero_ca] cols={cols} |W| after={[round(float(w[:, c].norm()), 4) for c in cols]}")

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    trainer = L.Trainer(accelerator="gpu", devices=1, num_nodes=1, logger=_NullLogger(),
                        enable_progress_bar=False, limit_val_batches=args.limit_val_batches)
    if args.seed is not None:
        L.seed_everything(args.seed)
        print(f"[seed] seed_everything({args.seed}) before validate")
    trainer.validate(model, datamodule=dm)

    cm = {k: float(v) for k, v in trainer.callback_metrics.items()}
    # ⛔ Fail loudly. A silent (absent) row reads like "the model scored nothing" when it actually
    # means the trajectory never ran -- the failure mode that wasted job 22757754.
    if not any("validation_sampling/contact" in k for k in cm):
        print("\n⛔ FATAL: no validation_sampling/contact_* metrics were produced. The sampling "
              "trajectory did not run -- check the logger gate and tmscore_n_samples.")
        return 2
    print(f"\n===== config={args.config_name}  arm={args.arm}  mask_regime={args.mask_regime}  "
          f"zero_ca={args.zero_ca_features}  seed={args.seed} =====")
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
