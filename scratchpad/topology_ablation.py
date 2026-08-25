"""Controlled with/without-topology ablation, matched on chain AND on t.

Why this exists as its own script: the single-step dumps did NOT establish that the topology
reference matters to a trained model. tri's masked sample sat at t=0.156 -- the only low-t sample
in the whole dump -- so its low score is confounded with noise level, and la's masked sample
scored a perfect 1.0000 at t=0.619. The effect has to be measured with t held fixed.

It is cheap because it disables the sampling trajectory (``tmscore_n_samples = 0``), which is the
part that costs ~25 minutes. Only the validation LOSS path runs, and that is the path that carries
a ground-truth contact map, a drawn t, and a built c_t -- exactly the single-step setting. Each
captured batch is then forwarded twice, with and without the topology_* keys, so t, c_t, the chain
and the noise draw are all IDENTICAL across the two arms of the comparison.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proteinfoundation.proteinflow.proteina import Proteina
from verify_conditioning_fix import as_plain_dict, precision_at_l5

MASK_ONLY = 1  # a reference of exactly one valid element is the MASK fallback, not a real topology


class BatchCapture:
    """Keeps the loss-path batches themselves, on device, for a second forward afterwards."""

    def __init__(self, n_keep):
        self.n_keep = n_keep
        self.batches = []
        self.n_calls = 0

    def __call__(self, module, args, output):
        batch = args[0] if args else None
        self.n_calls += 1
        if batch is None or not hasattr(batch, "__contains__"):
            return
        if "contact_map" not in batch or len(self.batches) >= self.n_keep:
            return
        self.batches.append(batch)

    def report(self):
        return f"hook fired {self.n_calls}x, kept {len(self.batches)} loss-path batches"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--ema_ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_batches", type=int, default=8)
    args = ap.parse_args()

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False
    # The trajectory is the expensive part and is not needed: the ablation is a single-step
    # question. 0 disables it outright (see validation_step_data's stage-1 branch).
    cfg_exp.validation_sampling.tmscore_n_samples = 0

    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    model = Proteina(cfg_exp, store_dir="/tmp/ablation_store")
    ck = torch.load(args.ema_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[EMA load] {args.ema_ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"[EMA load] epoch={ck.get('epoch')} global_step={ck.get('global_step')}", flush=True)

    cap = BatchCapture(args.n_batches)
    handle = model.nn.register_forward_hook(cap)
    wl = WandbLogger(project="ablation_probe", name=args.tag,
                     save_dir="/tmp/ablation_wandb", offline=True)
    trainer = L.Trainer(
        accelerator="gpu", devices=1, num_nodes=1, logger=wl,
        enable_checkpointing=False, enable_progress_bar=False,
        limit_val_batches=args.n_batches,
    )
    trainer.validate(model, datamodule=hydra.utils.instantiate(cfg_data.datamodule),
                     ckpt_path=None, verbose=False)
    handle.remove()
    print(f"\n[capture] {cap.report()}", flush=True)
    if not cap.batches:
        print("[ablation] NOTHING CAPTURED", flush=True)
        return

    model.nn.eval()
    print(f"\n===== {args.tag}: with vs without topology, matched on chain and t =====", flush=True)
    print(f"{'t':>7} {'L':>5} {'n_sse':>6} {'with':>8} {'without':>8} {'delta':>8}  {'mean|dP|':>9}", flush=True)
    deltas = []
    for batch in cap.batches:
        full = as_plain_dict(batch, device=model.device)
        stripped = {k: v for k, v in full.items() if not k.startswith("topology_")}
        with torch.no_grad():
            a = model.nn(full)["contact_map_pred"].float()
            b = model.nn(stripped)["contact_map_pred"].float()
        gt = full["contact_map"].float()
        mask = full["mask"].bool()
        t = full["t"]
        he = full.get("topology_he_tokens")
        for s in range(a.shape[0]):
            n_sse = int((he[s] > 0).sum()) if he is not None else 0
            pw = precision_at_l5(a[s], gt[s], mask[s])
            po = precision_at_l5(b[s], gt[s], mask[s])
            dmap = float((a[s] - b[s]).abs().mean())
            # A MASK-only reference SHOULD behave like no reference; flagging it keeps those rows
            # from being read as evidence either way.
            flag = "  <- MASK-only reference" if n_sse <= MASK_ONLY else ""
            print(f"{float(t[s]):7.3f} {int(mask[s].sum()):5d} {n_sse:6d} "
                  f"{pw:8.4f} {po:8.4f} {pw - po:+8.4f}  {dmap:9.5f}{flag}", flush=True)
            if n_sse > MASK_ONLY:
                deltas.append(pw - po)
    if deltas:
        m = sum(deltas) / len(deltas)
        n_pos = sum(1 for d in deltas if d > 0)
        print(f"\n[{args.tag}] real-reference samples n={len(deltas)}  "
              f"mean precision@L/5 gain from topology = {m:+.4f}  "
              f"({n_pos}/{len(deltas)} improved)", flush=True)
    else:
        print(f"\n[{args.tag}] no real-reference samples captured (all MASK dropout)", flush=True)


if __name__ == "__main__":
    main()
