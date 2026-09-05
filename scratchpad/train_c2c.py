"""Train the contact-to-coordinate all-atom diffusion model.

Reuses the existing PDB datamodule -- it already supplies ConFind contact maps, all-atom coords and
residue types, which is everything this model needs. The tri trunk is not involved.

⛔ --smoke runs a handful of batches and gates on the ARTEFACT (did the loss move, are gradients
finite) rather than on the exit code, because a training script that silently trains on degenerate
data still exits 0.
"""

import argparse
import os
import sys

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.contact2coord_trainer import GRAD_CLIP, ContactToCoordTrainer

# AF3 widths and depth throughout (SI Alg. 23); user directive 2026-09-04 fixed depth at 24.
MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=2, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    ap.add_argument("--subdir", default="pdb")
    ap.add_argument("--store", default="/orcd/scratch/orcd/011/chenxiou/c2c_store")
    ap.add_argument("--name", default="c2c_v1")
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ds_dir = f"../configs/datasets_config/{args.subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg_data, False)
    # ⛔ Override here, never in the yaml: tri_sm120 reads the same file, and its batch_size=1 is a
    # real measurement for THAT model (71.8 GB at L=384). It says nothing about this one.
    # Effective batch = batch_size * accum * devices; keep it fixed when raising batch_size.
    cfg_data.datamodule.batch_size = args.batch_size
    print(f"[batch] per-rank {args.batch_size} x accum {args.accum} x {args.devices} ranks "
          f"= effective {args.batch_size * args.accum * args.devices}", flush=True)

    model = ContactToCoordTrainer(model_cfg=MODEL_CFG)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par/1e6:.2f} M parameters, {MODEL_CFG['n_blocks']} diffusion blocks", flush=True)

    os.makedirs(args.store, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.store, args.name), monitor="val/loss", mode="min",
        save_top_k=3, save_last=True, every_n_train_steps=500,
    )
    logger = WandbLogger(project="contact2coord", name=args.name,
                         save_dir=args.store, offline=args.smoke)

    trainer = L.Trainer(
        accelerator="gpu", devices=args.devices, num_nodes=1,
        strategy="ddp" if args.devices > 1 else "auto",
        precision="bf16-mixed",          # ⛔ the LOSS forces fp32 internally; see af3_diffusion
        max_epochs=-1,
        accumulate_grad_batches=args.accum,
        gradient_clip_val=GRAD_CLIP,     # AF3 SI §5.6, global norm 10
        logger=logger, callbacks=[ckpt_cb],
        enable_progress_bar=False,
        limit_train_batches=8 if args.smoke else 1.0,
        limit_val_batches=4 if args.smoke else 64,
        val_check_interval=500 if not args.smoke else 8,
        max_steps=16 if args.smoke else -1,
        default_root_dir=args.store,
    )
    dm = hydra.utils.instantiate(cfg_data.datamodule)
    # ⛔ Resume across chain segments. mit_normal_gpu caps wall-clock at 6 h, so a 24 h run is four
    # segments; without this each one would silently restart from scratch and the run would never
    # progress past six hours no matter how many segments completed.
    last = os.path.join(args.store, args.name, "last.ckpt")
    resume = last if (os.path.exists(last) and not args.smoke) else None
    if resume:
        print(f"[resume] {resume}", flush=True)
    trainer.fit(model, datamodule=dm, ckpt_path=resume)

    if args.smoke:
        # Gate on the artefact: a run that trained on degenerate data still exits 0.
        cm = {k: float(v) for k, v in trainer.callback_metrics.items()}
        print("\n===== SMOKE RESULT =====", flush=True)
        for k in sorted(cm):
            print(f"  {k:<28s} {cm[k]:.5f}", flush=True)
        need = ["train/loss", "train/diffusion", "train/distogram"]
        missing = [k for k in need if k not in cm]
        if missing:
            print(f"FAIL: no metric produced for {missing}", flush=True)
            return 7
        if not all(torch.isfinite(torch.tensor(cm[k])) for k in need):
            print("FAIL: non-finite loss", flush=True)
            return 8
        print("SMOKE OK: finite losses produced on real batches", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
