"""How far off-origin are the training targets, and does that break the EDM preconditioning?

Two verified facts motivate this:
  1. GlobalRotationTransform (transforms.py:286) rotates about the ORIGIN on UNCENTERED deposited
     coordinates, so each epoch the centroid lands somewhere on a sphere of radius |centroid|.
  2. Nothing on the c2c training path centres anything, while INFERENCE re-centres every rollout
     step (contact2coord.py:217).

EDM assumes the data has scale SIGMA_DATA (16 A here); c_in/c_skip/c_out are all derived from it. A
large DC offset inflates the true scale and silently miscalibrates every one of those constants, and
it means the model trains on structures scattered over a sphere but samples from structures at the
origin.

Reports the offset magnitude, the RMS coordinate scale with and without centering, and what
SIGMA_DATA would have to be to match each.
"""

import argparse
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import SIGMA_DATA
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=64, c_z=32, c_token=64, c_atom=32, c_atompair=8, n_blocks=1, n_heads=2,
    n_tri_blocks=1, tri_hidden=16, transition_n=1, atom_blocks=1, atom_heads=2,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")
    mod = ContactToCoordTrainer(model_cfg=MODEL_CFG)

    offs, rms_raw, rms_cen, rgs = [], [], [], []
    it = iter(dm.train_dataloader())
    while len(offs) < args.n:
        try:
            raw = next(it)
        except StopIteration:
            break
        b = mod._prepare(raw, train=False)
        L = b["mask"].shape[1]
        x = b["atom_pos"].reshape(-1, L * 14, 3)
        am = b["atom_mask"]
        for j in range(x.shape[0]):
            if len(offs) >= args.n:
                break
            m = am[j].bool()
            if int(m.sum()) < 100:
                continue
            p = x[j][m].double().numpy()
            c = p.mean(0)
            offs.append(float(np.linalg.norm(c)))
            rms_raw.append(float(np.sqrt((p ** 2).sum(-1).mean())))
            rms_cen.append(float(np.sqrt(((p - c) ** 2).sum(-1).mean())))
            rgs.append(float(np.sqrt(((p - c) ** 2).sum(-1).mean())))

    o, rr, rc = np.array(offs), np.array(rms_raw), np.array(rms_cen)
    print(f"\nchains: {len(o)}   (these are batch['atom_pos'], the ACTUAL diffusion targets)\n")
    print(f"{'quantity':>34} {'mean':>9} {'median':>9} {'p95':>9} {'max':>9}")
    for nm, v in (("|centroid| offset from origin (A)", o),
                  ("RMS |x| as trained, UNcentred (A)", rr),
                  ("RMS |x| if centred (A)", rc)):
        print(f"{nm:>34} {v.mean():>9.2f} {np.median(v):>9.2f} "
              f"{np.percentile(v, 95):>9.2f} {v.max():>9.2f}")

    print(f"\nSIGMA_DATA in use: {SIGMA_DATA}")
    print(f"  scale implied by the UNCENTRED data : {rr.mean():.1f} A  "
          f"-> off by {rr.mean()/SIGMA_DATA:.2f}x")
    print(f"  scale implied by CENTRED data       : {rc.mean():.1f} A  "
          f"-> off by {rc.mean()/SIGMA_DATA:.2f}x")
    print("\n⛔ EDM's c_in/c_skip/c_out are all functions of SIGMA_DATA. If the true data scale does")
    print("   not match it, every one of those constants is miscalibrated, and the noise schedule")
    print("   covers the wrong range.")
    print("⛔ Training targets sit at a random point on a sphere of the radius above; inference")
    print("   starts centred at the origin and re-centres every step. That is a train/inference")
    print("   mismatch in the single degree of freedom the loss cannot see (it aligns translation")
    print("   away), so it is invisible in val/loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
