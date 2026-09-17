"""What weight should FAPE get, measured rather than guessed?

AF2 weights FAPE at 0.5 as its PRIMARY structure loss; AF3 has no FAPE at all and weights its
diffusion loss at 4.0. Neither number transfers, because the two terms have different natural
magnitudes on this model's outputs. So measure both on real batches at the live checkpoint and
report the weight that makes FAPE's CONTRIBUTION a given fraction of the diffusion term's.

⛔ Reports the raw terms and the implied weights; it does NOT pick one. The choice is the user's.

⭐ Also reports FAPE on a deliberately REFLECTED prediction, which is the number that matters: the
useful quantity is not FAPE's absolute size but the GAP between right-handed and mirrored at this
model's current error level -- the signal the fine-tune has to exploit.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import hydra
from omegaconf import OmegaConf

from proteinfoundation.proteinflow.contact2coord_trainer import (ALPHA_DIFFUSION,
                                                                 ContactToCoordTrainer)

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--n_batches", type=int, default=4)
ap.add_argument("--n_diff", type=int, default=8)
ap.add_argument("--fape_chunk", type=int, default=4)
args = ap.parse_args()

CFG_DIR = os.path.join(REPO, "configs", "datasets_config", "pdb")
assert os.path.isdir(CFG_DIR), f"config dir missing: {CFG_DIR}"
with hydra.initialize_config_dir(CFG_DIR, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name=args.dataset)
OmegaConf.set_struct(cfg_data, False)
cfg_data.datamodule.num_workers = 0
cfg_data.datamodule.prefetch_factor = None
dm = hydra.utils.instantiate(cfg_data.datamodule)
dm.setup("fit")

MODEL_CFG["n_diffusion_samples"] = args.n_diff
MODEL_CFG["diff_chunk"] = 0
MODEL_CFG["t_beta"] = (1.3, 2.0)
# w_fape=1.0 so the RAW term is logged; the weight is derived afterwards, not applied here.
model = ContactToCoordTrainer(model_cfg=MODEL_CFG, w_fape=1.0, fape_chunk=args.fape_chunk)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
if "ema" in ck:
    missing, unexpected = model.model.load_state_dict(
        {k: v for k, v in ck["ema"]["params"].items()}, strict=False)
    src = f"EMA (decay {ck['ema'].get('decay')})"
else:
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    src = "state_dict"
print(f"[load] {src}, step={ck.get('global_step')}, missing={len(missing)}, "
      f"unexpected={len(unexpected)}", flush=True)
assert len(missing) < 20, f"too many missing params ({len(missing)}) -- wrong MODEL_CFG?"

model = model.to("cuda").eval()

rows = []
it = iter(dm.val_dataloader())
for bi in range(args.n_batches):
    raw = next(it)
    with torch.no_grad():
        b = model._prepare(raw.to("cuda"), train=False)
        out = model.model(b)
        L = int(b["mask"].shape[1])
        f_ok = float(model._fape_loss(out["x_denoised"], out["x_gt_rep"],
                                      out["atom_mask_rep"], L))
        # ⭐ the same prediction, REFLECTED. Everything else identical, so the difference is
        # attributable to handedness and nothing else.
        xr = out["x_denoised"].reshape(-1, L, 14, 3).clone()
        xr[..., 2] = -xr[..., 2]
        f_mir = float(model._fape_loss(xr.reshape(out["x_denoised"].shape), out["x_gt_rep"],
                                       out["atom_mask_rep"], L))
        from proteinfoundation.nn.af3_diffusion import diffusion_loss
        dl, aux = diffusion_loss(out["x_denoised"], out["x_gt_rep"], out["sigma"],
                                 out["atom_mask_rep"], use_smooth_lddt=model.use_smooth_lddt)
        rows.append((f_ok, f_mir, float(dl.mean()), float((3.0 * aux["mse"]).sqrt().mean()),
                     float(out["sigma"].mean())))
    print(f"  batch {bi}: fape {f_ok:.4f}  fape(reflected) {f_mir:.4f}  "
          f"diffusion {rows[-1][2]:.4f}  rmsd {rows[-1][3]:.2f}", flush=True)

a = np.array(rows)
fape_m, fmir_m, dl_m, rmsd_m, sig_m = a.mean(0)
diff_contrib = ALPHA_DIFFUSION * dl_m

print(f"\n=== means over {len(a)} batches (n_diff={args.n_diff}) ===")
print(f"  FAPE (as predicted)      : {fape_m:.4f}")
print(f"  FAPE (prediction mirrored): {fmir_m:.4f}")
print(f"  ⭐ mirror GAP             : {fmir_m - fape_m:+.4f}")
print(f"  diffusion loss           : {dl_m:.4f}   x ALPHA_DIFFUSION {ALPHA_DIFFUSION} "
      f"= {diff_contrib:.4f}")
print(f"  denoising rmsd           : {rmsd_m:.2f} A   mean sigma {sig_m:.2f}")

print(f"\n=== w_fape that makes FAPE's contribution a given fraction of the diffusion term ===")
for frac in (0.1, 0.25, 0.5, 1.0, 2.0):
    print(f"  {frac:>4.2f} x diffusion -> w_fape = {frac * diff_contrib / max(fape_m, 1e-9):>8.3f}")
print("\n⚠️ FAPE is clamped into [0, 1]. If FAPE is already near 1 the model sits at the clamp and")
print("   the gradient is weak everywhere; that argues for fine-tuning from a GOOD checkpoint")
print("   (which is the plan) and, if it persists, for a smaller clamp rather than a bigger weight.")
