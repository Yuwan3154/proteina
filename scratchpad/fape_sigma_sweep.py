"""Where in sigma does FAPE actually SEE the mirror?

Job 22875900 measured a mirror gap of only +0.0606, with mean sigma 512.66 and a denoising RMSD of
12.29 A. Both the right-handed and the mirrored value sat near FAPE's clamp ceiling of 1.0, so the
gap was squeezed shut by SATURATION, not by any property of FAPE. A weight fitted to that number
would be calibrated on a regime where the term carries no information.

This resolves it by reporting the gap PER SIGMA BIN instead of pooled, which decides between the two
available knobs:
  - if the gap opens up at low sigma -> restrict/weight FAPE toward low sigma;
  - if it stays shut everywhere      -> the 10 A clamp is too loose for this model's error scale.

⛔ Per-sample, not per-batch: calls _fape_pairs one diffusion sample at a time so each value can be
paired with ITS OWN sigma. Pooling first is what hid the effect in the previous probe.
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

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--n_batches", type=int, default=6)
ap.add_argument("--n_diff", type=int, default=48)
ap.add_argument("--clamps", default="10.0,5.0,2.0,1.0")
args = ap.parse_args()

CLAMPS = [float(v) for v in args.clamps.split(",")]

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
MODEL_CFG["diff_chunk"] = 8
MODEL_CFG["t_beta"] = (1.3, 2.0)
model = ContactToCoordTrainer(model_cfg=MODEL_CFG, w_fape=0.0)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
if "ema" in ck:
    missing, _ = model.model.load_state_dict(
        {k: v for k, v in ck["ema"]["params"].items()}, strict=False)
else:
    missing, _ = model.load_state_dict(ck["state_dict"], strict=False)
print(f"[load] step={ck.get('global_step')} missing={len(missing)}", flush=True)
assert len(missing) < 20, f"too many missing params ({len(missing)})"
model = model.to("cuda").eval()

rec = []            # (sigma, rmsd, {clamp: (fape_ok, fape_mirror)})
it = iter(dm.val_dataloader())
for bi in range(args.n_batches):
    raw = next(it)
    with torch.no_grad():
        b = model._prepare(raw.to("cuda"), train=False)
        out = model.model(b)
        L = int(b["mask"].shape[1])
        xp = out["x_denoised"].reshape(-1, L, 14, 3)
        xt = out["x_gt_rep"].reshape(-1, L, 14, 3)
        m = out["atom_mask_rep"].reshape(-1, L, 14)
        fm = (m[:, :, 0] > 0.5) & (m[:, :, 1] > 0.5) & (m[:, :, 2] > 0.5)
        am = m[:, :, 1] > 0.5
        sig = out["sigma"].reshape(-1).float().cpu().numpy()
        xm = xp.clone()
        xm[..., 2] = -xm[..., 2]
        S = xp.shape[0]
        assert len(sig) == S, f"sigma has {len(sig)} entries for {S} samples -- cannot pair them"
        for s in range(S):
            sl = slice(s, s + 1)
            per = {}
            for cl in CLAMPS:
                a_, n_ = model._fape_pairs(xp[sl], xt[sl], fm[sl], am[sl], cl, 10.0)
                c_, _ = model._fape_pairs(xm[sl], xt[sl], fm[sl], am[sl], cl, 10.0)
                per[cl] = (float(a_ / n_.clamp_min(1) / 10.0), float(c_ / n_.clamp_min(1) / 10.0))
            d = (xp[sl, :, 1] - xt[sl, :, 1])[am[sl]]
            rec.append((float(sig[s]), float(d.norm(dim=-1).mean()), per))
    print(f"  batch {bi} done ({len(rec)} samples)", flush=True)

sigs = np.array([r[0] for r in rec])
print(f"\nsigma range {sigs.min():.2f} - {sigs.max():.2f}, n={len(rec)}")
EDGES = [0, 1, 4, 16, 64, 256, 1024, np.inf]

for cl in CLAMPS:
    print(f"\n=== clamp = {cl} A ===")
    print(f"{'sigma bin':>16} {'n':>4} {'mean CA err':>12} {'FAPE':>8} {'FAPE mirr':>10} {'GAP':>9}")
    for a, b_ in zip(EDGES, EDGES[1:]):
        sel = [r for r in rec if a <= r[0] < b_]
        if not sel:
            continue
        ok = np.mean([r[2][cl][0] for r in sel])
        mi = np.mean([r[2][cl][1] for r in sel])
        er = np.mean([r[1] for r in sel])
        lab = f"{a:g}-{b_:g}" if np.isfinite(b_) else f">{a:g}"
        print(f"{lab:>16} {len(sel):>4} {er:>12.2f} {ok:>8.4f} {mi:>10.4f} {mi-ok:>+9.4f}")

print("\n⭐ Read the GAP column. It is the only quantity the fine-tune can exploit: the amount by")
print("   which FAPE prefers the correct hand over its mirror at that noise level and clamp.")
print("   A gap that is ~0 everywhere for clamp=10 but opens at a tighter clamp says the clamp is")
print("   the problem; a gap that only opens at low sigma says restrict FAPE to low sigma.")
