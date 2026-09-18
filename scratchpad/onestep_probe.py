"""Can the c2c model produce a clean structure in ONE denoising step from pure noise?

⭐ The question: AF3-style diffusion modules often reconstruct most of a monomer at step 1 of 20 when
the trunk's constraints are good. If ours does too, a one-step (pure-noise -> clean) fine-tune with a
frame-based loss becomes a sensible direction. If it does not, that fine-tune would be teaching the
model something it currently cannot do at all, which is a much larger change than a fine-tune.

Method: ONE fixed validation batch, encoded ONCE, then rolled out at several step counts. The trunk
output (s, z) is therefore identical across all of them -- the ONLY thing that varies is the number
of denoising steps, so any difference is attributable to that and nothing else.

⛔ Scored with the same handedness_metrics the runs log, so the numbers are comparable to every
mirror figure already reported. Reports dist_mae (how well distances are satisfied) beside RMSD,
because the mirror work showed those two can move independently.
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
from proteinfoundation.utils.c2c_dump import handedness_metrics

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--steps", default="1,2,5,20,200")
ap.add_argument("--n_batches", type=int, default=2)
args = ap.parse_args()

STEPS = [int(v) for v in args.steps.split(",")]

# ⛔ hydra.initialize resolves its path relative to the CALLER'S FILE, not the cwd. The first version
# of this script lived outside the repo and passed "../proteina_tri/configs/...", which resolved to
# /orcd/scratch/orcd/011/proteina_tri/... -- the `chenxiou` component silently dropped. Job 22851194
# died on it AFTER copying a 3.2 GB checkpoint. Use an absolute dir derived from this file, and
# assert it exists so a bad path fails in milliseconds instead of after the copy.
CFG_DIR = os.path.join(REPO, "configs", "datasets_config", "pdb")
assert os.path.isdir(CFG_DIR), f"config dir missing: {CFG_DIR}"
with hydra.initialize_config_dir(CFG_DIR, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name=args.dataset)
OmegaConf.set_struct(cfg_data, False)
cfg_data.datamodule.num_workers = 0
cfg_data.datamodule.prefetch_factor = None
dm = hydra.utils.instantiate(cfg_data.datamodule)
dm.setup("fit")

model = ContactToCoordTrainer(model_cfg=MODEL_CFG)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
# ⛔ Prefer EMA: that is the weight set every reported metric was measured on. state_dict is the raw
# unaveraged model and would be a different thing from the numbers we are comparing against.
if "ema" in ck:
    sd = {k: v for k, v in ck["ema"]["params"].items()}
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    src = f"EMA (decay {ck['ema'].get('decay')})"
else:
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    src = "state_dict"
print(f"[load] {src} from {os.path.basename(args.ckpt)}; "
      f"step={ck.get('global_step')} missing={len(missing)} unexpected={len(unexpected)}")
assert len(missing) < 20, f"too many missing params ({len(missing)}) -- wrong MODEL_CFG?"

dev = "cuda"
model = model.to(dev).eval()

rows = {n: [] for n in STEPS}
it = iter(dm.val_dataloader())
for bi in range(args.n_batches):
    raw = next(it)
    b = model._prepare(raw.to(dev), train=False)
    L = b["mask"].shape[1]
    with torch.no_grad():
        # encode ONCE -- every step count then sees the identical trunk output
        s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
        gt_all = b["atom_pos"].reshape(-1, L, 14, 3)
        for n in STEPS:
            torch.manual_seed(1234 + bi)      # same starting noise across step counts
            coords = model.model.rollout(
                s, z, b["mask"], b["ref_feats"], b["ref_pos"], b["atom_to_token"],
                b["atom_mask"], b["ref_space_uid"], n_steps=n)
            gen_all = coords.reshape(-1, L, 14, 3)
            for j in range(gen_all.shape[0]):
                m = b["mask"][j].bool().cpu().numpy()
                g = gen_all[j, :, 1, :].float().cpu().numpy()[m]     # CA
                t = gt_all[j, :, 1, :].float().cpu().numpy()[m]
                if len(g) < 10:
                    continue
                h = handedness_metrics(g, t)
                if not h:
                    continue
                dm_ = float(np.abs(np.linalg.norm(g[:, None] - g[None], axis=-1)
                                   - np.linalg.norm(t[:, None] - t[None], axis=-1)).mean())
                # ⛔ is_mirrored needs proper > 2x reflected, so it CANNOT fire on a mediocre
                # structure where both superpositions land similarly -- exactly the regime
                # here (20 steps: proper 3.40 == reflected 3.40). refl-sign (proper >
                # reflected) is the sensitive read and is what every cross-run claim uses.
                rows[n].append((h["rmsd_proper"], h["rmsd_reflected"], dm_, h["is_mirrored"],
                                1.0 if h["rmsd_proper"] > h["rmsd_reflected"] else 0.0))
    print(f"  batch {bi} done")

print(f"\n{'steps':>6} {'n':>4} {'proper RMSD':>12} {'refl RMSD':>11} {'dist MAE':>10} "
      f"{'mirrored':>9} {'refl-sign':>10}")
for n in STEPS:
    a = np.array(rows[n])
    if not len(a):
        print(f"{n:>6} {'-':>4}")
        continue
    print(f"{n:>6} {len(a):>4} {a[:,0].mean():>12.2f} {a[:,1].mean():>11.2f} "
          f"{a[:,2].mean():>10.2f} {a[:,3].mean():>9.3f} {a[:,4].mean():>10.3f}")

base = np.array(rows[max(STEPS)])
one = np.array(rows[min(STEPS)])
if len(base) and len(one):
    print(f"\none-step vs {max(STEPS)}-step: proper RMSD {one[:,0].mean():.2f} vs {base[:,0].mean():.2f} A, "
          f"dist MAE {one[:,2].mean():.2f} vs {base[:,2].mean():.2f} A")
    print("VERDICT:", "one step is already close" if one[:, 2].mean() < 1.5 * base[:, 2].mean()
          else "one step is substantially worse than the full rollout")
