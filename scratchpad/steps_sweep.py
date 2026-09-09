"""Does the number of reverse-diffusion steps change the mirror rate or the accuracy?

⛔ THE SWEPT VALUES ARE GROUNDED, NOT INVENTED. 200 is AF3's published inference step count and is
already a named constant in this repo (`FULL_INFERENCE_STEPS`, af3_diffusion.py, cited to SI 3.7.1
and DeepMind diffusion_head.py:126). 50 is what every measurement in this investigation has used, and
20 is the repo's `MINI_ROLLOUT_STEPS` (SI 4.1). So the arms are: the mini-rollout, our working
setting, and AF3's full setting.

Two questions at once:
  1. MIRROR RATE. Expected null -- the sampler is reflection-equivariant along the whole trajectory
     and the commit-step probe showed the hand is already strongly set within the first 5 steps
     (|hand-0.5| = 0.2955 of a maximum 0.5). More steps refine geometry, not handedness.
  2. ACCURACY. Genuinely open and worth knowing regardless: we have been reporting every number in
     this investigation at 50 steps, and if AF3's 200 is materially better then the quoted CA-RMSDs
     understate the model.

⛔ Same chains and same seed in every arm, so only the step count differs.
⛔ RMSD is also reported over the NON-MIRRORED subset. Mirrored structures carry huge proper-Kabsch
RMSD, so a whole-set mean silently mixes an accuracy effect with a mirror-rate effect -- exactly the
confound that made the sigma-start sweep's RMSD column unreadable.
"""

import argparse
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# ⛔ The repo root comes from THIS file's location, never a hardcoded checkout: a fix-D checkpoint
# (to_hand_s) only loads against proteina_mirror, and a hardcoded proteina_sh silently imported the
# wrong model even when launched from the right checkout (job 22373592, strict-load failure).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import _ca_dihedrals

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)
ARMS = [20, 50, 200]   # MINI_ROLLOUT_STEPS, our working value, FULL_INFERENCE_STEPS


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(((a @ (u @ np.diag([1.0, 1.0, d]) @ vt) - b) ** 2).sum(-1).mean()))


def helix_pos_frac(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert "ema" in ck, "refusing to score the unaveraged model"
    # ⛔ Build the model from the checkpoint's OWN saved config, not this file's MODEL_CFG: a fix-D
    # run trained with p_mirror>0 must be measured with p_mirror>0, or rollout() injects no hand
    # label and the sweep silently scores the label-less model (the coin flip again).
    cfg = dict(ck["hyper_parameters"]["model_cfg"], n_diffusion_samples=8)
    print(f"[cfg] from checkpoint: p_mirror={cfg.get('p_mirror', 'n/a')} t_beta={cfg.get('t_beta', 'n/a')}",
          flush=True)
    model = ContactToCoordTrainer(model_cfg=cfg)
    model.model.load_state_dict(ck["ema"]["params"], strict=True)
    print(f"[load] EMA @ step {ck.get('global_step')}", flush=True)
    model = model.to(dev).eval()

    # ⛔ The validation epoch yields ONE CHAIN PER CLUSTER -- 254, not the 4158 Lightning prints.
    # Asking for more raises StopIteration mid-run (job 22258436 died that way at --n 400). Cap to
    # what actually exists rather than crashing, and say so.
    loader = dm.val_dataloader()
    avail = len(loader)
    n_use = min(args.n, avail)
    if n_use < args.n:
        print(f"[data] requested --n {args.n} but the val epoch holds {avail} chains; using {n_use}",
              flush=True)
    batches = []
    it = iter(loader)
    for _ in range(n_use):
        raw = next(it)
        b = model._prepare(raw, train=False)
        batches.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()})

    per_arm = {}
    print(f"\n{'steps':>7} {'mirrored':>10} {'rmsd_all_med':>13} {'rmsd_UNMIRRORED_med':>21} {'n_unmir':>8}")
    for steps in ARMS:
        hands, rmsds = [], []
        for i, b in enumerate(batches):
            L = b["mask"].shape[1]
            keep = b["mask"][0].bool()
            gt = b["atom_pos"].reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            with torch.no_grad():
                s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
                torch.manual_seed(31_000 + i)
                c = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                        b["atom_to_token"], b["atom_mask"], b["ref_space_uid"],
                                        n_steps=steps)
            ca = c.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            hands.append(helix_pos_frac(ca))
            rmsds.append(kabsch_rmsd(ca, gt))
        h, r = np.array(hands), np.array(rmsds)
        ok = ~np.isnan(h)
        unmir = ok & (h <= 0.5)
        per_arm[steps] = (h, r)
        print(f"{steps:7d} {100*float((h[ok] > 0.5).mean()):9.1f}% {np.median(r):13.3f} "
              f"{np.median(r[unmir]) if unmir.sum() else float('nan'):21.3f} {int(unmir.sum()):8d}",
              flush=True)

    # ⛔⛔ THE PAIRED COMPARISON IS THE ONLY FAIR ONE. Each arm's unmirrored subset contains a
    # DIFFERENT set of chains (membership, not just count, differs), so comparing medians across arms
    # compares different chain sets and is a selection confound, not an accuracy measurement.
    # Restrict to chains that came out unmirrored in EVERY arm, so all arms are scored on identical
    # chains, and report the per-chain paired difference against the 200-step reference.
    hs = np.stack([per_arm[s][0] for s in ARMS])
    rs = np.stack([per_arm[s][1] for s in ARMS])
    common = np.all(~np.isnan(hs) & (hs <= 0.5), axis=0)
    print(f"\n  PAIRED on the {int(common.sum())} chains unmirrored in ALL arms:")
    if common.sum() >= 5:
        ref = rs[ARMS.index(200)][common]
        for j, steps in enumerate(ARMS):
            v = rs[j][common]
            d = v - ref
            print(f"    steps {steps:4d}: rmsd_med {np.median(v):7.3f}   "
                  f"median delta vs 200 steps {np.median(d):+7.3f}   "
                  f"n_better_than_200 {int((d < 0).sum()):3d}/{int(common.sum())}")
    else:
        print("    too few chains unmirrored in every arm to compare -- report nothing.")
    print("\n  Expect the mirror column to be flat (the hand is set in the first few steps).")
    print("  The UNMIRRORED rmsd column is the honest accuracy comparison across step counts.")


if __name__ == "__main__":
    main()
