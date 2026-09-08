"""Mirror rate of a ONE-STRUCTURE overfit run: K rollouts of the pinned chain, per checkpoint.

The A/B is smooth_lddt ON vs OFF (user directive 2026-09-08): the LDDT term is distance-only and
therefore reflection-invariant, while the chiral MSE is EDM-weighted down to ~1/sd^2 at high sigma.
If the achiral term is what lets the sampler stay reflection-equivariant, the OFF arm should learn
the hand of its single training structure and the ON arm should not.

Reads the exact chain the run trained on from <run>/overfit_batch.pt, so there is no risk of scoring
a different chain than the one that was pinned. Same seeds and step count as steps_sweep.py.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from steps_sweep import MODEL_CFG, helix_pos_frac, kabsch_rmsd  # noqa: E402

from proteinfoundation.nn.af3_diffusion import MINI_ROLLOUT_STEPS  # noqa: E402
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer  # noqa: E402
from proteinfoundation.utils.c2c_dump import _kabsch_rmsd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir holding overfit_batch.pt")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints to score")
    ap.add_argument("--k", type=int, default=128, help="rollouts per checkpoint (SE ~4.4%% at p=0.5)")
    ap.add_argument("--steps", type=int, default=MINI_ROLLOUT_STEPS)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    raw = torch.load(os.path.join(args.run, "overfit_batch.pt"), weights_only=False).to(dev)
    model = ContactToCoordTrainer(model_cfg=dict(MODEL_CFG, n_diffusion_samples=8)).to(dev).eval()
    b = model._prepare(raw, train=False)
    L = b["mask"].shape[1]
    keep = b["mask"][0].bool()
    gt = b["atom_pos"].reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
    print(f"[data] pinned structure: L={int(keep.sum())} (padded {L}), {args.k} rollouts x "
          f"{args.steps} steps per checkpoint", flush=True)

    print(f"\n{'step':>7} {'mirrored':>10} {'+-SE':>6} {'rmsd_unmir_med':>15} "
          f"{'rmsd_refl_of_mir_med':>21} {'n_unmir':>8} {'n_mir':>6}")
    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert "ema" in ck, "refusing to score the unaveraged model"
        model.model.load_state_dict(ck["ema"]["params"], strict=True)
        step = ck.get("global_step")
        hands, rp, rr = [], [], []
        with torch.no_grad():
            s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
            for i in range(args.k):
                torch.manual_seed(31_000 + i)
                c = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                        b["atom_to_token"], b["atom_mask"], b["ref_space_uid"],
                                        n_steps=args.steps)
                ca = c.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
                hands.append(helix_pos_frac(ca))
                rp.append(kabsch_rmsd(ca, gt))
                rr.append(_kabsch_rmsd(ca, gt, allow_reflection=True))
        h, rp, rr = np.array(hands), np.array(rp), np.array(rr)
        ok = ~np.isnan(h)
        mir = ok & (h > 0.5)
        unmir = ok & (h <= 0.5)
        p = float(mir.sum()) / max(int(ok.sum()), 1)
        se = float(np.sqrt(p * (1 - p) / max(int(ok.sum()), 1)))
        print(f"{step:7d} {100*p:9.1f}% {100*se:5.1f} "
              f"{np.median(rp[unmir]) if unmir.sum() else float('nan'):15.3f} "
              f"{np.median(rr[mir]) if mir.sum() else float('nan'):21.3f} "
              f"{int(unmir.sum()):8d} {int(mir.sum()):6d}", flush=True)
    print("\n  rmsd_unmir_med: proper-Kabsch CA-RMSD of the right-handed samples to the pinned chain.")
    print("  rmsd_refl_of_mir_med: CA-RMSD of the MIRRORED samples once a reflection is allowed --")
    print("  small means they are accurate mirror images, large means they are simply wrong.")


if __name__ == "__main__":
    main()
