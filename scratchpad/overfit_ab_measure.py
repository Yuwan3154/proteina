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

from steps_sweep import MODEL_CFG, helix_pos_frac  # noqa: E402

from proteinfoundation.nn.af3_diffusion import MINI_ROLLOUT_STEPS  # noqa: E402
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer  # noqa: E402
from proteinfoundation.utils.c2c_dump import handedness_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir holding overfit_batch.pt")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints to score")
    ap.add_argument("--k", type=int, default=128, help="rollouts per checkpoint (SE ~4.4%% at p=0.5)")
    ap.add_argument("--steps", type=int, default=MINI_ROLLOUT_STEPS)
    # EMA(0.999) is still 0.999^N of the INIT after N steps: 61% at 500, 13% at 2000. For a short
    # overfit run the EMA is a blur of the whole trajectory, so RAW weights are the primary readout
    # and EMA is reported beside them rather than instead of them.
    ap.add_argument("--weights", choices=["raw", "ema", "both"], default="both")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    raw = torch.load(os.path.join(args.run, "overfit_batch.pt"), weights_only=False).to(dev)
    model = ContactToCoordTrainer(model_cfg=dict(MODEL_CFG, n_diffusion_samples=8)).to(dev).eval()
    b = model._prepare(raw, train=False)
    L = b["mask"].shape[1]
    keep = b["mask"][0].bool()
    gt = b["atom_pos"].reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
    # ⛔ The CA-dihedral criterion (helix_pos_frac > 0.5) is a coin flip on beta-rich chains: the
    # NATIVE 6kn9_B scores 0.490. For a single KNOWN target the proper-vs-reflected superposition
    # gap is exact, so `is_mirrored` from handedness_metrics (p > 2r and p - r > 1 A, the same
    # operational definition the validation dump logs) is the primary criterion; helix_pos is
    # reported beside it only for continuity with the 254-chain sweeps.
    nat = helix_pos_frac(gt)
    print(f"[data] pinned structure: L={int(keep.sum())} (padded {L}), {args.k} rollouts x "
          f"{args.steps} steps per checkpoint | native helix_pos_frac={nat:.3f} "
          f"({'UNRELIABLE for this chain' if 0.3 < nat < 0.7 else 'usable'})", flush=True)

    print(f"\n{'step':>7} {'wts':>4} {'mirrored':>10} {'+-SE':>6} {'rmsd_proper_notmir':>18} "
          f"{'rmsd_refl_of_mir':>16} {'n_notmir':>9} {'n_mir':>6} {'helix_pos>.5':>13}")
    variants = ["raw", "ema"] if args.weights == "both" else [args.weights]
    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        step = ck.get("global_step")
        for wts in variants:
            if wts == "ema":
                assert "ema" in ck, "checkpoint carries no EMA"
                model.model.load_state_dict(ck["ema"]["params"], strict=True)
            else:
                sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
                model.model.load_state_dict(sd, strict=True)
            hands, rp, rr, mirs = [], [], [], []
            with torch.no_grad():
                s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
                for i in range(args.k):
                    torch.manual_seed(31_000 + i)
                    c = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                            b["atom_to_token"], b["atom_mask"], b["ref_space_uid"],
                                            n_steps=args.steps)
                    ca = c.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
                    hm = handedness_metrics(ca, gt)
                    hands.append(hm.get("helix_pos_frac", float("nan")))
                    rp.append(hm["rmsd_proper"])
                    rr.append(hm["rmsd_reflected"])
                    mirs.append(hm["is_mirrored"])
            h, rp, rr, mirs = np.array(hands), np.array(rp), np.array(rr), np.array(mirs)
            mir = mirs > 0.5          # reflection fits DISTINCTLY better than any proper rotation
            n = len(mirs)
            p = float(mir.sum()) / n
            se = float(np.sqrt(p * (1 - p) / n))
            hp = float(np.mean(h[np.isfinite(h)] > 0.5)) if np.isfinite(h).any() else float("nan")
            print(f"{step:7d} {wts:>4} {100*p:9.1f}% {100*se:5.1f} "
                  f"{np.median(rp[~mir]) if (~mir).sum() else float('nan'):18.3f} "
                  f"{np.median(rr[mir]) if mir.sum() else float('nan'):16.3f} "
                  f"{int((~mir).sum()):9d} {int(mir.sum()):6d} {100*hp:12.1f}%", flush=True)
    print("\n  mirrored: is_mirrored = proper RMSD > 2x the reflection-allowed RMSD AND gap > 1 A")
    print("  (the validation dump's operational definition; exact for a single KNOWN target).")
    print("  rmsd_proper_notmir: median proper-Kabsch CA-RMSD of the NON-mirrored samples --")
    print("  ~1 A means folded and right-handed, ~20 A means unfolded garbage.")
    print("  rmsd_refl_of_mir: median CA-RMSD of the MIRRORED samples once a reflection is allowed.")
    print("  helix_pos>.5: the CA-dihedral criterion, continuity only -- unreliable on beta-rich chains.")

if __name__ == "__main__":
    main()
