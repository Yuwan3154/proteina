"""Is the mirror invisible because the losses are IMBALANCED? Measure, per noise level.

Hypothesis under test: the chirality-blind terms (smooth-lDDT, distogram) swamp the one term that
can see handedness (the det=+1 aligned MSE), so the symmetry never gets broken.

⭐ The measurement is exact, not a simulation, because a perfect mirror is a perfect competitor: it
preserves EVERY pairwise distance, so smooth-lDDT and the distogram give it an IDENTICAL score to
the correct structure. Their discriminating margin is therefore exactly 0 by construction, and the
only question is how large the MSE margin is relative to the total loss the optimiser sees.

Structural detail that makes this worth checking: in `diffusion_loss`, smooth-lDDT is added OUTSIDE
the EDM weight, i.e. loss = w*mse + smooth_lddt with w = (s^2+sd^2)/(s*sd)^2. As s grows, w -> 1/sd^2
= 1/256, so at HIGH noise -- where the global fold and hence handedness are decided -- the
mirror-sensitive term is scaled down ~250x while the blind term stays at weight 1.

⛔ Reports the MARGIN (mirrored minus correct), not the loss value. A large loss that is equally
large for both hypotheses supplies no symmetry-breaking gradient at all.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import (
    SIGMA_DATA, diffusion_loss, sample_noise_level, smooth_lddt, weighted_rigid_align,
)

ALPHA_DIFFUSION = 4.0


def read_atoms(path):
    xs = []
    for line in open(path):
        if line.startswith("ATOM"):
            xs.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return torch.tensor(xs, dtype=torch.float32)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_chir/gen05_gt.pdb")
    args = ap.parse_args()

    x_gt = read_atoms(args.pdb)
    mask = torch.ones(x_gt.shape[:2])
    x_mir = x_gt.clone()
    x_mir[..., 2] *= -1.0                      # a perfect mirror: identical distances, wrong hand

    d_gt = torch.cdist(x_gt, x_gt)
    d_mi = torch.cdist(x_mir, x_mir)
    print(f"structure: {os.path.basename(args.pdb)}  atoms={x_gt.shape[1]}")
    print(f"max |pairwise distance difference| between the two hypotheses: "
          f"{(d_gt - d_mi).abs().max():.3e}   <- 0 means the blind terms CANNOT tell them apart\n")

    # What noise levels does training actually sample? sigma = 16*exp(-1.2 + 1.5*N(0,1)).
    torch.manual_seed(0)
    s = sample_noise_level((200000,), torch.device("cpu"))
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    qv = torch.quantile(s, qs)
    print("training sigma distribution: " +
          "  ".join(f"p{int(q*100)}={v:.2f}" for q, v in zip(qs.tolist(), qv.tolist())))
    print()

    print(f"{'sigma':>8} {'EDM w':>10} {'w*mse mirr':>11} {'w*mse corr':>11} {'MSE margin':>11} "
          f"{'lddt mirr':>10} {'lddt corr':>10} {'lddt margin':>12} {'signal frac':>12}")
    rows = []
    for sig in [0.25, 0.5, 1.0, 2.0, 4.8, 10.0, 20.0, 40.0, 80.0, 160.0]:
        sigma = torch.tensor([sig])
        out = {}
        for tag, x_pred in (("corr", x_gt), ("mirr", x_mir)):
            # Exactly the training loss path, term by term.
            aligned = weighted_rigid_align(x_pred, x_gt, mask, mask)
            err = ((x_pred - aligned) ** 2).sum(-1)
            mse = (err * mask).sum(1) / mask.sum(1).clamp_min(1e-8) / 3.0
            w = (sigma ** 2 + SIGMA_DATA ** 2) / (sigma * SIGMA_DATA) ** 2
            out[tag] = (float(w * mse), float(smooth_lddt(x_pred, aligned, mask)), float(w))
        wm_m, ld_m, w = out["mirr"]
        wm_c, ld_c, _ = out["corr"]
        margin_mse = wm_m - wm_c
        margin_lddt = ld_m - ld_c
        total_mirror = ALPHA_DIFFUSION * (wm_m + ld_m)
        frac = margin_mse * ALPHA_DIFFUSION / total_mirror if total_mirror > 0 else float("nan")
        rows.append((sig, w, margin_mse, margin_lddt, frac))
        print(f"{sig:>8.2f} {w:>10.4f} {wm_m:>11.3f} {wm_c:>11.3f} {margin_mse:>11.3f} "
              f"{ld_m:>10.4f} {ld_c:>10.4f} {margin_lddt:>12.2e} {100*frac:>11.1f}%")

    print("\nsignal frac = the share of the TOTAL loss at a mirrored prediction that actually")
    print("              discriminates against the mirror. The rest supplies gradient in other")
    print("              directions and is indifferent to handedness.")
    print("\n⚠️ lddt margin is ~0 at EVERY noise level, exactly as predicted: a reflection preserves")
    print("   every distance. The distogram term is blind for the same reason and is omitted.")
    lo = [r for r in rows if r[0] >= 10.0]
    hi = [r for r in rows if r[0] <= 1.0]
    if lo and hi:
        print(f"\n⭐ signal fraction at HIGH noise (sigma>=10, where the fold is decided): "
              f"{100*np.mean([r[4] for r in lo]):.1f}%")
        print(f"⭐ signal fraction at LOW noise  (sigma<=1, where the fold is already fixed): "
              f"{100*np.mean([r[4] for r in hi]):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
