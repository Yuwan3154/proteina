"""What does the collapse actually look like? Read the validation sample dumps across steps.

Two runs have now degraded at ~the same STEP with different learning rates, which argues against
"lr too high". The dumped CA distance matrices say what the failure mode IS, at zero GPU cost:

  - radius of gyration vs ground truth  -> collapse to a point / blow-up to a gas
  - mean and max CA-CA distance         -> scale error
  - fraction of CA-CA below 3.0 A       -> steric collapse (real CA-CA is ~3.8 A minimum)
  - correlation of d_gen with d_gt      -> is the SHAPE right even when the SCALE is wrong?

That last one is the discriminator worth having: a model whose distances correlate well but are
mis-scaled is a very different (and much more fixable) problem from one that has lost the fold.
"""

import argparse
import glob
import os
import sys

import numpy as np


def rg_from_dist(d):
    """Radius of gyration from a pairwise distance matrix: Rg^2 = (1/2N^2) sum_ij d_ij^2."""
    n = d.shape[0]
    return float(np.sqrt((d ** 2).sum() / (2.0 * n * n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_v1/samples")
    ap.add_argument("--name", default="val00")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.root, "step*")))
    if not dirs:
        print(f"no sample dirs under {args.root}")
        return 2
    print(f"{'step':>8} {'L':>4} {'Rg_gen':>8} {'Rg_gt':>7} {'ratio':>6} "
          f"{'mean_gen':>9} {'mean_gt':>8} {'max_gen':>8} {'<3A %':>7} {'corr':>6}")
    for dpath in dirs:
        f = os.path.join(dpath, f"{args.name}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        dg, dt = z["dist_gen"].astype(np.float64), z["dist_gt"].astype(np.float64)
        n = dg.shape[0]
        iu = np.triu_indices(n, k=1)
        g, t = dg[iu], dt[iu]
        step = int(os.path.basename(dpath).replace("step", ""))
        rg_g, rg_t = rg_from_dist(dg), rg_from_dist(dt)
        frac_close = 100.0 * float((g < 3.0).mean())
        corr = float(np.corrcoef(g, t)[0, 1]) if g.std() > 1e-8 else float("nan")
        print(f"{step:>8} {n:>4} {rg_g:>8.2f} {rg_t:>7.2f} {rg_g/rg_t:>6.2f} "
              f"{g.mean():>9.2f} {t.mean():>8.2f} {g.max():>8.1f} {frac_close:>7.1f} {corr:>6.3f}")
    print("\nRg ratio ~1 = right size. <<1 = collapsed to a point. >>1 = blown apart.")
    print("corr is over the upper triangle: shape agreement independent of overall scale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
