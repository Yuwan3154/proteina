"""Does ANDing the Ramachandran signal with the CA detector remove its false positives?

The CA pseudo-dihedral detector calls 5/254 NATIVE chains mirrored (2.0% false-positive rate). Those
5 sit at frac-phi>0 = 0.0730, far below true mirrors (0.4397) and near the native rate (0.0287), so a
second criterion should veto them.

⛔ REPORTED AS A THRESHOLD SWEEP, deliberately. Committing to one phi cutoff would be an invented
value; sweeping shows the improvement across a wide band and lets the reader see how insensitive it
is. Nothing downstream depends on a particular choice.

Ground truth for "is this chain really mirrored" is taken to be the CA detector's call on
GENERATED chains (the population whose 51.8% rate is established) and "not mirrored" for NATIVES,
which are real deposited proteins. So the two error rates below are:
  FP = natives wrongly called mirrored   (should go to 0)
  TP = generated-mirrored still caught   (should stay at 131)
"""

import glob
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")
sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh/scratchpad")

from ramachandran_check import ca_trace, hand, phi_psi

GEN = "/orcd/scratch/orcd/011/chenxiou/c2c_gen_254"


def collect(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        ph = phi_psi(p)
        h = hand(p)
        if len(ph) and not np.isnan(h):
            out.append((h, float((ph > 0).mean())))
    return np.asarray(out)


def main():
    nat = collect(f"{GEN}/*_gt.pdb")
    gen = collect(f"{GEN}/*_gen.pdb")
    nat_ca = nat[:, 0] > 0.5
    gen_ca = gen[:, 0] > 0.5
    print(f"natives   n={len(nat)}   CA-detector calls mirrored: {int(nat_ca.sum())} "
          f"({100*nat_ca.mean():.1f}%  <- FALSE POSITIVES)")
    print(f"generated n={len(gen)}   CA-detector calls mirrored: {int(gen_ca.sum())} "
          f"({100*gen_ca.mean():.1f}%)")
    print(f"\n{'phi_cut':>8} {'native FP':>11} {'gen flagged':>13} {'notes':>8}")
    print("-" * 46)
    for cut in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        nfp = int((nat_ca & (nat[:, 1] > cut)).sum())
        gfl = int((gen_ca & (gen[:, 1] > cut)).sum())
        print(f"{cut:8.2f} {nfp:11d} {gfl:13d} {'':>8}")
    print(f"\nCA-detector alone: native FP {int(nat_ca.sum())}, generated flagged {int(gen_ca.sum())}")
    print("⛔ MEASURED TRADE-OFF, not a free win. Removing all 5 native false positives needs a cut")
    print("   of ~0.20, and that also drops true detections from 131 to 123. At 0.10 one FP survives.")
    print("   So the combined detector buys SPECIFICITY at the cost of SENSITIVITY -- for rejection")
    print("   resampling a false positive only costs one extra rollout, while a false negative ships")
    print("   a mirrored structure, so the CA detector alone is the better operating point.")
    print("   The Ramachandran signal's real value is as INDEPENDENT PHYSICAL EVIDENCE that mirrored")
    print("   samples are strained/invalid, and as grounding for a training term -- not as an upgrade")
    print("   to the detector.")


if __name__ == "__main__":
    main()
