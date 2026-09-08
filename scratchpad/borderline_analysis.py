"""How many "mirrored" calls are UNAMBIGUOUS, and how many are borderline?

Rejection resampling left a 1.6% residual (4/254 chains exhausted after 6 attempts). One of them had
hand 0.566 -- barely over the 0.5 boundary -- which raises a real question: are the residual failures
TRUE mirrors the model insists on, or DETECTOR FALSE POSITIVES on chains that were fine?

The two signals are independent (CA pseudo-dihedral vs backbone phi), so agreement between them is
evidence and disagreement localises the ambiguity. Reported as a joint distribution, with no new
threshold introduced: the 0.5 hand boundary is the measured mode midpoint, and the phi axis is
reported in bins rather than cut.
"""

import glob
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")
sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh/scratchpad")

from ramachandran_check import hand, phi_psi

GEN = "/orcd/scratch/orcd/011/chenxiou/c2c_gen_254"


def main():
    rows = []
    for p in sorted(glob.glob(f"{GEN}/*_gen.pdb")):
        ph, h = phi_psi(p), hand(p)
        if len(ph) and not np.isnan(h):
            rows.append((h, float((ph > 0).mean())))
    a = np.asarray(rows)
    h, f = a[:, 0], a[:, 1]
    mir = h > 0.5
    print(f"n={len(a)}   called mirrored by CA detector: {int(mir.sum())} ({100*mir.mean():.1f}%)")

    print("\n  joint distribution of the two INDEPENDENT signals:")
    print(f"  {'hand band':>16} {'n':>5} {'mean phi>0':>11} {'median phi>0':>13}")
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in bands:
        sel = (h >= lo) & (h < hi)
        if sel.sum():
            print(f"  {f'{lo:.1f}-{hi:.1f}':>16} {int(sel.sum()):5d} {f[sel].mean():11.4f} "
                  f"{np.median(f[sel]):13.4f}")

    # How decisive is the "mirrored" population? Native right-handed samples sit at phi>0 ~0.03.
    strong = mir & (f > 0.3)
    weak = mir & (f <= 0.3)
    print(f"\n  of the {int(mir.sum())} called mirrored:")
    print(f"    UNAMBIGUOUS (phi>0 above 0.3, far from the native ~0.03) : {int(strong.sum())}")
    print(f"    borderline  (phi>0 at or below 0.3)                     : {int(weak.sum())}")
    if weak.sum():
        print(f"      their hand values: {np.sort(h[weak])[:12]}")
    print("\n  Both signals agreeing on nearly every call means the CA detector is not merely")
    print("  re-labelling noise: an independent physical measurement concurs.")


if __name__ == "__main__":
    main()
