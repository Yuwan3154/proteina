"""Are the rejection-ACCEPTED structures physically sound, or merely right-handed?

Rejection resampling reports CA-RMSD 1.350 A median, but RMSD alone cannot tell whether the accepted
samples are stereochemically clean. If the detector is only selecting on a CA-dihedral statistic, it
could in principle pass strained structures that happen to score well on that one axis.

Checked against two INDEPENDENT references measured earlier on the same pipeline:
  natives, right-handed fold     : frac phi>0 = 0.0287, signed volume +2.8575, frac-L 0.9981
  c2c generated, right-handed    : frac phi>0 = 0.0303, signed volume +2.4428, frac-L 0.9990
  c2c generated, MIRRORED        : frac phi>0 = 0.4397, signed volume +1.7998, frac-L 0.9988
Accepted samples should land on the FIRST two rows, not the third.
"""

import glob
import sys

import numpy as np

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")
sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh/scratchpad")

from ramachandran_check import ca_trace, hand, phi_psi
from residue_chirality import signed_volumes

DIRS = [
    ("REJECTION-ACCEPTED", "/orcd/scratch/orcd/011/chenxiou/c2c_gen_reject254/*_gen.pdb"),
    ("baseline generated (all)", "/orcd/scratch/orcd/011/chenxiou/c2c_gen_254/*_gen.pdb"),
    ("natives", "/orcd/scratch/orcd/011/chenxiou/c2c_gen_254/*_gt.pdb"),
]


def main():
    for label, pat in DIRS:
        paths = sorted(glob.glob(pat))
        rows = []
        for p in paths:
            ph, h, v = phi_psi(p), hand(ca_trace(p)), signed_volumes(p)
            if len(ph) and not np.isnan(h) and len(v):
                rows.append((h, float((ph > 0).mean()), float(v.mean()), float((v > 0).mean())))
        a = np.asarray(rows)
        if not len(a):
            print(f"{label:>26}: NOTHING SCORED"); continue
        print(f"{label:>26}  n={len(a):4d}  mirrored {100*float((a[:,0]>0.5).mean()):5.1f}%  "
              f"phi>0 {a[:,1].mean():.4f}  signed_vol {a[:,2].mean():+7.4f}  "
              f"frac-L {a[:,3].mean():.4f}")
    print("\n  Accepted samples should match the NATIVE row on phi>0 and signed volume.")
    print("  If they match the mirrored profile instead, the detector is selecting on the wrong axis.")


if __name__ == "__main__":
    main()
