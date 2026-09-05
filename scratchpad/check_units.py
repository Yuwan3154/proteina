"""Decide the coordinate units empirically, not from config comments.

The c2c model mixes two sources of geometry:
  - atom_pos, from the proteina dataset (batch["coords"]);
  - ref_pos, from openfold's residue_constants via atom14_features.
Every AF3 constant in af3_diffusion (SIGMA_DATA=16, S_MAX=160, S_TRANS=1, the smooth-lDDT
thresholds 0.5/1/2/4 and its 15 cutoff) is an ANGSTROM value. If either source is nanometres,
those constants are off by 10x and the whole EDM parameterisation is mis-scaled.

Ground truth used here: a peptide bond N-CA is 1.458 A and consecutive CA-CA is 3.80 A
(Engh & Huber 1991). Both appear below in whatever units the arrays actually carry.
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.atom_features import atom14_features
from proteinfoundation.openfold_stub.np import residue_constants as rc

DATA = os.environ.get("DATA_PATH", "/orcd/pool/006/chenxiou/proteina/data")


def main():
    print("=== ref_pos, from residue_constants ===")
    aatype = torch.zeros(1, 4, dtype=torch.long)          # ALA
    mask = torch.ones(1, 4)
    _, ref_pos, _, _ = atom14_features(aatype, mask)
    r = ref_pos.reshape(1, 4, 14, 3)[0, 0]
    n_ca = (r[0] - r[1]).norm().item()
    ca_c = (r[1] - r[2]).norm().item()
    print(f"  ALA N-CA = {n_ca:.4f}   CA-C = {ca_c:.4f}")
    ref_unit = "ANGSTROM" if 1.2 < n_ca < 1.7 else ("NANOMETRE" if 0.12 < n_ca < 0.17 else "???")
    print(f"  => ref_pos is {ref_unit}   (N-CA is 1.458 A = 0.1458 nm)")

    print("\n=== batch['coords'], from a real processed .pt ===")
    hits = sorted(glob.glob(os.path.join(DATA, "pdb_train", "processed", "*.pt")))[:1]
    if not hits:
        hits = sorted(glob.glob(os.path.join(DATA, "pdb_train", "**", "*.pt"), recursive=True))[:1]
    if not hits:
        print("  no .pt found; cannot check")
        return 2
    d = torch.load(hits[0], map_location="cpu", weights_only=False)
    print(f"  file: {os.path.basename(hits[0])}")
    coords = d["coords"] if isinstance(d, dict) else getattr(d, "coords")
    print(f"  coords shape {tuple(coords.shape)}")
    ca = coords[:, 1, :] if coords.dim() == 3 else coords
    step = (ca[1:] - ca[:-1]).norm(dim=-1)
    med = step.median().item()
    extent = (ca.max(0).values - ca.min(0).values).norm().item()
    print(f"  median consecutive CA-CA = {med:.4f}")
    print(f"  bounding-box diagonal    = {extent:.2f}  (L={ca.shape[0]})")
    data_unit = "ANGSTROM" if 3.0 < med < 4.5 else ("NANOMETRE" if 0.30 < med < 0.45 else "???")
    print(f"  => coords are {data_unit}   (consecutive CA-CA is 3.80 A = 0.380 nm)")

    print("\n=== verdict ===")
    print(f"  ref_pos={ref_unit}  coords={data_unit}")
    if ref_unit != data_unit:
        print("  ⛔ MISMATCH: the two geometry sources are in different units.")
    if data_unit == "NANOMETRE":
        print("  ⛔ AF3 constants (SIGMA_DATA=16, S_MAX=160, smooth_lddt 0.5/1/2/4/15) are")
        print("     ANGSTROM values applied to nanometre data -- every length scale is 10x off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
