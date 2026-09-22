"""ConFind vs CB-8A contact maps on the SAME processed chains, read through the REAL ContactMapTransform.

Answers, for a ConFind c2c twin of c2c_cb8_tbeta:
  1. what ConFind maps hold on the diagonal and at |i-j| = 1, 2 (CB-8A: diag 1 by construction)
  2. off-diagonal density of each definition on identical chains
  3. how many chains LACK contact_map_confind (ConFind mode raises -> __getitem__ skips the chain)
  4. whether the stored maps look like sparse CPU-ConFind output or dense Frame2ConFind probabilities

Run (CPU, mit_quicktest): python measure_confind_vs_cb8_maps.py --n 400
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

REPO = "/orcd/scratch/orcd/011/chenxiou/proteina_tri"
sys.path.insert(0, REPO)

from proteinfoundation.datasets.transforms import ContactMapTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed")
ap.add_argument("--n", type=int, default=400)
args = ap.parse_args()

# processed/ is a symlink to a prefix-sharded tree; a flat glob returns nothing
files = sorted(glob.glob(os.path.join(args.dir, "**", "*.pt"), recursive=True))
assert files, f"no .pt under {args.dir}"
step = max(1, len(files) // args.n)
sample = files[::step][: args.n]
print(f"[data] {len(files)} chains available, sampling {len(sample)} (stride {step})")

# thresholds/cutoffs copied from the two dataset yamls, not chosen here
t_cf = ContactMapTransform(contact_method="confind", confind_contact_threshold=0.01)
t_cb = ContactMapTransform(contact_method="distance", contact_atom_type="CB",
                           contact_distance_cutoff=8.0, cb_fill="pseudo_cb")

rows, missing, short = [], [], []
for path in sample:
    g = torch.load(path, weights_only=False)
    name = os.path.basename(path)
    raw = getattr(g, "contact_map_confind", None)
    if raw is None:
        missing.append(name)
        continue
    g.coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
    g.coord_mask = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].float()
    L = g.coords.shape[0]
    if L < 10:
        short.append((name, L))
        continue
    raw = torch.as_tensor(raw).float()
    cf = t_cf._contact_map_from_confind_precomputed(g).bool()
    cb = t_cb._contact_map_from_distance(g).bool()
    eye = torch.eye(L, dtype=torch.bool)
    off = ~eye
    sep1 = torch.diag(torch.ones(L - 1, dtype=torch.bool), 1)
    sep2 = torch.diag(torch.ones(L - 2, dtype=torch.bool), 2)
    rows.append(dict(
        L=L,
        cf_diag=float(cf[eye].float().mean()), cb_diag=float(cb[eye].float().mean()),
        raw_diag_max=float(raw[eye].max()),
        cf_sep1=float(cf[sep1].float().mean()), cb_sep1=float(cb[sep1].float().mean()),
        cf_sep2=float(cf[sep2].float().mean()), cb_sep2=float(cb[sep2].float().mean()),
        cf_dens=float(cf[off].float().mean()), cb_dens=float(cb[off].float().mean()),
        raw_nonzero=float((raw[off] > 0).float().mean()),
        raw_sub_thr=float(((raw[off] > 0) & (raw[off] < 0.01)).float().mean()),
        jacc=float((cf & cb & off).sum()) / max(1.0, float(((cf | cb) & off).sum())),
    ))

print(f"[missing contact_map_confind] {len(missing)} of {len(sample)}")
for n in missing:
    print(f"    {n}")
print(f"[short L<10] {len(short)}: {short[:20]}")
assert rows, "zero chains scored -- vacuous"
print(f"[scored] {len(rows)} chains")
for k in ["cf_diag", "cb_diag", "raw_diag_max", "cf_sep1", "cb_sep1", "cf_sep2", "cb_sep2",
          "cf_dens", "cb_dens", "raw_nonzero", "raw_sub_thr", "jacc"]:
    v = np.array([r[k] for r in rows])
    print(f"{k:>13s}  mean {v.mean():.4f}  median {np.median(v):.4f}  "
          f"min {v.min():.4f}  max {v.max():.4f}")
