"""How much does the CB-8A contact map actually MOVE when the glycine fill switches from AF2-style
CA-substitution to the RoseTTAFold virtual CB?

⛔ `scratchpad/test_pseudo_cb.py` already proved the virtual CB lands 0.056 A from a real CB. That is
a per-ATOM number. A consumer of the contact map (ContactEBM, the c2c conditioning, the tri target)
never sees atoms -- it sees BINARY PAIRS. This measures the pair-level delta, which is the number
that says whether a downstream model needs a fine-tune or a retrain.

⛔ Runs the REAL `ContactMapTransform._contact_map_from_distance` under both settings, on real
processed chains. Both maps come from the same graph object, so the ONLY difference between them is
the fill rule.

⛔ CPU-only and deliberately so -- it is pure coordinate arithmetic and must not sit in the GPU queue.

Run: python scratchpad/measure_cb_flip.py --n 200
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.transforms import ContactMapTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed_parking")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--cutoff", type=float, default=8.0)
args = ap.parse_args()

# ⛔ `processed/` is a SYMLINK to a prefix-sharded tree (00/, 01/, ...) with zero .pt at the top
# level, so a flat glob silently returns nothing and the run would report on an empty sample.
files = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
if not files:
    files = sorted(glob.glob(os.path.join(args.dir, "**", "*.pt"), recursive=True))
assert files, f"no .pt under {args.dir} (flat or recursive)"
# deterministic stride sample -- no RNG, so a re-run reproduces the same chains exactly
step = max(1, len(files) // args.n)
sample = files[::step][: args.n]
print(f"[data] {len(files)} chains available, sampling {len(sample)} (stride {step})")

t_ca = ContactMapTransform(contact_atom_type="CB", contact_distance_cutoff=args.cutoff, cb_fill="ca")
t_pcb = ContactMapTransform(contact_atom_type="CB", contact_distance_cutoff=args.cutoff,
                            cb_fill="pseudo_cb")

tot_ca = tot_pcb = tot_flip = tot_gain = tot_lose = 0
tot_res = tot_nocb = 0
per_chain = []
shifts = []
skipped = []

for path in sample:
    g = torch.load(path, weights_only=False)
    if not hasattr(g, "coords") or not hasattr(g, "coord_mask"):
        skipped.append((os.path.basename(path), "no coords/coord_mask"))
        continue
    g.coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
    g.coord_mask = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].float()
    L = g.coords.shape[0]
    if L < 10:
        skipped.append((os.path.basename(path), f"L={L}"))
        continue

    m_ca = t_ca._contact_map_from_distance(g).bool()
    m_pcb = t_pcb._contact_map_from_distance(g).bool()
    off = ~torch.eye(L, dtype=torch.bool)

    n_ca = int((m_ca & off).sum()) // 2
    n_pcb = int((m_pcb & off).sum()) // 2
    flip = (m_ca != m_pcb) & off
    n_flip = int(flip.sum()) // 2
    n_gain = int(((~m_ca) & m_pcb & off).sum()) // 2
    n_lose = int((m_ca & (~m_pcb) & off).sum()) // 2

    nocb = g.coord_mask[:, 3] < 0.5
    if int(nocb.sum()):
        ca_pts = g.coords[:, 3, :].clone()
        ca_pts[nocb] = g.coords[nocb, 1, :]
        pcb_pts = ContactMapTransform._fill_missing_cb_pseudo(
            g.coords[:, 3, :].clone(), g.coords, g.coord_mask, nocb)
        shifts.append((pcb_pts[nocb] - ca_pts[nocb]).norm(dim=-1).numpy())

    tot_ca += n_ca; tot_pcb += n_pcb; tot_flip += n_flip
    tot_gain += n_gain; tot_lose += n_lose
    tot_res += L; tot_nocb += int(nocb.sum())
    per_chain.append(n_flip / max(n_ca, 1))

# ⛔ Every skip is printed, not just counted -- a count is not evidence that the skips were benign.
print(f"[skips] {len(skipped)} of {len(sample)}")
for name, why in skipped[:20]:
    print(f"    {name}: {why}")

n_used = len(per_chain)
assert n_used > 0, "zero chains scored -- the measurement is vacuous, not clean"
pc = np.array(per_chain)
sh = np.concatenate(shifts) if shifts else np.array([])

print(f"\n=== CB-8A contact map: AF2 CA-fill vs RoseTTAFold pseudo-CB ===")
print(f"chains scored              : {n_used}")
print(f"residues                   : {tot_res}   without CB (GLY/unresolved): {tot_nocb} "
      f"({100.0*tot_nocb/max(tot_res,1):.2f}%)")
if sh.size:
    print(f"contact-centre shift (A)   : mean {sh.mean():.3f}  median {np.median(sh):.3f}  "
          f"min {sh.min():.3f}  max {sh.max():.3f}  n={sh.size}")
print(f"\ncontacts, pooled unordered pairs:")
print(f"  AF2 CA-fill              : {tot_ca}")
print(f"  RoseTTAFold pseudo-CB    : {tot_pcb}  ({tot_pcb-tot_ca:+d}, "
      f"{100.0*(tot_pcb-tot_ca)/max(tot_ca,1):+.2f}%)")
print(f"  flipped pairs            : {tot_flip}  (gained {tot_gain}, lost {tot_lose})")
print(f"  flipped / CA-fill pairs  : {100.0*tot_flip/max(tot_ca,1):.2f}%")
print(f"\nper-chain flip fraction    : mean {100*pc.mean():.2f}%  median {100*np.median(pc):.2f}%  "
      f"p95 {100*np.percentile(pc,95):.2f}%  max {100*pc.max():.2f}%")
