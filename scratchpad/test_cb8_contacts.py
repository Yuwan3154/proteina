"""Gate for the CB-8 Å contact definition (ContactEBM convention) in ContactMapTransform.

What must hold, or the retrain silently trains on a different definition than ContactEBM:
  1. CB atom where resolved, CA where the CB is NOT resolved (glycine or missing) -- decided by the
     atom mask, not by residue identity;
  2. `d <= 8.0` inclusive, symmetric, diagonal 1;
  3. bit-identical to ContactEBM's own formula (contact_targets.py::contact_map_cb, copied here);
  4. residues with no CA carry no contacts (ContactEBM masks those pairs; the fill value would
     otherwise collapse them onto one point);
  5. the c2c augmentation passes the diagonal through unchanged (1 for CB-8 maps, 0 for ConFind).
"""

import os
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.contact_augment import augment_contacts
from proteinfoundation.datasets.transforms import ContactMapTransform

PASS, FAIL = [], []
CA, CB = 1, 3


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def contactebm_cb_map(atom37, atom37_mask, cutoff=8.0):
    """ContactEBM/contact_ebm/data/contact_targets.py::cb_coords + contact_map_cb, verbatim."""
    cb = atom37[:, CB, :].clone()
    missing_cb = atom37_mask[:, CB] < 0.5
    cb[missing_cb] = atom37[missing_cb, CA, :]
    d = torch.cdist(cb.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
    return d <= cutoff


def random_graph(L, seed, gly_every=7, drop_cb_at=(), drop_ca_at=()):
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(L, 37, 3, generator=g) * 6.0
    mask = torch.ones(L, 37, dtype=torch.bool)
    residue_type = torch.randint(0, 20, (L,), generator=g)
    for i in (range(0, L, gly_every) if gly_every else ()):   # glycines: no CB atom
        residue_type[i] = 7
        mask[i, CB] = False
        coords[i, CB] = 1e-5                     # proteina's fill value for unresolved atoms
    for i in drop_cb_at:
        mask[i, CB] = False
        coords[i, CB] = 1e-5
    for i in drop_ca_at:
        mask[i, CA] = False
        coords[i, CA] = 1e-5
    return Data(coords=coords, coord_mask=mask, residue_type=residue_type)


def main():
    tf = ContactMapTransform(contact_atom_type="CB", contact_distance_cutoff=8.0, contact_method="distance")

    # 1+3: identical to ContactEBM's formula, including glycine/missing-CB fallback to CA
    g = random_graph(64, seed=0, drop_cb_at=(10, 11))
    ours = tf._contact_map_from_distance(g).bool()
    ref = contactebm_cb_map(g.coords, g.coord_mask.float())
    check("identical to ContactEBM formula (64 res, 9 Gly, 2 unresolved CB)", torch.equal(ours, ref),
          f"diff cells={int((ours != ref).sum())}")

    # 1: the fallback is decided by the MASK, not by residue identity -- a GLN (index 5) with a resolved
    # CB keeps its CB (the old code used residue_type == 5 as 'glycine')
    g = random_graph(20, seed=1, gly_every=None)
    g.residue_type[:] = 5
    g.coords[3, CB] = g.coords[3, CA] + torch.tensor([9.0, 0.0, 0.0])   # CB far from own CA
    g.coords[4, CA] = g.coords[3, CB] + torch.tensor([1.0, 0.0, 0.0])
    g.coords[4, CB] = g.coords[4, CA]
    ours = tf._contact_map_from_distance(g).bool()
    check("GLN with resolved CB uses its CB (mask-gated, not residue_type==5)", bool(ours[3, 4]))

    # 2: inclusive cutoff, symmetry, diagonal
    g = random_graph(6, seed=2, gly_every=None)
    g.coords[:, CB] = torch.zeros(6, 3)
    g.coords[1, CB] = torch.tensor([8.0, 0.0, 0.0])          # exactly 8.0 -> contact
    g.coords[2, CB] = torch.tensor([8.0 + 1e-3, 0.0, 0.0])   # just above -> no contact
    ours = tf._contact_map_from_distance(g).bool()
    check("d == 8.0 counts as a contact (inclusive)", bool(ours[0, 1]))
    check("d = 8.001 is not a contact", not bool(ours[0, 2]))
    check("symmetric", torch.equal(ours, ours.T))
    check("diagonal is 1", bool(torch.diagonal(ours).all()))

    # 4: a residue with no CA (and no CB) carries no contacts
    g = random_graph(30, seed=3, drop_ca_at=(5,), drop_cb_at=(5,))
    ours = tf._contact_map_from_distance(g).bool()
    check("residue without CA has an all-zero row and column", int(ours[5].sum()) == 0 and int(ours[:, 5].sum()) == 0)
    others = torch.ones(30, dtype=torch.bool); others[5] = False
    ref = contactebm_cb_map(g.coords, g.coord_mask.float())
    check("all other pairs still match ContactEBM", torch.equal(ours[others][:, others], ref[others][:, others]))

    # 5: augmentation passes the diagonal through (1 for CB-8) and keeps symmetry
    gen = torch.Generator().manual_seed(0)
    m = tf._contact_map_from_distance(random_graph(48, seed=4)).float()[None]
    mask = torch.ones(1, 48)
    for mode in ("balanced", "uniform"):
        a = augment_contacts(m, mask, rate=0.1, mode=mode, generator=gen)
        check(f"augment[{mode}] keeps the CB-8 diagonal at 1", bool((torch.diagonal(a[0]) == 1).all()))
        check(f"augment[{mode}] output symmetric", torch.equal(a[0], a[0].T))
    m0 = m.clone(); m0[0].fill_diagonal_(0.0)
    a = augment_contacts(m0, mask, rate=0.1, mode="balanced", generator=gen)
    check("augment keeps a zero (ConFind-style) diagonal at 0", float(torch.diagonal(a[0]).abs().sum()) == 0.0)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
