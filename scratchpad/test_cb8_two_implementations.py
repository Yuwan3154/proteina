"""CB-8 Å is implemented TWICE and the two must agree, or the model trains on one definition while
its topology references were built from another.

  transforms.ContactMapTransform._contact_map_from_distance  -- atom37 order (CB at index 3), used for
      the model's input contact map, after pdb_data reorders coords at __getitem__.
  precompute_synthetic_topology_index.cb8_contacts           -- ATOM_NUMBERING order (CB at index 4),
      used offline to build the SSE contact blocks stored in the index.

This runs both on the SAME real chains, through the same reorder the dataset performs, and asserts
they are bit-identical. It also reports the positive rate, which the plan recorded as ~0.076 for CB-8
(vs 0.042 for ConFind at threshold 0.01) -- a number the definition must reproduce on real data.

usage: python test_cb8_two_implementations.py --processed-dir <dir> --manifest <shard_manifest.json>
                                              --index <topology_index.pt> [--n 25]
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.datasets.transforms import ContactMapTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.precompute_synthetic_topology_index import cb8_contacts, CB_DISK_IDX

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", required=True, help="any topology index, only for its chain ids")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()

    manifest = json.load(open(a.manifest))
    idx = torch.load(a.index, map_location="cpu", weights_only=False, mmap=True)
    ids = [str(s) for s in idx["ids"]]
    rng = np.random.default_rng(0)
    tf = ContactMapTransform(contact_atom_type="CB", contact_distance_cutoff=8.0, contact_method="distance")

    n_cmp = n_diff = 0
    dens, dens_conf, lens = [], [], []
    for stem in rng.choice(ids, size=min(len(ids), a.n * 8), replace=False):
        if n_cmp >= a.n:
            break
        pt = _processed_path_sharded(pathlib.Path(a.processed_dir), str(stem), manifest)
        if not pt.exists():
            continue
        g = torch.load(str(pt), map_location="cpu", weights_only=False)
        if getattr(g, "coords", None) is None or getattr(g, "coord_mask", None) is None:
            continue
        # builder path: coords exactly as stored on disk
        m_builder = cb8_contacts(g.coords, g.coord_mask.bool(), CB_DISK_IDX)
        # transform path: the dataset's atom37 reorder first (pdb_data.__getitem__)
        g.coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        g.coord_mask = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        m_tf = tf._contact_map_from_distance(g)
        n_cmp += 1
        if not torch.equal(m_builder.bool(), m_tf.bool()):
            n_diff += 1
            print(f"    {stem}: {int((m_builder.bool() != m_tf.bool()).sum())} cells differ")
        L = m_tf.shape[0]
        lens.append(L)
        dens.append(float(m_tf.float().mean()))
        raw = getattr(g, "contact_map_confind", None)
        if raw is not None:
            dens_conf.append(float((raw.float() >= 0.01).float().mean()))

    check(f"both CB-8 implementations agree on {n_cmp} real chains", n_cmp >= 10 and n_diff == 0,
          f"{n_diff} of {n_cmp} chains differ")
    if dens:
        print(f"\n  CB-8 positive rate: mean {np.mean(dens):.4f} (median {np.median(dens):.4f}, "
              f"n={len(dens)}, L median {int(np.median(lens))})")
        check("CB-8 density is near the recorded 0.076", 0.05 <= float(np.mean(dens)) <= 0.11,
              f"{np.mean(dens):.4f}")
    if dens_conf:
        print(f"  ConFind@0.01 positive rate on the same chains: mean {np.mean(dens_conf):.4f}")
        check("CB-8 is denser than ConFind (recorded 0.076 vs 0.042)",
              float(np.mean(dens)) > float(np.mean(dens_conf)),
              f"{np.mean(dens):.4f} vs {np.mean(dens_conf):.4f}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
