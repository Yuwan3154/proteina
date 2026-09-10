"""D19 mechanism: for a chain whose template length differs, compare the SOURCES directly.

The index says 15.2% of template rows cover a different number of residues than their native, and
gaps do not explain it. Candidate mechanism: the templates were generated from openfold's version of
the chain (slim_struct_train.list) while the native row is built from proteina's processed .pt, and
the two pipelines select different residue sets for the same PDB chain.

This takes the worst offenders and prints, per chain, the residue count of (a) the proteina processed
.pt, (b) the template npz on disk, and (c) the index's stored runs -- so the disagreement is pinned
to a source, not inferred.
"""

import argparse
import json
import pathlib
import zlib

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.datasets.sse_topology import DSSP_GAP

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--processed-dir", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--trees", nargs="+", required=True)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()

idx = torch.load(a.index, map_location="cpu", weights_only=False)
ids = idx["ids"]
runs_flat, runs_off = idx["runs_flat"], idx["runs_offset"]
mem_flat, mem_off = idx["members_offset"], idx["members_flat"]
mem_off, mem_flat = idx["members_offset"], idx["members_flat"]
native_rows = torch.nonzero(idx["row_is_native"]).flatten()
manifest = json.load(open(a.manifest))


def total(r):
    b = runs_flat[runs_off[r]:runs_off[r + 1]]
    return 0 if b.numel() == 0 else int(b[:, 1].long().sum())


# worst offenders by |template - native|
worst = []
for c in range(min(int(mem_off.numel()) - 1, 4000)):
    rows = mem_flat[mem_off[c]:mem_off[c + 1]].tolist()
    if not rows:
        continue
    n_tot = total(int(native_rows[c]))
    if n_tot == 0:
        continue
    for r in rows:
        d = total(r) - n_tot
        if d:
            worst.append((abs(d), d, c, r, n_tot))
worst.sort(reverse=True)

print(f"{'chain':>14} {'idx_native':>10} {'idx_tpl':>8} {'diff':>6} {'processed.pt':>13} {'npz_atom_mask':>14}")
seen = set()
shown = 0
for _, d, c, r, n_tot in worst:
    stem = str(ids[int(native_rows[c])]).split("@")[0]
    if stem in seen:
        continue
    seen.add(stem)
    pt_len = -1
    p = _processed_path_sharded(pathlib.Path(a.processed_dir), stem, manifest)
    if p.exists():
        g = torch.load(str(p), map_location="cpu", weights_only=False)
        pt_len = int(g.coords.shape[0])
    npz_len = -1
    shard = f"shard{zlib.crc32(stem.encode()) % 1000:04d}"
    for tree in a.trees:
        f = pathlib.Path(tree) / shard / f"{stem}.npz"
        if not f.exists():
            f = pathlib.Path(tree) / f"{stem}.npz"
        if f.exists():
            z = np.load(f)
            if "atom_mask" in z:
                npz_len = int(z["atom_mask"].shape[0])
            break
    print(f"{stem:>14} {n_tot:>10} {total(r):>8} {d:>+6} {pt_len:>13} {npz_len:>14}")
    shown += 1
    if shown >= a.n:
        break

print("\nidx_native / processed.pt disagreeing => the NATIVE row is not proteina's chain")
print("idx_tpl / npz_atom_mask disagreeing    => the TEMPLATE row is not the npz on disk")
print("processed.pt / npz_atom_mask disagreeing => the two PIPELINES chose different residue sets")
