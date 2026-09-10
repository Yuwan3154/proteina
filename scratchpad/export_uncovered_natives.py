"""List the chains whose 25%-identity CLUSTER has no T2 synthetic template, and write their all-atom
natives as PDB files (from the processed .pt, graph order, consecutive numbering) for Protpardelle-1c
partial diffusion on SuperCloud -- the same generator that made the T2 templates.

Output layout mirrors the T2 natives tree: <out>/natives/shardNN/<chain>.pdb (crc32 % 100 shards, so
no directory exceeds the 1024-file rule) + <out>/manifest.csv (chain,status,length,resid_span,pdb)
+ <out>/uncovered_chains.txt. Every chain that could not be exported is listed with a reason.

usage: python export_uncovered_natives.py <topology_index.pt> <index_band.npz> <processed_dir> <manifest.json> <out_dir>
"""

import csv
import json
import os
import pathlib
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.openfold_stub.np import residue_constants as rc
from proteinfoundation.utils.constants import ATOM_NUMBERING

ATOM_NAMES = [None] * len(ATOM_NUMBERING)
for name, i in ATOM_NUMBERING.items():
    ATOM_NAMES[i] = name


def write_pdb(coords, coord_mask, residue_type, path):
    """All resolved atoms, PDB atom names from the on-disk order, residues numbered 1..L in graph order."""
    lines, serial = [], 0
    for i in range(coords.shape[0]):
        rt = int(residue_type[i])
        resname = rc.restype_1to3.get(rc.restypes[rt], "UNK") if 0 <= rt < 20 else "UNK"
        for a in range(coords.shape[1]):
            if not bool(coord_mask[i, a]):
                continue
            name = ATOM_NAMES[a]
            x, y, z = (float(v) for v in coords[i, a])
            serial += 1
            pdb_name = f" {name:<3}" if len(name) < 4 else name
            lines.append(f"ATOM  {serial:5d} {pdb_name} {resname} A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2}")
    lines.append("TER")
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return serial


def main():
    idx_path, band_path, processed_dir, manifest_path, out = sys.argv[1:6]
    index = torch.load(idx_path, map_location="cpu", weights_only=False, mmap=True)
    ids = [str(s) for s in index["ids"]]
    cluster_of = np.asarray(index["cluster_of"])
    t2 = {str(c) for c in np.load(band_path, allow_pickle=True)["chains"]}
    has_t = np.array([i in t2 for i in ids])
    n_cl = int(cluster_of.max()) + 1
    covered = np.bincount(cluster_of, weights=has_t.astype(float), minlength=n_cl) > 0
    uncovered = [i for i, c in zip(ids, cluster_of) if not covered[c]]
    print(f"clusters {n_cl}, uncovered {int((~covered).sum())}, chains to export {len(uncovered)}", flush=True)

    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else None
    os.makedirs(os.path.join(out, "natives"), exist_ok=True)
    with open(os.path.join(out, "uncovered_chains.txt"), "w") as fh:
        fh.write("\n".join(uncovered) + "\n")
    rows, n_ok = [], 0
    for k, stem in enumerate(uncovered):
        pt = _processed_path_sharded(pathlib.Path(processed_dir), stem, manifest)
        shard = f"shard{zlib.crc32(stem.encode()) % 100:02d}"
        os.makedirs(os.path.join(out, "natives", shard), exist_ok=True)
        pdb = os.path.join(out, "natives", shard, f"{stem}.pdb")
        if not pt.exists():
            rows.append((stem, "missing_pt", 0, 0, ""))
            continue
        g = torch.load(str(pt), map_location="cpu", weights_only=False)
        coords, cmask, rtype = getattr(g, "coords", None), getattr(g, "coord_mask", None), getattr(g, "residue_type", None)
        if coords is None or cmask is None or rtype is None:
            rows.append((stem, "no_coords_or_types", 0, 0, ""))
            continue
        L = int(coords.shape[0])
        if int(cmask[:, 1].sum()) < 5:
            rows.append((stem, "too_few_ca", L, L, ""))
            continue
        n_atoms = write_pdb(coords, cmask, rtype, pdb)
        rows.append((stem, "ok", L, L, pdb))
        n_ok += 1
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(uncovered)} exported", flush=True)
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chain", "status", "length", "resid_span", "pdb"])
        w.writerows(rows)
    bad = [r for r in rows if r[1] != "ok"]
    print(f"exported {n_ok}/{len(uncovered)}; not exported: {len(bad)} -> {[r[:2] for r in bad[:10]]}", flush=True)
    lens = [r[2] for r in rows if r[1] == "ok"]
    print(f"length: min {min(lens)} median {int(np.median(lens))} max {max(lens)}", flush=True)
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
