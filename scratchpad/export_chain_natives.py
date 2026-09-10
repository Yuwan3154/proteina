"""Export all-atom native PDBs for an EXPLICIT chain list (and append them to a manifest), for chains
that the cluster-level backfill does not reach -- e.g. the fixed validation chains, which are drawn by
name regardless of the eligible-ids restriction, so every one of them needs its own template.

usage: python export_chain_natives.py --chains a.txt --processed-dir <dir> --manifest-in <shard_manifest.json>
                                      --out <dir> --append-manifest <manifest.csv>
"""

import argparse
import csv
import json
import os
import pathlib
import sys
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from scratchpad.export_uncovered_natives import write_pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--manifest-in", required=True, help="shard_manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append-manifest", required=True, help="manifest.csv the generator reads")
    a = ap.parse_args()

    manifest = json.load(open(a.manifest_in))
    want = [l.strip() for l in open(a.chains) if l.strip()]
    have = set()
    if os.path.exists(a.append_manifest):
        with open(a.append_manifest) as fh:
            have = {r["chain"] for r in csv.DictReader(fh)}
    rows = []
    for stem in want:
        if stem in have:
            print(f"  {stem}: already in the manifest, skipping")
            continue
        pt = _processed_path_sharded(pathlib.Path(a.processed_dir), stem, manifest)
        if not pt.exists():
            print(f"  {stem}: FATAL no processed .pt at {pt}")
            return 2
        g = torch.load(str(pt), map_location="cpu", weights_only=False)
        shard = f"shard{zlib.crc32(stem.encode()) % 100:02d}"
        os.makedirs(os.path.join(a.out, "natives", shard), exist_ok=True)
        pdb = os.path.join(a.out, "natives", shard, f"{stem}.pdb")
        n_atoms = write_pdb(g.coords, g.coord_mask, g.residue_type, pdb)
        L = int(g.coords.shape[0])
        rows.append((stem, "ok", L, L, pdb))
        print(f"  {stem}: L={L}, {n_atoms} atoms -> {pdb}")
    if rows:
        # ⛔ append, never rewrite: the generator shards the manifest by ROW ORDER, so rewriting it
        # would reassign every chain to a different shard and break resumability of a running array.
        with open(a.append_manifest, "a", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"appended {len(rows)} rows to {a.append_manifest}")
    print("EXPORT_CHAINS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
