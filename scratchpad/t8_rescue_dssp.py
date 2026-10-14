"""DSSP before vs after the rescue protocols (the index builds each template's topology from its DSSP, so a rescue must
keep the secondary structure). Rosetta DSSP on heavy atoms only: input in/{tag}.pdb vs PyRosetta out/{tag}.{protocol}.pdb
and tmol out/tmol_b1_{i:04d}.pdb (i = manifest row; hydrogens dropped). Per template: fraction of residues with the same
H/E/L letter, and the helix / strand element counts (runs of H, runs of E) before -> after.
Env: /orcd/scratch/orcd/011/chenxiou/envs/pyrosetta. Usage: python scratchpad/t8_rescue_dssp.py --dir DIR
"""

import argparse
import itertools
import os

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.scoring.dssp import Dssp


def ss_of(path):
    # by atom NAME (tmol's PDB writer leaves the element column empty); also drops names like 1HB
    lines = [l for l in open(path) if l.startswith("ATOM") and l[12:16].strip().lstrip("0123456789")[:1] != "H"]
    pose = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, "".join(lines) + "END\n")
    return Dssp(pose).get_dssp_secstruct()


def n_elements(ss, c):
    return sum(1 for k, _ in itertools.groupby(ss) if k == c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    pyrosetta.init("-mute all")
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(args.dir, "manifest.tsv"))][1:]
    out = {}
    for i, r in enumerate(rows):
        tag, group = r[0], r[1]
        s0 = ss_of(os.path.join(args.dir, "in", f"{tag}.pdb"))
        cands = {"idealize": f"{tag}.idealize.pdb", "relax_cst": f"{tag}.relax_cst.pdb", "tmol_relax_cst": f"tmol_b1_{i:04d}.pdb"}
        for proto, fn in cands.items():
            p = os.path.join(args.dir, "out", fn)
            if not os.path.exists(p):
                continue
            s1 = ss_of(p)
            assert len(s1) == len(s0), f"{tag} {proto}: {len(s1)} vs {len(s0)} residues"
            out.setdefault((group, proto), []).append(
                (sum(a == b for a, b in zip(s0, s1)) / len(s0), n_elements(s0, "H"), n_elements(s1, "H"),
                 n_elements(s0, "E"), n_elements(s1, "E")))
    for (group, proto), v in sorted(out.items()):
        v = np.array(v)
        dh, de = v[:, 2] - v[:, 1], v[:, 4] - v[:, 3]
        print(f"{group:9s} {proto:15s} n={len(v):2d} | same DSSP letter median {np.median(v[:, 0]):.3f} min {v[:, 0].min():.3f} | "
              f"helices {np.median(v[:, 1]):.0f} -> {np.median(v[:, 2]):.0f} (changed in {int((dh != 0).sum())}) | "
              f"strands {np.median(v[:, 3]):.0f} -> {np.median(v[:, 4]):.0f} (changed in {int((de != 0).sum())})")


if __name__ == "__main__":
    main()
