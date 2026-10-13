"""T8 template rescue trial, stage 2 of 2 (stage 1: t8_rosetta_dump.py). For every template PDB in the manifest, run
standard Rosetta protocols at their DEFAULT settings and measure what the clash gate measures:
  idealize  = protocols.idealize.IdealizeMover() (ideal bond lengths / angles, coordinate-restrained)
  relax_cst = FastRelax(ref2015), coordinate-restrained to the start (= the command-line -constrain_relax_to_start_coords,
              which sets BOTH constrain_coords and constrain_relax_to_start_coords; the restraint's virtual root removed)
Per (template, protocol): non-bonded CA-CA pairs < cutoff before / after (same metric as the builder's ca_clash_pairs),
CA RMSD to the input (no superposition: restrained protocols keep the frame), largest consecutive CA-CA step
(IdealizeMover closes chain breaks), TM to the native (US-align, normalised by
the native, as the index builder's row TM), DSSP identity before vs after, ref2015 per residue, wall seconds.
Env: /orcd/scratch/orcd/011/chenxiou/envs/pyrosetta. Usage: python scratchpad/t8_rosetta_rescue.py --dir DIR
"""

import argparse
import os
import subprocess
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import remove_virtual_residues
from pyrosetta.rosetta.core.scoring.dssp import Dssp
from pyrosetta.rosetta.protocols.idealize import IdealizeMover
from pyrosetta.rosetta.protocols.relax import FastRelax

_ARGS = None


def ca_clash_pairs(ca, cutoff):
    """Same as precompute_synthetic_topology_index.ca_clash_pairs with every CA present."""
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    i, j = np.triu_indices(len(ca), k=1)
    return int(((j - i >= 2) & (d[i, j] < cutoff)).sum())


def max_step(ca):
    return float(np.linalg.norm(np.diff(ca, axis=0), axis=1).max())


def ca_of(pose):
    return np.array([list(pose.residue(i).xyz("CA")) for i in range(1, pose.total_residue() + 1)], dtype=np.float64)


def tm_to_native(ca, native_pdb, usalign):
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        for i, (x, y, z) in enumerate(ca, 1):
            fh.write(f"ATOM  {i:5d}  CA  GLY A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        fh.write("TER\nEND\n")
    out = subprocess.run([usalign, native_pdb, fh.name, "-TMscore", "0"], capture_output=True, text=True).stdout
    os.unlink(fh.name)
    return next(float(l.split("=")[1].split()[0]) for l in out.splitlines()
                if l.startswith("TM-score=") and "Structure_1" in l)


def init(args):
    global _ARGS
    _ARGS = args
    pyrosetta.init("-mute all")


def protocol(name):
    if name == "idealize":
        return IdealizeMover()
    fr = FastRelax()
    fr.set_scorefxn(pyrosetta.get_fa_scorefxn())
    fr.constrain_coords(True)
    fr.constrain_relax_to_start_coords(True)
    return fr


def job(row):
    tag, group, stem = row[0], row[1], row[2]
    a = _ARGS
    pose0 = pyrosetta.pose_from_pdb(os.path.join(a.dir, "in", f"{tag}.pdb"))
    sfx = pyrosetta.get_fa_scorefxn()
    ca0 = ca_of(pose0)
    assert len(ca0) == int(row[6]), f"{tag}: Rosetta read {len(ca0)} residues, the dump wrote {row[6]}"
    ss0 = Dssp(pose0).get_dssp_secstruct()
    nat = os.path.join(a.dir, "in", f"{stem}_native_ca.pdb")
    assert ca_clash_pairs(ca0, a.clash) == int(row[5]), f"{tag}: clash count differs from the dump's"
    base = dict(tag=tag, group=group, L=len(ca0), clash_before=ca_clash_pairs(ca0, a.clash), step_before=max_step(ca0),
                tm_before=tm_to_native(ca0, nat, a.usalign), e_before=sfx(pose0) / len(ca0))
    out = []
    for name in a.protocols:
        pose = pose0.clone()
        t0 = time.time()
        protocol(name).apply(pose)
        dt = time.time() - t0
        remove_virtual_residues(pose)
        assert pose.total_residue() == len(ca0), f"{tag} {name}: residue count changed"
        ca = ca_of(pose)
        pose.dump_pdb(os.path.join(a.dir, "out", f"{tag}.{name}.pdb"))
        ss = Dssp(pose).get_dssp_secstruct()
        out.append(dict(base, protocol=name, sec=dt, clash_after=ca_clash_pairs(ca, a.clash), step_after=max_step(ca),
                        rmsd_ca=float(np.sqrt(((ca - ca0) ** 2).sum(1).mean())),
                        tm_after=tm_to_native(ca, nat, a.usalign), e_after=sfx(pose) / len(ca),
                        dssp_same=sum(x == y for x, y in zip(ss0, ss)) / len(ss)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--protocols", nargs="+", default=["idealize", "relax_cst"])
    ap.add_argument("--clash", type=float, default=3.4)
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.dir, "out"), exist_ok=True)
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(args.dir, "manifest.tsv"))][1:]
    with Pool(args.workers, initializer=init, initargs=(args,)) as p:
        res = [r for rs in p.imap_unordered(job, rows) for r in rs]
    keys = ["tag", "group", "protocol", "L", "clash_before", "clash_after", "step_before", "step_after", "rmsd_ca",
            "tm_before", "tm_after",
            "dssp_same", "e_before", "e_after", "sec"]
    with open(os.path.join(args.dir, "results.tsv"), "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in sorted(res, key=lambda r: (r["group"], r["tag"], r["protocol"])):
            f.write("\t".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    for g in ("gated", "kept_1_k", "kept_0"):
        for name in args.protocols:
            s = [r for r in res if r["group"] == g and r["protocol"] == name]
            if not s:
                continue
            cb, ca = np.array([r["clash_before"] for r in s]), np.array([r["clash_after"] for r in s])
            print(f"{g:9s} {name:9s} n={len(s):3d} clashes median {np.median(cb):.0f} -> {np.median(ca):.0f}; "
                  f"<=4 after {int((ca <= 4).sum())}/{len(s)}, 0 after {int((ca == 0).sum())}/{len(s)}; "
                  f"CA RMSD median {np.median([r['rmsd_ca'] for r in s]):.2f} A; "
                  f"dTM median {np.median([r['tm_after'] - r['tm_before'] for r in s]):+.4f}; "
                  f"DSSP same {np.median([r['dssp_same'] for r in s]):.3f}; "
                  f"REU/res {np.median([r['e_before'] for r in s]):.1f} -> {np.median([r['e_after'] for r in s]):.1f}; "
                  f"s/template median {np.median([r['sec'] for r in s]):.1f}", flush=True)


if __name__ == "__main__":
    main()
