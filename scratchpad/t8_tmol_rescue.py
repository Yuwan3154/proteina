"""T8 template rescue trial on GPU (user 2026-10-13: "It may be better to try this task on the GPU nodes; try that."):
tmol FastRelax on the templates dumped by t8_rosetta_dump.py, at tmol's defaults (tmol.fast_relax: MonomerRelax2019
schedule, num_repeats 2, repack-only Dunbrack task, Cartesian minimization, constraint ramp), restrained to the start
by create_mainchain_coordinate_constraints (sd 0.5 A, tmol's match to Rosetta FastRelax) with constraint weight 1.0 (as
Rosetta FastRelax's coordinate_constraint under -constrain_relax_to_start_coords and the tmol tutorial). Score function
beta_nov2016_cart (tmol's; the CPU reference uses ref2015 in torsion space). tmol defaults that differ from the CPU twin
(--min cart): Cartesian minimization (bond geometry moves; pose_stack_from_pdb bonds every consecutive residue, so a chain break can be
closed) and restraints on N / CA / C only. Same metrics as t8_rosetta_rescue.py except DSSP: clash pairs
before / after, largest consecutive CA-CA step, CA RMSD to the input (no superposition), TM to the native, energy per
residue without the constraint term, GPU seconds per batch. Relaxed poses -> out/tmol_b{batch}_{i}.pdb (one MODEL per template).
--batch can change results (packing draws from one random stream per batch).
Env: /orcd/scratch/orcd/011/chenxiou/envs/tmol (tmol 0.1.63 release wheel +cu128torch2.11, torch 2.11.0+cu128). --min kin (user 2026-10-14 "2. Yes"): the same FastRelax with a torsion-space minimizer instead (run_kin_min at its defaults,
named torsions free, jumps fixed, as tutorial 06's kinematic form, but kin_dtype at run_kin_min's float32 default where
the tutorial passes float64), i.e. bond geometry fixed like Rosetta's FastRelax. Outputs out/tmol_kin_b{batch}_{i}.pdb and
results_tmol_kin_b{batch}.tsv.
Usage: python scratchpad/t8_tmol_rescue.py --dir DIR --batch N [--min cart|kin]
"""

import argparse
import os
import subprocess
import tempfile
import time

import numpy as np
import torch
import tmol
from tmol.kinematics import CartesianMoveMap, FoldForest, MoveMap
from tmol.pack import PackerPalette
from tmol.pose import PoseStackBuilder
from tmol.score import ScoreType, beta2016_score_function
from tmol.score.constraint import create_mainchain_coordinate_constraints


def ca_clash_pairs(ca, cutoff):
    """Same as precompute_synthetic_topology_index.ca_clash_pairs with every CA present."""
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    i, j = np.triu_indices(len(ca), k=1)
    return int(((j - i >= 2) & (d[i, j] < cutoff)).sum())


def max_step(ca):
    return float(np.linalg.norm(np.diff(ca, axis=0), axis=1).max())


def tm_to_native(ca, native_pdb, usalign):
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        for i, (x, y, z) in enumerate(ca, 1):
            fh.write(f"ATOM  {i:5d}  CA  GLY A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        fh.write("TER\nEND\n")
    out = subprocess.run([usalign, native_pdb, fh.name, "-TMscore", "0"], capture_output=True, text=True).stdout
    os.unlink(fh.name)
    return next(float(l.split("=")[1].split()[0]) for l in out.splitlines()
                if l.startswith("TM-score=") and "Structure_1" in l)


def ca_per_pose(ps):
    """[n_poses] arrays of CA coordinates, in block order (as constraint._utility.constrain_all_ca indexes them)."""
    coords = ps.coords.detach().cpu().numpy()
    out = []
    for p in range(ps.n_poses):
        rows = []
        for b in range(ps.max_n_blocks):
            if ps.is_real_block(p, b):
                bt = ps.block_type(p, b)
                rows.append(coords[p, int(ps.block_coord_offset[p, b]) + bt.atom_to_idx["CA"]])
        out.append(np.array(rows, dtype=np.float64))
    return out


def energy_per_pose(ps, sfxn):
    """Weighted total per pose without the constraint term (tutorial 06's score_components_by_pose)."""
    terms = sfxn.render_whole_pose_scoring_module(ps)(ps.coords, sum_terms=False, apply_weights=True)
    ci = [k for k, t in enumerate(sfxn.all_score_types()) if t == ScoreType.constraint]
    tot = terms.sum(dim=0) - (terms[ci[0]] if ci else 0)
    return tot.detach().cpu().numpy()


def kin_min(pose_stack, sfxn, *, fold_forest, move_map, verbose):
    """fast_relax's min_fn contract; run_kin_min at its own defaults (tutorial 06's kinematic form)."""
    return tmol.run_kin_min(pose_stack, sfxn, fold_forest, move_map, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--batch", type=int, default=1, help="templates per PoseStack")
    ap.add_argument("--clash", type=float, default=3.4)
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min", choices=("cart", "kin"), default="cart", help="FastRelax minimizer: tmol default (Cartesian) or torsion space")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0), flush=True)
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(args.dir, "manifest.tsv"))][1:]
    keys = ["tag", "group", "protocol", "L", "clash_before", "clash_after", "step_before", "step_after", "rmsd_ca",
            "tm_before", "tm_after", "e_before", "e_after", "sec_batch", "batch"]
    os.makedirs(os.path.join(args.dir, "out"), exist_ok=True)
    tag_min = "" if args.min == "cart" else "_kin"
    out_path = os.path.join(args.dir, f"results_tmol{tag_min}_b{args.batch}.tsv")
    with open(out_path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for s in range(0, len(rows), args.batch):
            chunk = rows[s:s + args.batch]
            poses = [tmol.pose_stack_from_pdb(os.path.join(args.dir, "in", f"{r[0]}.pdb"), dev) for r in chunk]
            ps = create_mainchain_coordinate_constraints(PoseStackBuilder.from_poses(poses, dev))
            sfxn = beta2016_score_function(dev)
            sfxn.set_weight(ScoreType.constraint, 1.0)
            ca0, e0 = ca_per_pose(ps), energy_per_pose(ps, sfxn)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            ff = FoldForest.reasonable_fold_forest(ps)
            if args.min == "cart":
                relaxed = tmol.fast_relax(ps, sfxn, PackerPalette(), CartesianMoveMap(), ff)
            else:
                mm = MoveMap.from_pose_stack(ps)
                mm.move_all_named_torsions = True
                mm.move_all_jumps = False
                mm.move_all_root_jumps = False
                relaxed = tmol.fast_relax(ps, sfxn, PackerPalette(), mm, ff, min_fn=kin_min)
            torch.cuda.synchronize()
            dt = time.time() - t0
            ca1, e1 = ca_per_pose(relaxed), energy_per_pose(relaxed, sfxn)
            # one model per template, in chunk order
            tmol.write_pose_stack_pdb(relaxed, os.path.join(args.dir, "out", f"tmol{tag_min}_b{args.batch}_{s // args.batch:04d}.pdb"))
            for k, r in enumerate(chunk):
                tag, group, stem, L = r[0], r[1], r[2], int(r[6])
                assert len(ca0[k]) == L == len(ca1[k]), f"{tag}: residues {len(ca0[k])} / {L} / {len(ca1[k])}"
                assert ca_clash_pairs(ca0[k], args.clash) == int(r[5]), f"{tag}: clash count differs from the dump's"
                nat = os.path.join(args.dir, "in", f"{stem}_native_ca.pdb")
                v = dict(tag=tag, group=group, protocol="tmol_relax_cst" + tag_min, L=L,
                         clash_before=ca_clash_pairs(ca0[k], args.clash), clash_after=ca_clash_pairs(ca1[k], args.clash),
                         step_before=max_step(ca0[k]), step_after=max_step(ca1[k]),
                         rmsd_ca=float(np.sqrt(((ca1[k] - ca0[k]) ** 2).sum(1).mean())),
                         tm_before=tm_to_native(ca0[k], nat, args.usalign), tm_after=tm_to_native(ca1[k], nat, args.usalign),
                         e_before=float(e0[k]) / L, e_after=float(e1[k]) / L, sec_batch=dt, batch=len(chunk))
                f.write("\t".join(f"{v[x]:.4f}" if isinstance(v[x], float) else str(v[x]) for x in keys) + "\n")
            f.flush()
            print(f"batch {s // args.batch}: {len(chunk)} templates, {dt:.1f} s, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
