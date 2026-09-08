"""In a MIRRORED generation, are the residues D or L? This decides whether reflection is free.

Why it matters. The sampler was measured exactly reflection-equivariant (job 22250760: where the
mirrored rollout tracked, hand_B = 1 - hand_A to within 0.024). So M applied to a sample is not a
post-hoc "repair" at all -- it is exactly the sample the model WOULD have produced from the
reflected seed. If a mirrored generation is a TRUE global mirror, its residues are D, and reflecting
restores BOTH the fold hand AND L-stereochemistry at zero cost: no second rollout, no repacking.

But if the model instead emits a CHIMERA -- L residues on a mirrored fold -- then reflection fixes
the fold and BREAKS every residue, and rejection resampling (2.10 rollouts/sample) is the only
correct route.

⛔ THIS IS EXACTLY THE QUESTION `chirality_agreement` CANNOT ANSWER: it reads 0.999 on mirrored
chains and is documented in c2c_dump.py as unable to see a mirror. So measure the signed volume
directly instead of trusting it.

Signed volume V = (N-CA) x (C-CA) . (CB-CA). Grounded reference value: atom_features.py records
+2.485..+2.667 for the L form across all 19 non-glycine types, so L is POSITIVE. Glycine has no CB
and is skipped.
"""

import argparse
import glob

import numpy as np


def read_residues(path):
    """-> list of dicts {name: xyz} per residue, in file order."""
    res, cur, key = [], {}, None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            rid = line[22:27]
            if rid != key:
                if cur:
                    res.append(cur)
                cur, key = {}, rid
            cur[line[12:16].strip()] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if cur:
        res.append(cur)
    return res


def signed_volumes(path):
    out = []
    for r in read_residues(path):
        if not all(a in r for a in ("N", "CA", "C", "CB")):
            continue  # glycine, or an unresolved slot
        n, ca, c, cb = (np.asarray(r[a]) for a in ("N", "CA", "C", "CB"))
        out.append(float(np.dot(np.cross(n - ca, c - ca), cb - ca)))
    return np.asarray(out)


def ca_trace(path):
    return np.asarray([r["CA"] for r in read_residues(path) if "CA" in r])


def helix_pos_frac_from_ca(ca):
    import sys
    sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")
    from proteinfoundation.utils.c2c_dump import _ca_dihedrals
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def summarise(paths, label):
    rows = []
    for p in paths:
        v = signed_volumes(p)
        if not len(v):
            continue
        h = helix_pos_frac_from_ca(ca_trace(p))
        if np.isnan(h):
            continue
        rows.append((h, float(v.mean()), float((v > 0).mean())))
    a = np.asarray(rows)
    if not len(a):
        print(f"=== {label} === NOTHING SCORED"); return
    mir = a[:, 0] > 0.5
    print(f"\n=== {label} ===  n={len(a)}   mirrored-by-fold: {int(mir.sum())}")
    for sel, nm in ((~mir, "fold RIGHT-handed"), (mir, "fold MIRRORED")):
        if sel.sum():
            print(f"  {nm:<20} n={int(sel.sum()):4d}  mean signed vol {a[sel,1].mean():+8.4f}  "
                  f"frac residues POSITIVE(=L) {a[sel,2].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_254")
    args = ap.parse_args()
    summarise(sorted(glob.glob(f"{args.gen_dir}/*_gt.pdb")), "NATIVES (defines the L sign)")
    summarise(sorted(glob.glob(f"{args.gen_dir}/*_gen.pdb")), "c2c GENERATED")
    print("\nIf 'fold MIRRORED' rows show NEGATIVE mean volume / low frac-L, the samples are TRUE")
    print("global mirrors and one reflection repairs fold AND stereochemistry at zero cost.")
    print("If they show POSITIVE / high frac-L, they are CHIMERAS and reflection would break them.")


if __name__ == "__main__":
    main()
