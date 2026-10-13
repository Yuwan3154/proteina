"""T8 template rescue trial, stage 1 of 2 (user 2026-10-13: "I wonder if these templates could be rescued by simple Rosetta
idealization protocols ... Try it out."). From a gated build part's skip log, a seeded sample of templates the clash gate
dropped (> k non-bonded CA-CA pairs < cutoff) plus controls the gate kept (1..k clashes and 0 clashes); writes each
template as a full-atom PDB (ATOM records only, residues numbered 1..L in template order, so Rosetta's pose index = the
gate's residue index) and each chain's native as a CA-only PDB, plus manifest.tsv. Stage 2: t8_rosetta_rescue.py.
Usage (PYTHONPATH=repo): python scratchpad/t8_rosetta_dump.py --part-dir DIR --pack ROOT --native-pack PACK --out DIR
"""

import argparse
import io
import os
import random

import numpy as np
import torch

from proteinfoundation.openfold_stub.np.residue_constants import atom_types, restype_1to3, restypes
from proteinfoundation.utils.precompute_synthetic_topology_index import CA_IDX, ca_clash_pairs, write_ca_pdb
from proteinfoundation.utils.t2_pack_reader import TemplatePack


def write_full_pdb(full, amask, aatype, path):
    n = 0
    with open(path, "w") as fh:
        for i in range(amask.shape[0]):
            assert int(aatype[i]) < len(restypes), f"{path}: residue {i} has aatype {int(aatype[i])} (X)"
            rn = restype_1to3[restypes[int(aatype[i])]]
            for a in np.flatnonzero(amask[i]):
                n += 1
                name = atom_types[a]
                col = name if len(name) == 4 else f" {name:<3s}"  # PDB cols 13-16: 1-letter elements start at 14
                x, y, z = (float(v) for v in full[i, a])
                fh.write(f"ATOM  {n:5d} {col} {rn} A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                         f"          {name[0]:>2s}\n")
        fh.write("TER\nEND\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-dir", required=True, help="build part dir with part_0000.pt + skips_0000.tsv")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--native-pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gated", type=int, default=40)
    ap.add_argument("--n-kept-clash", type=int, default=10)
    ap.add_argument("--n-kept-clean", type=int, default=10)
    ap.add_argument("--clash", type=float, default=3.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "in"), exist_ok=True)
    tp = TemplatePack(args.pack)
    draw = tp.col["draw"]
    gated = set()
    for l in open(os.path.join(args.part_dir, f"skips_{args.part:04d}.tsv")):
        s, w, r = l.rstrip("\n").split("\t")
        if r.startswith("ca_clashes_gt_"):
            gated.add((s, int(w[4:])))
    part = torch.load(os.path.join(args.part_dir, f"part_{args.part:04d}.pt"), map_location="cpu", weights_only=False)
    kept = [(ch["stem"], int(r[3])) for ch in part["chains"] for r in ch["rows"]]
    row_of = {}
    for s in {s for s, _ in list(gated) + kept}:
        rs = list(tp.rows(s))
        assert len({int(draw[r]) for r in rs}) == len(rs), f"{s}: draw ids repeat"
        row_of.update({(s, int(draw[r])): r for r in rs})

    def load(key):
        rec = tp._read(row_of[key])
        full = np.zeros((rec["atom_mask"].shape[0], 37, 3), dtype=np.float32)
        full[rec["atom_mask"]] = rec["coords"]
        return rec, full

    def n_clash(key):
        rec, full = load(key)
        return ca_clash_pairs(full[:, CA_IDX], rec["atom_mask"][:, CA_IDX], args.clash)

    rng = random.Random(args.seed)
    g = rng.sample(sorted(gated), min(args.n_gated, len(gated)))
    kept_n = {k: n_clash(k) for k in kept}
    kc = rng.sample(sorted(k for k, v in kept_n.items() if v > 0), args.n_kept_clash)
    k0 = rng.sample(sorted(k for k, v in kept_n.items() if v == 0), args.n_kept_clean)
    z = np.load(args.native_pack + ".idx.npz", allow_pickle=True)
    nat = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    rows = []
    with open(args.native_pack, "rb") as fh:
        for group, keys in (("gated", g), ("kept_1_k", kc), ("kept_0", k0)):
            for key in keys:
                rec, full = load(key)
                c = ca_clash_pairs(full[:, CA_IDX], rec["atom_mask"][:, CA_IDX], args.clash)
                stem, d = key
                tag = f"{stem}_tpl{d}"
                assert rec["atom_mask"][:, CA_IDX].all(), f"{tag}: a residue without CA"
                write_full_pdb(full, rec["atom_mask"], rec["aatype"], os.path.join(args.out, "in", f"{tag}.pdb"))
                npdb = os.path.join(args.out, "in", f"{stem}_native_ca.pdb")
                if not os.path.exists(npdb):
                    o, ln = nat[stem]
                    fh.seek(o)
                    gn = torch.load(io.BytesIO(fh.read(ln)), map_location="cpu", weights_only=False)
                    write_ca_pdb(gn.coords[:, CA_IDX].numpy(), (gn.coord_mask[:, CA_IDX] > 0.5).numpy(), npdb)
                rows.append(f"{tag}\t{group}\t{stem}\t{d}\t{rec['tm_native']:.4f}\t{c}\t{rec['atom_mask'].shape[0]}")
    with open(os.path.join(args.out, "manifest.tsv"), "w") as f:
        f.write("tag\tgroup\tstem\tdraw\ttm_native\tclash_pairs\tL\n" + "\n".join(rows) + "\n")
    print(f"wrote {len(rows)} templates ({len(g)} gated, {len(kc)} kept with clashes, {len(k0)} clean) to {args.out}")


if __name__ == "__main__":
    main()
