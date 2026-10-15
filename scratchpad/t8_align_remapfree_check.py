"""Remap-free spot check of the T8 alignment ground truth (ProteinEBMalign follow-up 2026-10-14: "re-align on CA files
written from the trainer's own arrays and require cell-set equality with the stored labels"). Shares no index-mapping
code with the builder: CA files are written here with ResSeq = STRUCTURE index + 1 (present CAs only), the alignment
block's written order is mapped back by reading ResSeq out of the files themselves, and each template residue's element
is rebuilt from the stored runs by cumulative length. Sample: --n rows from the build parts, half uniformly random,
half from natives with a CA-less residue strictly inside the chain (the case a residue COUNT check cannot see). Each
sampled row must equal (a) the part's stored target and (b) the merged index's align_flat row the trainer reads.
Usage (PYTHONPATH=repo): python scratchpad/t8_align_remapfree_check.py --parts-dir D --n-parts N --index IDX --pack PACK
    --native-pack NPACK [--n 128]
"""

import argparse
import io
import os
import random
import subprocess
import tempfile

import numpy as np
import torch

from proteinfoundation.utils.t2_pack_reader import TemplatePack

CA = 1
NONE = -1


def write_ca(ca, present, path):
    with open(path, "w") as fh:
        for n, i in enumerate(np.flatnonzero(present), 1):
            x, y, z = (float(v) for v in ca[i])
            fh.write(f"ATOM  {n:5d}  CA  GLY A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        fh.write("TER\nEND\n")


def resseq(path):
    return [int(l[22:26]) - 1 for l in open(path) if l.startswith("ATOM")]


def realign(nat_pdb, tpl_pdb, usalign):
    res = subprocess.run([usalign, nat_pdb, tpl_pdb, "-TMscore", "0"], capture_output=True, text=True)
    assert res.returncode == 0, f"USalign rc {res.returncode}"
    out = res.stdout.splitlines()
    k = next(i for i, l in enumerate(out) if l.startswith('(":" denotes'))
    sq, st = out[k + 1], out[k + 3]
    assert len(sq) == len(st), "ragged alignment block"
    rn, rt = resseq(nat_pdb), resseq(tpl_pdb)
    assert len(sq) - sq.count("-") == len(rn) and len(st) - st.count("-") == len(rt), "USalign read != written"
    pairs, a, b = [], 0, 0
    for x, y in zip(sq, st):
        if x != "-" and y != "-":
            pairs.append((rn[a], rt[b]))
        a += x != "-"
        b += y != "-"
    return pairs


def elem_of_residue(runs, L, element_types):
    elem, pos, e = np.full(L, NONE, dtype=np.int16), 0, 0
    for t, n in runs:
        if t in element_types:
            elem[pos:pos + n] = e
            e += 1
        pos += n
    assert pos == L, f"runs sum {pos} != L {L}"
    return elem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--n-parts", type=int, required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--pack", required=True, help="template pack root")
    ap.add_argument("--native-pack", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    args = ap.parse_args()
    z = np.load(args.native_pack + ".idx.npz", allow_pickle=True)
    off = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    nat_fh = open(args.native_pack, "rb")

    def native(stem):
        o, ln = off[stem]
        nat_fh.seek(o)
        return torch.load(io.BytesIO(nat_fh.read(ln)), map_location="cpu", weights_only=False)

    # every template row with its merged-index row number (merge order: parts in order, per chain native then templates)
    rows, r, element_types = [], 0, None
    for p in range(args.n_parts):
        d = torch.load(os.path.join(args.parts_dir, f"part_{p:04d}.pt"), map_location="cpu", weights_only=False)
        element_types = tuple(d["element_types"])
        for ch in d["chains"]:
            if ch["native"] is None:
                continue
            r += 1
            for (rrow, tm, rewind, k, align_b, tm_us) in ch["rows"]:
                rows.append((ch["stem"], int(k), int(rewind), rrow[0], align_b, r))
                r += 1
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    half, picked, interior = args.n // 2, [], []
    for row in rows:
        if len(picked) < half:
            picked.append(row)
            continue
        if len(interior) < args.n - half:
            ca = (native(row[0]).coord_mask[:, CA] > 0.5).numpy()
            idx = np.flatnonzero(~ca)
            if idx.size and ((idx > np.flatnonzero(ca).min()) & (idx < np.flatnonzero(ca).max())).any():
                interior.append(row)
        if len(interior) >= args.n - half:
            break
    print(f"template rows {len(rows)}; sampled {len(picked)} random + {len(interior)} with an interior CA-less native residue",
          flush=True)
    idx = torch.load(args.index, map_location="cpu", weights_only=False, mmap=True)
    tp = TemplatePack(args.pack)
    draw = tp.col["draw"]
    ok_part = ok_index = 0
    bad = []
    with tempfile.TemporaryDirectory() as td:
        for stem, k, rewind, runs, align_b, ridx in picked + interior:
            assert str(idx["ids"][ridx]) == f"{stem}@rw{rewind}#{k}", f"row counter off: {idx['ids'][ridx]} vs {stem}#{k}"
            g = native(stem)
            L = int(g.coords.shape[0])
            pn = os.path.join(td, "n.pdb")
            write_ca(g.coords[:, CA].numpy(), (g.coord_mask[:, CA] > 0.5).numpy(), pn)
            hits = [x for x in tp.rows(stem) if int(draw[x]) == k]
            assert len(hits) == 1, f"{stem}: draw {k} matches {len(hits)} pack rows"
            r_t = hits[0]
            rec = tp._read(r_t)
            full = np.zeros((rec["atom_mask"].shape[0], 37, 3), dtype=np.float32)
            full[rec["atom_mask"]] = rec["coords"]
            pt = os.path.join(td, "t.pdb")
            write_ca(full[:, CA], rec["atom_mask"][:, CA], pt)
            elem_t = elem_of_residue([tuple(x) for x in runs], int(rec["atom_mask"].shape[0]), element_types)
            exp = np.full(L, NONE, dtype=np.int16)
            for i, j in realign(pn, pt, args.usalign):
                exp[i] = elem_t[j]
            stored = np.frombuffer(align_b, dtype=np.int16)
            a, b = int(idx["align_offset"][ridx]), int(idx["align_offset"][ridx + 1])
            merged = idx["align_flat"][a:b].numpy()
            ok_part += np.array_equal(exp, stored)
            ok_index += np.array_equal(stored, merged)
            if not np.array_equal(exp, stored):
                bad.append(("rederived", stem, k, int((exp != stored).sum()), L))
            if not np.array_equal(stored, merged):
                bad.append(("merged", stem, k, int((stored != merged).sum()), L))
    n = len(picked) + len(interior)
    print(f"re-derived == stored (part): {ok_part} / {n}; stored == merged index row: {ok_index} / {n}")
    print("mismatches (kind, stem, draw, residues differing, L):", bad[:10])


if __name__ == "__main__":
    main()
