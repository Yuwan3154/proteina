"""Deck examples (user 2026-10-08): (1) does an output reproduce its template rather than the native? (2) what does the
tri actually see of each template's secondary structure?

Per picked query, secondary structure exactly as the synthetic index builder computes it: the template's DSSP is
compute_dssp_target on the FULL-LENGTH template npz slot (present atoms scattered into atom37, unresolved residues -1),
the native's is the processed .pt's dssp_target; reported as H/E/loop/gap fractions and helix/strand element counts
(dssp_to_runs, min_len 1, as the index). Template backbone completeness (N, CA, C, O present, over the full length) and
consecutive CA-CA distances of the exported template PDB. For each output (CB-8, ConFind): TM to the native and TM to
the template (USalign -TMscore 5; template and native share residue indexing and length).
Usage: python t7_template_diag.py EXAMPLES_DIR
"""

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_c2c_structures import usalign_tm
from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.datasets.sse_topology import DSSP_HELIX, DSSP_STRAND, dssp_to_runs
from proteinfoundation.openfold_stub.np.residue_constants import atom_order
from proteinfoundation.utils.dssp_utils import compute_dssp_target

USALIGN = "/home/chenxiou/.local/bin/USalign"
S = "/orcd/scratch/orcd/011/chenxiou"
DATA = "/orcd/pool/006/chenxiou/proteina/data/pdb_train"
ex = sys.argv[1]
POOL = f"{ex}/templates_band"  # byte-identical copy of the band tree the index was built from (pool layout)
alias = {}
with open(f"{S}/synth_alias/auth_key.tsv") as fh:
    head = fh.readline().rstrip("\n").split("\t")
    il, ia = head.index("label_id"), head.index("auth_id")
    for line in fh:
        f = line.rstrip("\n").split("\t")
        alias[f[il]] = f[ia]
man = json.load(open(f"{DATA}/shard_manifest.json"))


def read_atom37(path):
    res, order = {}, []
    for l in open(path):
        if not l.startswith("ATOM"):
            continue
        r = int(l[22:26])
        if r not in res:
            res[r] = {}
            order.append(r)
        res[r][l[12:16].strip()] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
    xyz = np.zeros((len(order), 37, 3), np.float32)
    m = np.zeros((len(order), 37), np.float32)
    for i, r in enumerate(order):
        for nm, x in res[r].items():
            if nm in atom_order:
                xyz[i, atom_order[nm]] = x
                m[i, atom_order[nm]] = 1
    return xyz, m


def template_dssp(stem, ref):
    """Builder path: scatter the slot's present atoms into a full-length atom37 tensor, DSSP with bool masks."""
    slot, rw = int(ref.split("#")[1]), int(ref.split("@rw")[1].split("#")[0])
    auth = alias[stem]
    npz = np.load(f"{POOL}/shard{zlib.crc32(auth.encode()) % 1000:04d}/{auth}.npz")
    assert int(npz["rewind_steps"][slot]) == rw, (stem, slot, rw)
    amask = torch.from_numpy(npz["atom_mask"].astype(bool))
    full = torch.zeros(amask.shape[0], 37, 3)
    full[amask] = torch.from_numpy(npz["coords"][slot]).float()
    d = compute_dssp_target(full[None], amask[:, atom_order["CA"]][None], coord_mask=amask[None], coord_layout="atom37")[0]
    bb = {a: int(amask[:, atom_order[a]].sum()) for a in ("N", "CA", "C", "O")}
    return d, bb, int(amask.shape[0])


def native_dssp(stem):
    g = torch.load(_processed_path_sharded(Path(f"{DATA}/processed"), stem, man), weights_only=False)
    return g.dssp_target


def sse_stats(d):
    n = len(d)
    runs = dssp_to_runs(d, min_len=1)
    nh = sum(1 for t, _ in runs if t == DSSP_HELIX)
    ne = sum(1 for t, _ in runs if t == DSSP_STRAND)
    f = lambda v: float((d == v).sum()) / n
    return dict(H=f(DSSP_HELIX), E=f(DSSP_STRAND), loop=f(0), gap=f(-1), n_helix=nh, n_strand=ne)


P = json.load(open(os.path.join(ex, "picks.json")))
print("cat stem | SSE frac H/E/loop/gap + n_helix/n_strand: native vs template | template backbone | TM(out,native) vs TM(out,template)")
for cat, ents in P.items():
    for e in ents:
        st = e["stem"]
        nat, tpl = f"{ex}/sup/{st}_native.pdb", f"{ex}/sup/{st}_template_sup.pdb"
        nx, _ = read_atom37(nat)
        tx, _ = read_atom37(tpl)
        assert nx.shape[0] == tx.shape[0], (st, nx.shape, tx.shape)
        td, bb, Lt = template_dssp(st, e["ref_id"])
        sn, stt = sse_stats(native_dssp(st)), sse_stats(td)
        ca = tx[:, atom_order["CA"]]
        cad = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
        print(f"\n{cat} {st} L={nx.shape[0]} template TM {usalign_tm(tpl, nat, USALIGN):.3f}")
        print(f"  native   H {sn['H']:.2f} E {sn['E']:.2f} loop {sn['loop']:.2f} gap {sn['gap']:.2f}  elements H{sn['n_helix']} E{sn['n_strand']}")
        print(f"  template H {stt['H']:.2f} E {stt['E']:.2f} loop {stt['loop']:.2f} gap {stt['gap']:.2f}  elements H{stt['n_helix']} E{stt['n_strand']}")
        print(f"  template backbone atoms {bb} of {Lt}; CA-CA min/median/max {cad.min():.2f}/{np.median(cad):.2f}/{cad.max():.2f} A, n>4.2 A {(cad > 4.2).sum()}")
        for d in ("CB8", "CF"):
            o = f"{ex}/sup/{st}_{d}_sup.pdb"
            print(f"  {d:3s} output: TM to native {usalign_tm(o, nat, USALIGN):.3f}  TM to template {usalign_tm(o, tpl, USALIGN):.3f}")
print("\n[done]")
