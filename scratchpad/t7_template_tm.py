"""Deck (user 2026-10-08): template quality vs the native, for every query of the T7 full set.

For each query: its pinned synthetic template (ref_id 'stem@rw<rewind>#<slot>', looked up exactly as the index builder
does: label stem -> auth id via the alias -> <pool>/shard{crc32(auth)%1000}/<auth>.npz, slot k), written as an atom37
PDB; the native from its processed .pt (PDB atom order -> atom37) as a PDB; the residue sequences must match (the
template is a partial-diffusion variant of the native); TM = USalign -TMscore 5 normalised by the native, the same
call that scored the c2c outputs. Writes OUT.tsv (stem, ref_id, L, tm_template) and the PDBs under PDB_DIR.
Usage: [TEMPLATE_ROOT=...] python t7_template_tm.py ROWS.jsonl OUT.tsv PDB_DIR
Every template's rewind_steps[slot] must equal the ref_id's rw (same template version the index was built from).
"""

import json
import os
import subprocess
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_c2c_structures import usalign_tm
from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.openfold_stub.np.residue_constants import atom_types, restype_1to3, restypes
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

S = "/orcd/scratch/orcd/011/chenxiou"
POOL = os.environ.get("TEMPLATE_ROOT", f"{S}/t2_pool_auth")  # shard{crc32}/<auth>.npz; T2 band tree copy (SuperCloud ~/pp1c_work/templates_band)
ALIAS = f"{S}/synth_alias/auth_key.tsv"
DATA = "/orcd/pool/006/chenxiou/proteina/data/pdb_train"
USALIGN = "/home/chenxiou/.local/bin/USalign"


def write_atom37_pdb(path, coords, mask, aatype):
    """coords [L,37,3], mask [L,37], aatype [L] (openfold restype index); residues numbered 1..n over kept residues."""
    lines, serial, n = [], 1, 0
    for i in range(coords.shape[0]):
        if mask[i, 1] < 0.5:  # no CA -> not a modelled residue
            continue
        n += 1
        aa3 = restype_1to3.get(restypes[int(aatype[i])], "UNK") if int(aatype[i]) < 20 else "UNK"
        for j, nm in enumerate(atom_types):
            if mask[i, j] < 0.5:
                continue
            x, y, z = (float(v) for v in coords[i, j])
            lines.append(f"ATOM  {serial:>5d} {nm:<4s} {aa3:>3s} A{n:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {nm[0]:>2s}")
            serial += 1
    open(path, "w").write("\n".join(lines + ["END"]) + "\n")
    return n


rows_file, out, pdb_dir = sys.argv[1:4]
os.makedirs(pdb_dir, exist_ok=True)
alias = {}
with open(ALIAS) as fh:
    head = fh.readline().rstrip("\n").split("\t")
    il, ia = head.index("label_id"), head.index("auth_id")
    for line in fh:
        f = line.rstrip("\n").split("\t")
        alias[f[il]] = f[ia]
refs = {}
for r in map(json.loads, open(rows_file)):
    refs.setdefault(r["stem"], r["ref_id"])
man = json.load(open(f"{DATA}/shard_manifest.json"))
res = []
for stem, ref in sorted(refs.items()):
    assert ref.split("@")[0] == stem, (stem, ref)
    slot = int(ref.split("#")[1])
    auth = alias[stem]
    npz = np.load(f"{POOL}/shard{zlib.crc32(auth.encode()) % 1000:04d}/{auth}.npz")
    rw = int(ref.split("@rw")[1].split("#")[0])
    assert int(npz["rewind_steps"][slot]) == rw, f"{stem}: slot {slot} rewind {int(npz['rewind_steps'][slot])} != ref_id rw{rw} (wrong template version)"
    t_mask, t_aa = npz["atom_mask"].astype(float), npz["aatype"]
    t_xyz = np.zeros(t_mask.shape + (3,), np.float32)
    t_xyz[t_mask.astype(bool)] = npz["coords"][slot]  # coords hold the PRESENT atoms only, as the index builder scatters them
    g = torch.load(_processed_path_sharded(__import__("pathlib").Path(f"{DATA}/processed"), stem, man), weights_only=False)
    n_xyz = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].numpy()
    n_mask = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].numpy().astype(float)
    n_aa = g.residue_type.numpy()
    tp, nmp = os.path.join(pdb_dir, f"{stem}_template.pdb"), os.path.join(pdb_dir, f"{stem}_native.pdb")
    nt = write_atom37_pdb(tp, t_xyz, t_mask, t_aa)
    nn = write_atom37_pdb(nmp, n_xyz, n_mask, n_aa)
    seq_t = "".join(restypes[int(a)] if int(a) < 20 else "X" for a, m in zip(t_aa, t_mask[:, 1]) if m > 0.5)
    seq_n = "".join(restypes[int(a)] if int(a) < 20 else "X" for a, m in zip(n_aa, n_mask[:, 1]) if m > 0.5)
    same = seq_t == seq_n
    tm = usalign_tm(tp, nmp, USALIGN) if same else float("nan")
    res.append((stem, ref, nn, nt, same, tm))
    print(f"{stem} {ref} native {nn} template {nt} same_seq {same} TM {tm:.3f}", flush=True)
with open(out, "w") as fh:
    fh.write("stem\tref_id\tL_native\tL_template\tsame_seq\ttm_template\n")
    for r in res:
        fh.write("\t".join(str(x) for x in r) + "\n")
bad = [r[0] for r in res if not r[4]]
print(f"[done] {len(res)} templates; sequence mismatch {len(bad)} {bad[:10]}")
