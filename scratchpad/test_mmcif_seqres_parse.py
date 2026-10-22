"""read_pdb_to_dataframe(mmcif_seqres_fields=True) + protein_to_pyg(residue_order="label_seq") (repair plan S1).

Synthetic mmCIFs, each run through the legacy path (defaults) and the fixed path:
  1 9aaa SER 82, 82A, 82B, GLY 83: legacy merges to 2 positions; fixed keeps 4 with their own CA, label_seq 1-4
  2 9bbb author 100, 101, 1, 2 at label 1-4: legacy sorts by author number; fixed follows label_seq
  3 9ccc 82A listed before 82 in the file: fixed still orders by label_seq
  4 9ddd same-resname altloc A/B on one position: both paths agree (pinned: altloc B CA wins)
  5 9eee SER/THR microheterogeneity on one label_seq_id -> SeqresCollideError (and pdb_data drops it without
    touching auto_excluded_non_canonical.txt); 9iii one id at two label_seq_id -> SeqresMergeError;
    label_seq mode on a legacy df -> SeqresMissingError
  6 9fff HETATM MSE inside the chain: dropped in both paths
  7 9ggg two chains, no insertion codes, monotone numbering, water '.': fixed == legacy on every field
  8 9hhh insertion codes '?' and '.' (biopandas turns both into the string 'None') -> ""
  9 9jjj case-1 chain A + a chain B atom 3 A from the CA of 82A only: through _process_single_pdb, ext_lig is set on
    82A with the flag on; legacy's merged 82 keeps the last CA (82B) and misses the contact
Cases 1 and 7 also go through PDBLightningDataModule._process_single_pdb / _load_and_process_pdb.
Must-raise checks run in a child process (non-zero exit + error class name in stderr).
Usage: python scratchpad/test_mmcif_seqres_parse.py   (CPU, temp dir only)
"""

import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import PDBLightningDataModule
from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = ["_atom_site.{}".format(c) for c in (
    "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id label_seq_id "
    "pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv pdbx_formal_charge auth_seq_id auth_comp_id "
    "auth_asym_id auth_atom_id pdbx_PDB_model_num").split()]


def atom(comp, lseq, aseq, ins="?", x=0.0, name="CA", rec="ATOM", alt=".", lasym="A", aasym="A", b=0.0):
    return dict(rec=rec, name=name, el=name[0], alt=alt, comp=comp, lasym=lasym, lseq=lseq, aseq=aseq, ins=ins, x=x,
                aasym=aasym, b=b)


def write_cif(d, code, atoms):
    lines = [f"data_{code.upper()}", f"_entry.id {code.upper()}", "#", "loop_"] + HEADER
    for k, a in enumerate(atoms):
        lines.append(f"{a['rec']} {k + 1} {a['el']} {a['name']} {a['alt']} {a['comp']} {a['lasym']} 1 {a['lseq']} "
                     f"{a['ins']} {a['x']:.3f} {k:.3f} 0.000 1.00 {a['b']:.2f} ? {a['aseq']} {a['comp']} {a['aasym']} "
                     f"{a['name']} 1")
    lines.append("#")
    path = os.path.join(d, f"{code}.cif")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def legacy(path, chains="all"):
    return protein_to_pyg(df=read_pdb_to_dataframe(path=path), path=path, chain_selection=chains,
                          keep_insertions=True, store_bfactor=True)


def fixed(path, chains="all"):
    return protein_to_pyg(df=read_pdb_to_dataframe(path=path, mmcif_seqres_fields=True), path=path,
                          chain_selection=chains, keep_insertions=True, store_bfactor=True, residue_order="label_seq")


def ca_x(g):
    return g.coords[:, 1, 0].tolist()  # atom37 index 1 = CA


def assert_graphs_equal(a, b, extra=()):
    assert set(b.keys()) == set(a.keys()) | set(extra), (sorted(a.keys()), sorted(b.keys()))
    for k in a.keys():
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor):
            assert va.dtype == vb.dtype and torch.equal(va, vb), k
        else:
            assert va == vb, (k, va, vb)


def must_raise(code, err):
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=os.environ.copy(), capture_output=True, text=True)
    assert p.returncode != 0 and err in p.stderr, (p.returncode, p.stderr[-2000:])


def fixed_cmd(path):
    return ("from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe\n"
            f"protein_to_pyg(df=read_pdb_to_dataframe(path={path!r}, mmcif_seqres_fields=True), path={path!r}, "
            "keep_insertions=True, residue_order='label_seq')")


def process_static(data_dir, code, chain, seqres_order):
    processed = pathlib.Path(data_dir) / ("processed_fix" if seqres_order else "processed_legacy")
    out = PDBLightningDataModule._process_single_pdb(
        (0, code, chain), raw_dir=pathlib.Path(data_dir) / "raw", processed_dir=processed, format_type="cif",
        store_het=False, store_bfactor=True, database_tag="pdb", shard_manifest=None, seqres_order=seqres_order)
    return out, processed


def case1(d):
    p = write_cif(d, "9aaa", [atom("SER", 1, 82, "?", 10.0), atom("SER", 2, 82, "A", 20.0),
                              atom("SER", 3, 82, "B", 30.0), atom("GLY", 4, 83, "?", 40.0)])
    gl = legacy(p)
    assert list(gl.residue_id) == ["A:SER:82:", "A:GLY:83:"], gl.residue_id
    gf = fixed(p)
    assert list(gf.residue_id) == ["A:SER:82:", "A:SER:82:A", "A:SER:82:B", "A:GLY:83:"], gf.residue_id
    assert ca_x(gf) == [10.0, 20.0, 30.0, 40.0], ca_x(gf)
    assert gf.label_seq_id.tolist() == [1, 2, 3, 4] and gf.label_seq_id.dtype == torch.int64, gf.label_seq_id
    assert gf.residues == ["SER", "SER", "SER", "GLY"] and gf.chains.tolist() == [0, 0, 0, 0]
    assert not any(r.endswith(":None") for r in gf.residue_id)
    print(f"PASS 1 82/82A/82B: legacy {len(gl.residue_id)} positions (bug), fixed {list(gf.residue_id)} CA x {ca_x(gf)}")
    return p


def case2(d):
    p = write_cif(d, "9bbb", [atom("ALA", 1, 100, x=1.0), atom("GLY", 2, 101, x=2.0),
                              atom("SER", 3, 1, x=3.0), atom("THR", 4, 2, x=4.0)])
    gl, gf = legacy(p), fixed(p)
    assert gl.residues == ["SER", "THR", "ALA", "GLY"] and ca_x(gl) == [3.0, 4.0, 1.0, 2.0], (gl.residues, ca_x(gl))
    assert gf.residues == ["ALA", "GLY", "SER", "THR"] and ca_x(gf) == [1.0, 2.0, 3.0, 4.0], (gf.residues, ca_x(gf))
    assert [int(r.split(":")[2]) for r in gf.residue_id] == [100, 101, 1, 2] and gf.label_seq_id.tolist() == [1, 2, 3, 4]
    print(f"PASS 2 author 100,101,1,2: legacy {gl.residues}, fixed {gf.residues}")


def case3(d):
    p = write_cif(d, "9ccc", [atom("SER", 2, 82, "A", 20.0), atom("SER", 1, 82, "?", 10.0),
                              atom("GLY", 3, 83, "?", 30.0)])
    gf = fixed(p)
    assert list(gf.residue_id) == ["A:SER:82:", "A:SER:82:A", "A:GLY:83:"], gf.residue_id
    assert ca_x(gf) == [10.0, 20.0, 30.0] and gf.label_seq_id.tolist() == [1, 2, 3], (ca_x(gf), gf.label_seq_id)
    print(f"PASS 3 82A listed first: fixed {list(gf.residue_id)}")


def case4(d):
    p = write_cif(d, "9ddd", [atom("ALA", 1, 1, x=1.0, name="N"), atom("ALA", 1, 1, x=1.5),
                              atom("SER", 2, 2, x=2.0, alt="A"), atom("SER", 2, 2, x=2.5, alt="B"),
                              atom("GLY", 3, 3, x=3.0)])
    gl, gf = legacy(p), fixed(p)
    assert_graphs_equal(gl, gf, extra=("label_seq_id",))
    assert list(gf.residue_id) == ["A:ALA:1:", "A:SER:2:", "A:GLY:3:"] and ca_x(gf) == [1.5, 2.5, 3.0], ca_x(gf)
    print(f"PASS 4 altloc A/B same resname: both paths {list(gf.residue_id)}, SER CA x = altloc B {ca_x(gf)[1]}")


def case5(d, data_dir):
    p = write_cif(d, "9eee", [atom("ALA", 1, 1, x=1.0), atom("SER", 2, 2, x=2.0, alt="A"),
                              atom("THR", 2, 2, x=2.5, alt="B"), atom("GLY", 3, 3, x=3.0)])
    assert legacy(p).residues == ["ALA", "SER", "THR", "GLY"]
    must_raise(fixed_cmd(p), "SeqresCollideError")
    write_cif(os.path.join(data_dir, "raw"), "9eee", [atom("ALA", 1, 1, x=1.0), atom("SER", 2, 2, x=2.0, alt="A"),
                                                      atom("THR", 2, 2, x=2.5, alt="B"), atom("GLY", 3, 3, x=3.0)])
    out, processed = process_static(data_dir, "9eee", "A", True)
    assert out is None and not (processed.parent / "auto_excluded_non_canonical.txt").exists()
    pm = write_cif(d, "9iii", [atom("ALA", 1, 1, x=1.0), atom("LYS", 2, 21, x=2.0), atom("LYS", 3, 21, x=3.0)])
    must_raise(fixed_cmd(pm), "SeqresMergeError")
    must_raise("from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe\n"
               f"protein_to_pyg(df=read_pdb_to_dataframe(path={p!r}), path={p!r}, residue_order='label_seq')",
               "SeqresMissingError")
    print("PASS 5 microhet -> SeqresCollideError (pdb_data: dropped, no auto_excluded entry); "
          "merge -> SeqresMergeError; legacy df -> SeqresMissingError")


def case6(d):
    p = write_cif(d, "9fff", [atom("ALA", 1, 1, x=1.0), atom("MSE", 2, 2, x=2.0, rec="HETATM"),
                              atom("GLY", 3, 3, x=3.0)])
    gl, gf = legacy(p), fixed(p)
    assert list(gl.residue_id) == list(gf.residue_id) == ["A:ALA:1:", "A:GLY:3:"], (gl.residue_id, gf.residue_id)
    assert gf.label_seq_id.tolist() == [1, 3]
    print(f"PASS 6 HETATM MSE dropped in both paths: {list(gf.residue_id)}, label_seq {gf.label_seq_id.tolist()}")


def case7_atoms():
    atoms = []
    for k, (comp, lasym, aasym, lseq, aseq) in enumerate([("ALA", "A", "A", 1, 5), ("GLY", "A", "A", 2, 6),
                                                          ("SER", "A", "A", 3, 7), ("LYS", "B", "B", 1, 1),
                                                          ("LEU", "B", "B", 2, 2)]):
        for j, name in enumerate(("N", "CA", "C")):
            atoms.append(atom(comp, lseq, aseq, x=3.8 * k + 0.5 * j, name=name, lasym=lasym, aasym=aasym,
                              b=10.0 * k + j))
    atoms.append(atom("HOH", ".", 101, x=4.0, name="O", rec="HETATM", lasym="C", aasym="A"))
    return atoms


def case7(d):
    p = write_cif(d, "9ggg", case7_atoms())
    gl, gf = legacy(p), fixed(p)
    assert_graphs_equal(gl, gf, extra=("label_seq_id",))
    assert gf.label_seq_id.tolist() == [1, 2, 3, 1, 2] and len(gf.residue_id) == 5
    for ch in ("A", "B"):
        assert_graphs_equal(legacy(p, [ch]), fixed(p, [ch]), extra=("label_seq_id",))
    print(f"PASS 7 no insertion codes: fixed == legacy on {sorted(gl.keys())} (all chains, A, B)")


def case8(d):
    p = write_cif(d, "9hhh", [atom("ALA", 1, 1, "?", 1.0), atom("GLY", 2, 2, ".", 2.0), atom("GLY", 3, 2, "A", 3.0)])
    df = read_pdb_to_dataframe(path=p, mmcif_seqres_fields=True)
    assert df["insertion"].tolist() == ["", "", "A"], df["insertion"].tolist()
    gf = fixed(p)
    assert list(gf.residue_id) == ["A:ALA:1:", "A:GLY:2:", "A:GLY:2:A"], gf.residue_id
    print(f"PASS 8 '?' and '.' -> '': {list(gf.residue_id)}")


def case9(data_dir):
    write_cif(os.path.join(data_dir, "raw"), "9jjj", [
        atom("SER", 1, 82, "?", 10.0), atom("SER", 2, 82, "A", 20.0), atom("SER", 3, 82, "B", 30.0),
        atom("GLY", 4, 83, "?", 40.0), atom("LYS", 1, 1, x=20.0, lasym="B", aasym="B")])  # y = row 4 -> 3 A from 82A
    graphs = {}
    for so in (False, True):
        out, processed = process_static(data_dir, "9jjj", "A", so)
        assert out == "9jjj_A.pt", out
        graphs[so] = torch.load(processed / out, weights_only=False)
    assert graphs[True].ext_lig.tolist() == [0, 1, 0, 0], (graphs[True].residue_id, graphs[True].ext_lig)
    assert graphs[False].ext_lig.tolist() == [0, 0], graphs[False].ext_lig
    print(f"PASS 9 ext_lig near 82A: fixed {dict(zip(graphs[True].residue_id, graphs[True].ext_lig.tolist()))}, "
          f"legacy {graphs[False].ext_lig.tolist()}")


def entry_points(data_dir):
    raw = os.path.join(data_dir, "raw")
    write_cif(raw, "9aaa", [atom("SER", 1, 82, "?", 10.0), atom("SER", 2, 82, "A", 20.0),
                            atom("SER", 3, 82, "B", 30.0), atom("GLY", 4, 83, "?", 40.0)])
    write_cif(raw, "9ggg", case7_atoms())
    loaded = {}
    for code in ("9aaa", "9ggg"):
        for so in (False, True):
            out, processed = process_static(data_dir, code, "A", so)
            assert out == f"{code}_A.pt", out
            loaded[code, so] = torch.load(processed / out, weights_only=False)
    a_old, a_new = loaded["9aaa", False], loaded["9aaa", True]
    assert list(a_old.residue_id) == ["A:SER:82:", "A:GLY:83:"], a_old.residue_id
    assert list(a_new.residue_id) == ["A:SER:82:", "A:SER:82:A", "A:SER:82:B", "A:GLY:83:"], a_new.residue_id
    assert set(a_new.keys()) == set(a_old.keys()) and "label_seq_id" not in a_new.keys(), sorted(a_new.keys())
    assert ca_x(a_new) == [10.0, 20.0, 30.0, 40.0] and a_new.residue_pdb_idx.tolist() == [82, 82, 82, 83]
    assert a_new.sequence == "SSSG" and a_new.coord_mask.shape == (4, 37) and a_new.ext_lig.shape == (4,)
    assert_graphs_equal(loaded["9ggg", False], loaded["9ggg", True])
    assert loaded["9ggg", True].ext_lig.tolist() == [0, 0, 1], loaded["9ggg", True].ext_lig  # SER CA 3.9 A from chain B
    print(f"PASS static _process_single_pdb: 9aaa_A legacy {len(a_old.residue_id)} -> fixed "
          f"{list(a_new.residue_id)}; 9ggg_A identical .pt fields with seqres_order on/off")
    for so in (False, True):
        dm = PDBLightningDataModule(data_dir=data_dir, dataselector=None, seqres_order=so, num_workers=1)
        dm.processed_dir = pathlib.Path(data_dir) / f"processed_inst_{int(so)}"
        assert dm._load_and_process_pdb((0, "9aaa", "A")) == "9aaa_A.pt"
        g = torch.load(dm.processed_dir / "9aaa_A.pt", weights_only=False)
        assert len(g.residue_id) == (4 if so else 2) and "label_seq_id" not in g.keys(), (so, g.residue_id)
    print("PASS instance _load_and_process_pdb: 9aaa_A 2 positions (off) / 4 positions (on)")


def main():
    d = tempfile.mkdtemp()
    data_dir = os.path.join(d, "pdb_train")
    os.makedirs(os.path.join(data_dir, "raw"))
    case1(d)
    case2(d)
    case3(d)
    case4(d)
    case5(d, data_dir)
    case6(d)
    case7(d)
    case8(d)
    case9(data_dir)
    entry_points(data_dir)
    print("ALL PASS")


if __name__ == "__main__":
    main()
