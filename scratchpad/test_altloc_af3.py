"""resolve_altlocs_af3 + protein_to_pyg(resolve_altlocs=True): port of AlphaFold3 ResolveMmcifAltLocs (mmcif_altlocs.cc).

The AF3 repo ships no test for mmcif_altlocs (its header's "See tests" are not open-sourced), so expected values below
are hand-derived from the C++ source.
  U  helpers: AtomEquiv (equal / H<->D prefix swap), GroupBy (the cc comment example; runs compare to the run's FIRST value)
  R1 partial interleaved per atom: highest occupancy per atom name     R2 tie -> 'A' (either file order)
  R3 three alt ids                                                    R4 blocked A-then-B: every group has 1 row -> all kept
  R5 whole microhet: highest total occupancy wins; tie -> alphabetical (either file order)
  R6 whole, one comp: one alt id for every atom (differs from partial); an atom present only in B keeps its row
  R7 whole tie-break quirk: the run's smallest id is compared with the chosen run's BEST id (AF3 literal)
  R8 '.' and '?' are no-altloc and split pending runs                 R9 unparseable occupancy -> 0.0, counted only if read
Through mmCIF files (protein_to_pyg(residue_order="label_seq", resolve_altlocs=True)):
  F1 partial interleaved CA/CB coords   F2 microhet resolved (no SeqresCollideError; pdb_data writes the chain)
  F3 blocked -> AltlocDuplicateError (pdb_data drops it, no auto_excluded entry)   F4 partial microhet -> SeqresCollideError
  F5 unparseable occupancy in a file    F6 reader: '.', '?', numeric alt ids     F7 resolve_altlocs + auth order -> ValueError
  F8 determinism (two calls identical)  F9 hydrogens resolved before deprotonation
  F10 CYS (ATOM) / CSO (HETATM) whole microhet resolved over both record types (file order from _atom_site.id)
  P1-P3 inner tie, mean-not-sum, float32 arithmetic;  A1 residues split by author number + insertion code (AF3 layout)
  L  legacy: default read_pdb_to_dataframe == HEAD's, and resolve_altlocs=False graphs == HEAD graphs on every synthetic
     file (test_mmcif_seqres_parse cases + the altloc files here), default and seqres df, all chains and chain A
Must-raise checks run in a child process (non-zero exit + error class name in stderr).
Usage: python scratchpad/test_altloc_af3.py   (CPU, temp dir only; needs git for the HEAD copy)
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch

from proteinfoundation.graphein_utils.graphein_utils import (_atom_equiv, _group_by, protein_to_pyg,
                                                            read_pdb_to_dataframe, resolve_altlocs_af3)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mmcif_seqres_parse as T  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLIDE = ("9eee", "8bbb", "8ddd")  # microhet: label_seq order without resolve_altlocs raises (HEAD and now)


def rdf(rows):
    """rows: (comp, label_seq, atom, alt, occupancy) -> the columns resolve_altlocs_af3 reads (chain A)."""
    return pd.DataFrame({"chain_id": "A", "label_seq_id": [float(r[1]) for r in rows], "label_comp_id": [r[0] for r in rows],
                         "label_atom_id": [r[2] for r in rows], "label_alt_id": [r[3] for r in rows],
                         "occupancy": [r[4] for r in rows]})


def kept(rows):
    k, n_bad = resolve_altlocs_af3(rdf(rows))
    return k.tolist(), n_bad


def unit_helpers():
    cases = [("CA", "CA", True), ("HA", "DA", True), ("DA", "HA", True), ("H", "D", True), ("HD1", "DD1", True),
             ("HA", "HB", False), ("CA", "CB", False), ("", "CA", False), ("CA", "", False), ("", "", True),
             ("HA", "TA", False), ("DH", "HH", True)]
    for a, b, want in cases:
        assert _atom_equiv(a, b) is want, (a, b)
    assert _group_by(["B", "B", "B", "C", "C"], 0, 5, lambda x, y: x == y) == [(0, 3), (3, 2)]
    assert _group_by(["X", "B", "B", "C", "C", "Y"], 1, 4, lambda x, y: x == y) == [(1, 2), (3, 2)]
    assert _group_by(["HA", "DA", "HA", "CA"], 0, 4, _atom_equiv) == [(0, 3), (3, 1)]
    assert _group_by([], 0, 0, _atom_equiv) == []
    # a run is compared with its FIRST value, not the previous one (visible with a non-transitive predicate)
    assert _group_by([0, 1, 2], 0, 3, lambda x, y: abs(x - y) <= 1) == [(0, 2), (2, 1)]
    print(f"PASS U AtomEquiv {len(cases)} pairs, GroupBy 5 cases")


def unit_resolver():
    # R1 partial interleaved: N '.', CA A .6 / B .4, CB A .3 / B .7
    assert kept([("SER", 1, "N", ".", 1.0), ("SER", 1, "CA", "A", 0.6), ("SER", 1, "CA", "B", 0.4),
                 ("SER", 1, "CB", "A", 0.3), ("SER", 1, "CB", "B", 0.7)]) == ([0, 1, 4], 0)
    # R2 tie -> 'A' in either order
    assert kept([("SER", 1, "N", ".", 1.0), ("SER", 1, "CA", "A", 0.5), ("SER", 1, "CA", "B", 0.5)]) == ([0, 1], 0)
    assert kept([("SER", 1, "N", ".", 1.0), ("SER", 1, "CA", "B", 0.5), ("SER", 1, "CA", "A", 0.5)]) == ([0, 2], 0)
    # R3 three alt ids
    assert kept([("SER", 1, "N", ".", 1.0), ("SER", 1, "CA", "A", 0.3), ("SER", 1, "CA", "B", 0.3),
                 ("SER", 1, "CA", "C", 0.4)]) == ([0, 3], 0)
    assert kept([("SER", 1, "N", ".", 1.0), ("SER", 1, "CA", "C", 0.3), ("SER", 1, "CA", "B", 0.3),
                 ("SER", 1, "CA", "A", 0.3)]) == ([0, 3], 0)
    # R4 blocked A then B: CA, CB, CA, CB are four 1-row groups -> all kept
    assert kept([("SER", 1, "N", ".", 1.0), ("SER", 1, "CA", "A", 0.6), ("SER", 1, "CB", "A", 0.6),
                 ("SER", 1, "CA", "B", 0.4), ("SER", 1, "CB", "B", 0.4)]) == ([0, 1, 2, 3, 4], 0)
    # R5 whole microhet (every row has an alt id), plus neighbour residues without altlocs
    mh = lambda oa, ob: [("ALA", 1, "CA", ".", 1.0), ("SER", 2, "N", "A", oa), ("SER", 2, "CA", "A", oa),
                         ("THR", 2, "N", "B", ob), ("THR", 2, "CA", "B", ob), ("GLY", 3, "CA", ".", 1.0)]
    assert kept(mh(0.6, 0.4)) == ([0, 1, 2, 5], 0)
    assert kept(mh(0.4, 0.6)) == ([0, 3, 4, 5], 0)
    assert kept(mh(0.5, 0.5)) == ([0, 1, 2, 5], 0)
    assert kept([("THR", 2, "N", "B", 0.5), ("THR", 2, "CA", "B", 0.5), ("SER", 2, "N", "A", 0.5),
                 ("SER", 2, "CA", "A", 0.5)]) == ([2, 3], 0)
    # R6 whole, one comp: mean A .55 > B (.1+.8+.1)/3 -> A for N and CA (partial would take CA B); CB only in B kept
    assert kept([("SER", 1, "N", "A", 0.9), ("SER", 1, "N", "B", 0.1), ("SER", 1, "CA", "A", 0.2),
                 ("SER", 1, "CA", "B", 0.8), ("SER", 1, "CB", "B", 0.1)]) == ([0, 2, 4], 0)
    # the mean is per alt id over its own rows: B's extra CB .8 lifts B to (.1+.8+.8)/3 > .55 -> B everywhere
    assert kept([("SER", 1, "N", "A", 0.9), ("SER", 1, "N", "B", 0.1), ("SER", 1, "CA", "A", 0.2),
                 ("SER", 1, "CA", "B", 0.8), ("SER", 1, "CB", "B", 0.8)]) == ([1, 3, 4], 0)
    assert kept([("SER", 1, "N", "B", 0.1), ("SER", 1, "N", "A", 0.9)]) == ([1], 0)
    # R7 quirk: SER total .25+.75=1 (best id C), THR B total 1; tie compares min('B') with SER's BEST id 'C' -> THR
    assert kept([("SER", 1, "N", "A", 0.25), ("SER", 1, "CA", "C", 0.75), ("THR", 1, "N", "B", 1.0)]) == ([2], 0)
    # R8 '.' and '?' are no-altloc; each splits the pending run (CA A/B resolved before '?' N, then CB group of 1)
    assert kept([("SER", 1, "CA", "A", 0.4), ("SER", 1, "CA", "B", 0.6), ("SER", 1, "N", "?", 1.0),
                 ("SER", 1, "CB", "A", 0.2), ("SER", 1, "C", ".", 1.0)]) == ([1, 2, 3, 4], 0)
    # R9 'abc' -> 0.0 loses to .3 (counted); a bad value on a no-altloc row or a 1-row group is never read
    assert kept([("SER", 1, "N", ".", "abc"), ("SER", 1, "CA", "A", "abc"), ("SER", 1, "CA", "B", "0.3"),
                 ("SER", 1, "CB", "A", None)]) == ([0, 2, 3], 1)
    assert kept([("SER", 1, "N", ".", 1.0), ("SER", 1, "CA", "A", None), ("SER", 1, "CA", "B", None)]) == ([0, 1], 2)
    # residues are runs of (chain, label_seq): the same atom name in the next residue is a new group
    assert kept([("SER", 1, "CA", "A", 0.4), ("GLY", 2, "CA", "B", 0.6)]) == ([0, 1], 0)
    # P0 whole: within an atom-name run the FIRST later row with the best id wins (the run's first row is the default)
    assert kept([("SER", 1, "N", "B", 0.4), ("SER", 1, "N", "A", 0.5), ("SER", 1, "N", "A", 0.5)]) == ([1], 0)
    # P1 inner tie: one comp, ids B then A at .5 -> A
    assert kept([("SER", 1, "N", "B", 0.5), ("SER", 1, "N", "A", 0.5)]) == ([1], 0)
    # P2 mean, not sum: SER A 3 rows x .4 (mean .4, sum 1.2) vs THR B 1 row .6 -> THR
    assert kept([("SER", 1, "N", "A", 0.4), ("SER", 1, "CA", "A", 0.4), ("SER", 1, "C", "A", 0.4),
                 ("THR", 1, "N", "B", 0.6)]) == ([3], 0)
    # P3 float32: .1f + .2f == .3f ties (then the R7 quirk picks THR); float64 .1 + .2 > .3 would keep SER [0, 1]
    assert kept([("SER", 1, "N", "A", 0.1), ("SER", 1, "CA", "C", 0.2), ("THR", 1, "N", "B", 0.3)]) == ([2], 0)
    # A1 author split (AF3 layout with auth_seq_id + ins code): one label_seq, auth 5 / 5A -> two residues, both kept;
    # one auth residue spanning label_seq 1 and 2 -> one residue, partial CA A/B resolved across it
    d = rdf([("SER", 1, "CA", "A", 0.4), ("SER", 1, "CA", "B", 0.6)])
    assert resolve_altlocs_af3(d.assign(residue_number=[5, 5], insertion=["", "A"]))[0].tolist() == [0, 1]
    assert resolve_altlocs_af3(d.assign(residue_number=[5, 5], insertion=["", ""]))[0].tolist() == [1]
    d = rdf([("SER", 1, "N", ".", 1.0), ("SER", 2, "CA", "A", 0.4), ("SER", 2, "CA", "B", 0.6)])
    assert resolve_altlocs_af3(d)[0].tolist() == [0, 2]
    d2 = rdf([("SER", 1, "CA", "A", 0.4), ("SER", 2, "CA", "B", 0.6)])
    assert resolve_altlocs_af3(d2)[0].tolist() == [0, 1]
    assert resolve_altlocs_af3(d2.assign(residue_number=[5, 5], insertion=["", ""]))[0].tolist() == [1]
    print("PASS R1-R9, P1-P3, A1 resolver expected indices (partial, tie, 3 ids, blocked, whole microhet, whole one-comp, "
          "quirk, '.'/'?', bad occupancy, inner tie, mean-not-sum, float32, author residue split)")


def acif(d, code, atoms):
    """test_mmcif_seqres_parse.write_cif with per-atom occupancy (string, may be unparseable)."""
    lines = [f"data_{code.upper()}", f"_entry.id {code.upper()}", "#", "loop_"] + T.HEADER
    for k, a in enumerate(atoms):
        lines.append(f"{a['rec']} {k + 1} {a['el']} {a['name']} {a['alt']} {a['comp']} {a['lasym']} 1 {a['lseq']} "
                     f"{a['ins']} {a['x']:.3f} {k:.3f} 0.000 {a.get('occ', '1.00')} {a['b']:.2f} ? {a['aseq']} "
                     f"{a['comp']} {a['aasym']} {a['name']} 1")
    lines.append("#")
    path = os.path.join(d, f"{code}.cif")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def at(comp, lseq, name, alt=".", occ="1.00", x=0.0, rec="ATOM", el=None):
    a = T.atom(comp, lseq, lseq, x=x, name=name, alt=alt, rec=rec)
    a["occ"] = occ
    if el is not None:
        a["el"] = el
    return a


def res(path, chains="all"):
    return protein_to_pyg(df=read_pdb_to_dataframe(path=path, mmcif_seqres_fields=True), path=path,
                          chain_selection=chains, keep_insertions=True, store_bfactor=True, residue_order="label_seq",
                          resolve_altlocs=True)


def res_cmd(path):
    return ("from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe\n"
            f"protein_to_pyg(df=read_pdb_to_dataframe(path={path!r}, mmcif_seqres_fields=True), path={path!r}, "
            "keep_insertions=True, residue_order='label_seq', resolve_altlocs=True)")


def xs(g, atom37):
    return [round(v, 3) for v in g.coords[:, atom37, 0].tolist()]  # float32 coords; written with 3 decimals


ALT_FILES = {
    "8aaa": [at("ALA", 1, "CA", x=1.0), at("SER", 2, "N", x=2.0), at("SER", 2, "CA", "A", "0.60", 2.1),
             at("SER", 2, "CA", "B", "0.40", 2.2), at("SER", 2, "CB", "A", "0.30", 2.3),
             at("SER", 2, "CB", "B", "0.70", 2.4), at("GLY", 3, "CA", x=3.0)],
    "8bbb": [at("ALA", 1, "CA", x=1.0), at("SER", 2, "N", "A", "0.60", 2.0), at("SER", 2, "CA", "A", "0.60", 2.1),
             at("THR", 2, "N", "B", "0.40", 2.2), at("THR", 2, "CA", "B", "0.40", 2.3), at("GLY", 3, "CA", x=3.0)],
    "8ccc": [at("ALA", 1, "CA", x=1.0), at("SER", 2, "N", x=2.0), at("SER", 2, "CA", "A", "0.60", 2.1),
             at("SER", 2, "CB", "A", "0.60", 2.2), at("SER", 2, "CA", "B", "0.40", 2.3),
             at("SER", 2, "CB", "B", "0.40", 2.4), at("GLY", 3, "CA", x=3.0)],
    "8ddd": [at("ALA", 1, "CA", x=1.0), at("SER", 2, "N", x=2.0), at("SER", 2, "CA", "A", "1.00", 2.1),
             at("THR", 2, "CB", "B", "1.00", 2.2), at("GLY", 3, "CA", x=3.0)],
    "8eee": [at("ALA", 1, "CA", x=1.0), at("SER", 2, "N", x=2.0), at("SER", 2, "CA", "A", "abc", 2.1),
             at("SER", 2, "CA", "B", "0.30", 2.2), at("GLY", 3, "CA", x=3.0)],
    "8fff": [at("ALA", 1, "CA", "?", x=1.0), at("SER", 2, "CA", "1", "0.30", 2.1), at("SER", 2, "CA", "2", "0.70", 2.2)],
    "8ggg": [at("ALA", 1, "N", x=1.0), at("ALA", 1, "CA", x=1.1), at("ALA", 1, "H", x=1.2, el="H"),
             at("SER", 2, "N", "A", "0.60", 2.0), at("SER", 2, "N", "B", "0.40", 2.1),
             at("SER", 2, "CA", "A", "0.30", 2.2), at("SER", 2, "CA", "B", "0.70", 2.3),
             at("SER", 2, "H", ".", "1.00", 2.4, el="H")],
    "8hhh": [at("ALA", 1, "CA", x=1.0), at("CYS", 2, "N", "A", "0.60", 2.0), at("CYS", 2, "CA", "A", "0.60", 2.1),
             at("CSO", 2, "N", "B", "0.40", 2.2, rec="HETATM"), at("CSO", 2, "CA", "B", "0.40", 2.3, rec="HETATM"),
             at("GLY", 3, "CA", x=3.0)],
    "8iii": [at("ALA", 1, "CA", x=1.0), at("CYS", 2, "N", "A", "0.40", 2.0), at("CYS", 2, "CA", "A", "0.40", 2.1),
             at("CSO", 2, "N", "B", "0.60", 2.2, rec="HETATM"), at("CSO", 2, "CA", "B", "0.60", 2.3, rec="HETATM"),
             at("GLY", 3, "CA", x=3.0)],
}


def file_cases(d, data_dir):
    p = {c: acif(d, c, a) for c, a in ALT_FILES.items()}
    # F1 partial: CA A (2.1), CB B (2.4)
    g = res(p["8aaa"])
    assert list(g.residue_id) == ["A:ALA:1:", "A:SER:2:", "A:GLY:3:"] and xs(g, 1) == [1.0, 2.1, 3.0], xs(g, 1)
    assert xs(g, 4)[1] == 2.4 and g.label_seq_id.tolist() == [1, 2, 3], xs(g, 4)  # atom37 index 4 = CB
    print(f"PASS F1 partial: SER CA x {xs(g, 1)[1]} (A .6), CB x {xs(g, 4)[1]} (B .7)")
    # F2 microhet resolved by WHOLE -> SER, no SeqresCollideError; pdb_data writes the chain
    g = res(p["8bbb"])
    assert g.residues == ["ALA", "SER", "GLY"] and xs(g, 1) == [1.0, 2.1, 3.0] and xs(g, 0)[1] == 2.0, (g.residues, xs(g, 1))
    T.must_raise(T.fixed_cmd(p["8bbb"]), "SeqresCollideError")
    raw = os.path.join(data_dir, "raw")
    for c in ("8bbb", "8ccc"):
        acif(raw, c, ALT_FILES[c])
    out, processed = T.process_static(data_dir, "8bbb", "A", True)
    assert out == "8bbb_A.pt", out
    gp = torch.load(processed / out, weights_only=False)
    assert list(gp.residue_id) == ["A:ALA:1:", "A:SER:2:", "A:GLY:3:"] and gp.sequence == "ASG", gp.residue_id
    print(f"PASS F2 microhet SER .6 / THR .4 -> {g.residues} (legacy-order path still collides); pdb_data wrote {out}")
    # F3 blocked -> AltlocDuplicateError; pdb_data drops it without an auto_excluded entry
    T.must_raise(res_cmd(p["8ccc"]), "AltlocDuplicateError")
    out, processed = T.process_static(data_dir, "8ccc", "A", True)
    assert out is None and not (processed.parent / "auto_excluded_non_canonical.txt").exists(), out
    print("PASS F3 blocked A-then-B -> AltlocDuplicateError; pdb_data returned None, no auto_excluded entry")
    # F4 partial microhet (SER N without alt id): PARTIAL keeps SER CA and THR CB -> collide check fires
    T.must_raise(res_cmd(p["8ddd"]), "SeqresCollideError")
    print("PASS F4 partial microhet -> SeqresCollideError")
    # F5 unparseable occupancy 'abc' -> 0.0 < .3 -> B
    df5 = read_pdb_to_dataframe(path=p["8eee"], mmcif_seqres_fields=True)
    assert resolve_altlocs_af3(df5)[1] == 1 and xs(res(p["8eee"]), 1) == [1.0, 2.2, 3.0]
    print("PASS F5 occupancy 'abc' read as 0.0 (1 counted), CA B kept")
    # F6 reader: '.' and '?' -> '.', numeric ids kept as strings; '2' (.7) wins
    assert read_pdb_to_dataframe(path=p["8fff"], mmcif_seqres_fields=True)["label_alt_id"].tolist() == [".", "1", "2"]
    assert read_pdb_to_dataframe(path=p["8aaa"], mmcif_seqres_fields=True)["label_alt_id"].tolist() == \
        [".", ".", "A", "B", "A", "B", "."]
    assert xs(res(p["8fff"]), 1) == [1.0, 2.2]
    print("PASS F6 reader alt ids '.','?' -> '.', numeric '1'/'2' kept; numeric id resolution")
    # F7
    T.must_raise("from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe\n"
                 f"protein_to_pyg(df=read_pdb_to_dataframe(path={p['8aaa']!r}, mmcif_seqres_fields=True), "
                 f"path={p['8aaa']!r}, resolve_altlocs=True)", "ValueError: resolve_altlocs needs")
    T.must_raise("from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe\n"
                 f"df = read_pdb_to_dataframe(path={p['8aaa']!r}, mmcif_seqres_fields=True).drop(columns='label_alt_id')\n"
                 f"protein_to_pyg(df=df, path={p['8aaa']!r}, residue_order='label_seq', resolve_altlocs=True)",
                 "SeqresMissingError")
    print("PASS F7 resolve_altlocs with auth order -> ValueError; without label_alt_id -> SeqresMissingError")
    # F8 determinism
    for c in ("8aaa", "8bbb", "8eee", "8ggg"):
        T.assert_graphs_equal(res(p[c]), res(p[c]))
    print("PASS F8 two calls identical on 4 files")
    # F9 SER: the '.' hydrogen makes it PARTIAL (N A .6, CA B .7); stripping H first would have made it WHOLE (A)
    g = res(p["8ggg"])
    assert xs(g, 0) == [1.0, 2.0] and xs(g, 1) == [1.1, 2.3], (xs(g, 0), xs(g, 1))
    print(f"PASS F9 resolution before deprotonation: SER N x {xs(g, 0)[1]}, CA x {xs(g, 1)[1]} (PARTIAL)")
    # F10 CYS A (ATOM) / CSO B (HETATM) on one position: WHOLE over both record types, as AF3
    g = res(p["8hhh"])
    assert g.residues == ["ALA", "CYS", "GLY"] and xs(g, 1) == [1.0, 2.1, 3.0], (g.residues, xs(g, 1))
    g = res(p["8iii"])  # CSO wins -> CYS rows are gone and the HETATM CSO is not a standard residue
    assert g.residues == ["ALA", "GLY"] and g.label_seq_id.tolist() == [1, 3], g.residues
    g = protein_to_pyg(df=read_pdb_to_dataframe(path=p["8iii"], mmcif_seqres_fields=True), path=p["8iii"],
                       keep_insertions=True, residue_order="label_seq", resolve_altlocs=True, keep_hets=["CSO"],
                       remove_nonstandard=False)
    assert list(g.residue_id) == ["A:ALA:1:", "A:CSO:2:", "A:GLY:3:"] and xs(g, 1) == [1.0, 2.3, 3.0], g.residue_id
    print("PASS F10 CYS(ATOM)/CSO(HETATM) microhet: CYS .6 -> CYS; CSO .6 -> residue dropped (HETATM), "
          "kept as resolved het with keep_hets=['CSO']")
    return p


def head_module(d):
    src = subprocess.run(["git", "show", "HEAD:proteinfoundation/graphein_utils/graphein_utils.py"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(d, "gu_head.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("gu_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legacy_identity(d, alt_paths):
    head = head_module(d)
    sets = {"9aaa": [T.atom("SER", 1, 82, "?", 10.0), T.atom("SER", 2, 82, "A", 20.0), T.atom("SER", 3, 82, "B", 30.0),
                     T.atom("GLY", 4, 83, "?", 40.0)],
            "9bbb": [T.atom("ALA", 1, 100, x=1.0), T.atom("GLY", 2, 101, x=2.0), T.atom("SER", 3, 1, x=3.0),
                     T.atom("THR", 4, 2, x=4.0)],
            "9ccc": [T.atom("SER", 2, 82, "A", 20.0), T.atom("SER", 1, 82, "?", 10.0), T.atom("GLY", 3, 83, "?", 30.0)],
            "9ddd": [T.atom("ALA", 1, 1, x=1.0, name="N"), T.atom("ALA", 1, 1, x=1.5), T.atom("SER", 2, 2, x=2.0, alt="A"),
                     T.atom("SER", 2, 2, x=2.5, alt="B"), T.atom("GLY", 3, 3, x=3.0)],
            "9eee": [T.atom("ALA", 1, 1, x=1.0), T.atom("SER", 2, 2, x=2.0, alt="A"), T.atom("THR", 2, 2, x=2.5, alt="B"),
                     T.atom("GLY", 3, 3, x=3.0)],
            "9fff": [T.atom("ALA", 1, 1, x=1.0), T.atom("MSE", 2, 2, x=2.0, rec="HETATM"), T.atom("GLY", 3, 3, x=3.0)],
            "9ggg": T.case7_atoms(),
            "9hhh": [T.atom("ALA", 1, 1, "?", 1.0), T.atom("GLY", 2, 2, ".", 2.0), T.atom("GLY", 3, 2, "A", 3.0)]}
    paths = [T.write_cif(d, c, a) for c, a in sets.items()] + list(alt_paths.values())
    n_df = n_graph = 0
    for p in paths:
        dh, dn = head.read_pdb_to_dataframe(path=p), read_pdb_to_dataframe(path=p)
        assert dh.equals(dn) and list(dh.columns) == list(dn.columns), p
        n_df += 1
        for ch in ("all", ["A"]):
            kw = dict(path=p, chain_selection=ch, keep_insertions=True, store_bfactor=True)
            T.assert_graphs_equal(head.protein_to_pyg(df=dh.copy(), **kw), protein_to_pyg(df=dn.copy(), **kw))
            n_graph += 1
            sh = head.read_pdb_to_dataframe(path=p, mmcif_seqres_fields=True)
            sn = read_pdb_to_dataframe(path=p, mmcif_seqres_fields=True)
            assert sh.equals(sn[sh.columns]), p
            if os.path.basename(p)[:4] in COLLIDE:
                continue
            kw["residue_order"] = "label_seq"
            T.assert_graphs_equal(head.protein_to_pyg(df=sh.copy(), **kw), protein_to_pyg(df=sn.copy(), **kw))
            n_graph += 1
    for c in COLLIDE:
        T.must_raise(T.fixed_cmd(os.path.join(d, f"{c}.cif")), "SeqresCollideError")
    print(f"PASS L legacy: {n_df} default dfs == HEAD; {n_graph} graphs (resolve_altlocs=False) == HEAD over "
          f"{len(paths)} files x (all, A) x (default df, seqres df + label_seq; {len(COLLIDE)} microhet files: "
          "SeqresCollideError as at HEAD)")


def main():
    d = tempfile.mkdtemp()
    data_dir = os.path.join(d, "pdb_train")
    os.makedirs(os.path.join(data_dir, "raw"))
    unit_helpers()
    unit_resolver()
    alt_paths = file_cases(d, data_dir)
    legacy_identity(d, alt_paths)
    print("ALL PASS")


if __name__ == "__main__":
    main()
