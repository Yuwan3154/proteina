"""reprocess_seqres_chains: old_to_new altloc matching + --all-stems / --write-changed-only classification (local fixture).

Every fixture but 7hhh gets 8 trailing GLY (label 101-108; pydssp needs a few residues). Stored "old" pack entries are made by the real path with seqres_order=False (+ a dummy contact_map_confind, which the
comparison must skip); main() then runs as a child process on a 2-part split.
  7aaa no altloc, no insertion code                       -> unchanged (nothing written)
  7bbb CA altloc A/B occupancy tie (old scatter kept B, AF3 keeps A) -> changed, backbone_changed, n_matched_altloc 1
  7ccc CB altloc A .6 / B .4 (old kept B)                 -> changed (coords), backbone unchanged
  7ddd GLY 2A (insertion code only)                       -> changed, fields residue_id only, rid_exact False, rid_mod_ins True
  7eee SER 82/82A/82B (merged, 2 -> 4 residues)           -> changed, backbone_changed, n_unmatched 0
  7fff SER .6 / THR .4 microhet (collide)                 -> all-stems: processed, changed, n_unmatched 1 (THR) recorded
  7ggg no raw mmCIF                                       -> failed:process_returned_none, no raw_mtime
  7hhh 513 residues + a CA altloc tie                     -> dropped_over512
  7iii 513 residues, no altloc (unchanged)                -> dropped_over512 (the length check comes first)
  7jjj = 7aaa with the stored CA of residue 1 moved 9 A   -> failed:unmatched row, part completes, nothing written
  7kkk = 7eee with the stored merged SER 82 CA = (x of 82A, y of 82, z) -> match 3 (union of candidates), winner = tie
       82 / 82A on 2 components -> lowest new index 0, chimera
  7lll bucketed raw/ll/7lll.cif (pack built from it) + a flat raw/7lll.cif with SER 3 moved -> raw_replaced, raw_used =
       flat, SER 3 matched by residue_id (match 4, n_matched_id 1), nothing fails
label_seq sidecar: every processed stem (all but 7ggg), unchanged 7aaa included; n_stems summed over parts == 10.
Stems mode (--stems [7eee_A, 7fff_A]): 7eee written, 7fff dropped_collide.
old_to_new unit: 7bbb without altloc rows -> unmatched (the reviewer's 9ddd repro); per-component chimera (1, 5, 3) of
rows (1, 2, 3) / (4, 5, 6) -> match 2; (1, 5, 7) -> 0.
Usage: python scratchpad/test_reprocess_seqres_chains.py   (CPU, temp dir only)
"""

import gzip
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch_geometric.data import Data

from proteinfoundation.graphein_utils.graphein_utils import read_pdb_to_dataframe
from proteinfoundation.utils.reprocess_seqres_chains import COLUMNS, altloc_rows, diff_fields, old_to_new, reprocess_chain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_altloc_af3 as A  # noqa: E402
import raw_version_audit as R  # noqa: E402
import test_mmcif_seqres_parse as T  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bb(comp, lseq, x, aseq=None, ins="?", alt_ca=None, alt_cb=None):
    """N, CA, C, O (+ CB) of one residue; alt_ca / alt_cb = [(alt id, occupancy, x)] replace that atom's row."""
    rows = []
    for j, name in enumerate(("N", "CA", "C", "O", "CB")):
        alts = {"CA": alt_ca, "CB": alt_cb}.get(name)
        if name == "CB" and alt_cb is None:
            continue
        for alt, occ, xx in alts or [(".", "1.00", x + 0.1 * j)]:
            a = T.atom(comp, lseq, lseq if aseq is None else aseq, ins, xx, name=name, alt=alt)
            a["occ"] = occ
            rows.append(a)
    return rows


TAIL = [r for i in range(8) for r in bb("GLY", 101 + i, 101.0 + 3.8 * i)]  # pydssp needs >= ~6 residues
FIX = {
    "7aaa": bb("ALA", 1, 1.0) + bb("GLY", 2, 2.0) + bb("SER", 3, 3.0),
    "7bbb": bb("ALA", 1, 1.0) + bb("SER", 2, 2.0, alt_ca=[("A", "0.50", 2.1), ("B", "0.50", 2.6)]) + bb("GLY", 3, 3.0),
    "7ccc": bb("ALA", 1, 1.0) + bb("SER", 2, 2.0, alt_cb=[("A", "0.60", 2.4), ("B", "0.40", 2.9)]) + bb("GLY", 3, 3.0),
    "7ddd": bb("ALA", 1, 1.0) + bb("GLY", 2, 2.0, aseq=2, ins="A"),
    "7eee": bb("SER", 1, 1.0, aseq=82) + bb("SER", 2, 2.0, aseq=82, ins="A") + bb("SER", 3, 3.0, aseq=82, ins="B")
            + bb("GLY", 4, 4.0, aseq=83),
    "7fff": bb("ALA", 1, 1.0) + [dict(r, alt="A", occ="0.60") for r in bb("SER", 2, 2.0)]
            + [dict(r, alt="B", occ="0.40") for r in bb("THR", 2, 2.5)] + bb("GLY", 3, 3.0),
    "7hhh": [dict(T.atom("ALA", i, i, x=float(i)), occ="1.00") for i in range(1, 513)]
            + [dict(T.atom("ALA", 513, 513, x=513.0, alt="A"), occ="0.50"),
               dict(T.atom("ALA", 513, 513, x=513.5, alt="B"), occ="0.50")],
    "7iii": [dict(T.atom("ALA", i, i, x=float(i)), occ="1.00") for i in range(1, 514)],
    "7jjj": bb("ALA", 1, 1.0) + bb("GLY", 2, 2.0) + bb("SER", 3, 3.0),
}
FIX["7kkk"] = FIX["7eee"]
CATS = {"7aaa_A": ("ok", 0), "7bbb_A": ("ok", 0), "7ccc_A": ("ok", 0), "7ddd_A": ("ok", 1), "7eee_A": ("merged", 0),
        "7fff_A": ("collide", 0), "7ggg_A": ("ok", 0), "7hhh_A": ("ok", 0), "7iii_A": ("ok", 0), "7jjj_A": ("ok", 0),
        "7kkk_A": ("merged", 0), "7lll_A": ("ok", 0)}
MAN = {"depth_table": {"ll": 2}}  # only 7lll is bucketed (raw/ll/)
HIST_LOOP = """loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 2019-01-01
2 'Structure model' 1 1 2021-03-04
3 'Structure model' 1 2 2020-07-08
#
"""
HIST_SINGLE = """_pdbx_audit_revision_history.ordinal             1
_pdbx_audit_revision_history.data_content_type   'Structure model'
_pdbx_audit_revision_history.revision_date       2018-05-06
_pdbx_audit_revision_history.part_number         ?
#
"""
LLL_BUCKET = bb("ALA", 1, 1.0) + bb("GLY", 2, 2.0) + bb("SER", 3, 3.0)
LLL_FLAT = bb("ALA", 1, 1.0) + bb("GLY", 2, 2.0) + bb("SER", 3, 3.5)


def build(root):
    data_dir = root / "pdb_train"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "shard_manifest.json").write_text(json.dumps(MAN))
    for c, atoms in FIX.items():
        A.acif(str(data_dir / "raw"), c, atoms if c in ("7hhh", "7iii") else atoms + TAIL)
    (data_dir / "raw" / "ll").mkdir()
    with open(A.acif(str(data_dir / "raw" / "ll"), "7lll", LLL_BUCKET + TAIL), "a") as f:
        f.write(HIST_SINGLE)
    olds = {}
    for stem in sorted(CATS):
        if stem == "7ggg_A":
            continue
        status, g = reprocess_chain(stem, data_dir / "raw", root / "stage_old" / "processed", MAN, False, Data())
        assert g is not None, (stem, status)
        g.contact_map_confind = torch.zeros(len(g.residue_id), len(g.residue_id), dtype=torch.int8)
        olds[stem] = g
    olds["7ggg_A"] = olds["7aaa_A"].clone()
    olds["7jjj_A"].coords = olds["7jjj_A"].coords.clone()
    olds["7jjj_A"].coords[0, 1, 0] += 9.0
    alt = altloc_rows(read_pdb_to_dataframe(path=str(data_dir / "raw" / "7kkk.cif"), mmcif_seqres_fields=True), "A")
    olds["7kkk_A"].coords = olds["7kkk_A"].coords.clone()
    olds["7kkk_A"].coords[0, 1] = torch.stack([alt["A:SER:82:A"]["CA"][0, 0], alt["A:SER:82:"]["CA"][0, 1],
                                               alt["A:SER:82:"]["CA"][0, 2]])
    with open(A.acif(str(data_dir / "raw"), "7lll", LLL_FLAT + TAIL), "a") as f:  # re-download shadowing the bucket
        f.write(HIST_LOOP)
    pack = root / "fix.pack"
    stems, offs, lens, off = [], [], [], 0
    with open(pack, "wb") as fh:
        for stem in sorted(olds):
            buf = io.BytesIO()
            torch.save(olds[stem], buf)
            b = buf.getvalue()
            fh.write(b)
            stems.append(stem); offs.append(off); lens.append(len(b)); off += len(b)
    np.savez(str(pack) + ".idx.npz", stems=np.array(stems, dtype=object), offsets=np.array(offs, dtype=np.int64),
             lengths=np.array(lens, dtype=np.int64))
    audit = root / "audit.tsv"
    audit.write_text("stem\tstatus\ta\tb\tc\tn_ins\n" + "".join(f"{s}\t{c}\t0\t0\t0\t{n}\n" for s, (c, n) in CATS.items()))
    return data_dir, pack, audit, olds


def run_main(args, out_root):
    for part in (0, 1):
        p = subprocess.run([sys.executable, "-m", "proteinfoundation.utils.reprocess_seqres_chains", *args,
                            "--part", str(part), "--n-parts", "2", "--out-root", str(out_root)], cwd=REPO,
                           env=os.environ.copy(), capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
    rows = {}
    for f in sorted((out_root / "status").glob("*.tsv")):
        lines = f.read_text().splitlines()
        assert tuple(lines[0].split("\t")) == COLUMNS, lines[0]
        for line in lines[1:]:
            r = dict(zip(COLUMNS, line.split("\t")))
            rows[r["stem"]] = r
    side, ls, n_stems = {}, {}, 0
    for f in sorted((out_root / "sidecars").glob("*.pt")):
        sc = torch.load(f, weights_only=False)
        side.update(sc["old_to_new"])
        ls.update(sc["label_seq"])
        n_stems += sc["n_stems"]
    assert n_stems == len(rows), (n_stems, len(rows))
    written = sorted(p.stem for p in (out_root / "processed").rglob("*.pt")) if (out_root / "processed").exists() else []
    return rows, side, ls, written


def all_stems(root, data_dir, pack, audit):
    rows, side, ls, written = run_main(["--pack", str(pack), "--data-dir", str(data_dir), "--audit", str(audit),
                                        "--all-stems", "--write-changed-only"], root / "out_all")
    assert sorted(rows) == sorted(CATS), sorted(rows)
    st = {s: r["status"] for s, r in rows.items()}
    assert st == {"7aaa_A": "unchanged", "7bbb_A": "changed", "7ccc_A": "changed", "7ddd_A": "changed",
                  "7eee_A": "changed", "7fff_A": "changed", "7ggg_A": "failed:process_returned_none",
                  "7hhh_A": "dropped_over512", "7iii_A": "dropped_over512", "7jjj_A": "failed:unmatched",
                  "7kkk_A": "changed", "7lll_A": "changed"}, st
    bc = {s: r["backbone_changed"] for s, r in rows.items()}
    assert bc == {"7aaa_A": "False", "7bbb_A": "True", "7ccc_A": "False", "7ddd_A": "False", "7eee_A": "True",
                  "7fff_A": "True", "7ggg_A": "", "7hhh_A": "True", "7iii_A": "False", "7jjj_A": "True",
                  "7kkk_A": "True", "7lll_A": "True"}, bc
    assert rows["7aaa_A"]["fields"] == "" and rows["7aaa_A"]["rid_exact"] == "True"
    assert "coords" in rows["7bbb_A"]["fields"].split(",") and rows["7bbb_A"]["n_matched_altloc"] == "1"
    assert rows["7bbb_A"]["n_unmatched"] == "0" and rows["7bbb_A"]["n_matched_exact"] == "10"
    assert "coords" in rows["7ccc_A"]["fields"].split(",") and rows["7ccc_A"]["n_matched_exact"] == "11"
    r = rows["7ddd_A"]
    assert (r["fields"], r["rid_exact"], r["rid_mod_ins"]) == ("residue_id", "False", "True"), r
    r = rows["7eee_A"]
    assert (r["L_old"], r["L_new"], r["n_unmatched"], r["rid_mod_ins"]) == ("10", "12", "0", "False"), r
    r = rows["7fff_A"]
    assert (r["category"], r["L_old"], r["L_new"], r["n_unmatched"]) == ("collide", "12", "11", "1"), r
    assert rows["7ggg_A"]["raw_mtime"] == "" and all(rows[s]["raw_mtime"] for s in CATS if s != "7ggg_A")
    assert all("contact_map_confind" not in r["fields"] for r in rows.values())
    changed = ["7bbb_A", "7ccc_A", "7ddd_A", "7eee_A", "7fff_A", "7kkk_A", "7lll_A"]
    assert written == changed and sorted(side) == changed, (written, sorted(side))
    r = rows["7kkk_A"]
    assert (r["n_matched_merged"], r["n_unmatched"], r["raw_replaced"]) == ("1", "0", "False"), r
    k = side["7kkk_A"]
    assert k["match"][0] == 3 and k["index"][0] == 0 and bool(k["chimera"][0]), k
    r = rows["7lll_A"]
    assert (r["raw_replaced"], r["n_matched_id"], r["n_unmatched"]) == ("True", "1", "0"), r
    assert r["raw_used"] == str(data_dir.resolve() / "raw" / "7lll.cif") and side["7lll_A"]["match"][2] == 4, r
    assert all(rows[s]["raw_replaced"] == "False" for s in CATS if s not in ("7lll_A", "7ggg_A"))
    r = rows["7jjj_A"]
    assert (r["n_unmatched"], r["n_matched_exact"], r["fields"].split(",")[0]) == ("1", "10", "coords"), r
    assert sorted(ls) == sorted(s for s in CATS if s != "7ggg_A") and ls["7aaa_A"].tolist() == [1, 2, 3] + list(
        range(101, 109)), sorted(ls)
    assert side["7bbb_A"]["match"].tolist() == [1, 2] + [1] * 9, side["7bbb_A"]["match"]
    print(f"PASS all-stems --write-changed-only: {len(rows)} stems over 2 parts, status {st}; backbone_changed "
          f"{bc}; written + sidecars = changed only {changed}")


def stems_mode(root, data_dir, pack, audit):
    sf = root / "stems.txt"
    sf.write_text("7eee_A\n7fff_A\n")
    rows, _, _, written = run_main(["--pack", str(pack), "--data-dir", str(data_dir), "--audit", str(audit),
                                 "--stems", str(sf)], root / "out_stems")
    assert {s: r["status"] for s, r in rows.items()} == {"7eee_A": "written", "7fff_A": "dropped_collide"}, rows
    assert written == ["7eee_A"] and rows["7eee_A"]["fields"] != ""
    print("PASS stems mode: 7eee_A written, 7fff_A dropped_collide")


def unit_old_to_new(data_dir, olds):
    path = str(data_dir / "raw" / "7bbb.cif")
    df = read_pdb_to_dataframe(path=path, mmcif_seqres_fields=True)
    _, new = reprocess_chain("7bbb_A", data_dir / "raw", data_dir.parent / "stage_new" / "processed", {}, True, Data())
    old = olds["7bbb_A"]
    assert A.xs(old, 1)[1] == 2.6 and A.xs(new, 1)[1] == 2.1, (A.xs(old, 1), A.xs(new, 1))
    assert old_to_new(old, new, {})["index"].tolist() == [0, -1] + list(range(2, 11))
    m = old_to_new(old, new, altloc_rows(df, "A"))
    assert m["index"].tolist() == list(range(11)) and m["match"].tolist() == [1, 2] + [1] * 9 and m["chimera"][1]
    _, n7 = reprocess_chain("7eee_A", data_dir / "raw", data_dir.parent / "stage_new" / "processed", MAN, True, Data())
    o = olds["7eee_A"].clone()
    o.coords = o.coords.clone()
    o.coords[0, 1] = torch.tensor([1.0, 2.0, 3.0])
    nine = torch.tensor([[9.0, 9.0, 9.0]])
    alt = {n7.residue_id[0]: {"CA": torch.tensor([[1.0, 9.0, 9.0]])}, n7.residue_id[1]: {"CA": torch.tensor([[9.0, 2.0, 3.0]])},
           n7.residue_id[2]: {"CA": nine}}
    m7 = old_to_new(o, n7, alt)
    assert (int(m7["match"][0]), int(m7["index"][0]), bool(m7["chimera"][0])) == (3, 1, True), m7
    assert int(old_to_new(o, n7, {n7.residue_id[0]: {"CA": nine}})["match"][0]) == 0
    assert int(old_to_new(o, n7, {n7.residue_id[0]: {"CA": nine}}, by_id=True)["match"][0]) == 4
    o.residue_id = ["A:SER:99:"] + list(o.residue_id[1:])  # no identical residue_id -> still unmatched (fail closed)
    assert int(old_to_new(o, n7, {n7.residue_id[0]: {"CA": nine}}, by_id=True)["match"][0]) == 0
    rid = new.residue_id[1]
    for xyz, want in (((1.0, 5.0, 3.0), 2), ((1.0, 5.0, 7.0), 0), ((4.0, 5.0, 6.0), 2)):
        o = old.clone()
        o.coords = o.coords.clone()
        o.coords[1, 1] = torch.tensor(xyz)
        alt = {rid: {"CA": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}}
        assert old_to_new(o, new, alt)["match"].tolist()[1] == want, (xyz, want)
    print("PASS old_to_new: 7bbb CA B (old) unmatched without altloc rows [0, -1, 2..], match [1, 2, 1..] with them; "
          "per-component chimera (1,5,3) -> 2, (1,5,7) -> 0, whole row (4,5,6) -> 2; merged union -> match 3 on the "
          "candidate supplying 2 of 3 components; by_id -> match 4")


def unit_diff_fields():
    base = dict(residue_id=["A:ALA:1:"], coords=torch.tensor([[float("nan"), 1.0]]))
    assert diff_fields(Data(**base, res=float("nan")), Data(**base, res=float("nan")))[0] == []
    assert diff_fields(Data(**base, res=float("nan")), Data(**base, res=2.0))[0] == ["res"]
    assert diff_fields(Data(**base, res=1), Data(**base, res=1.0))[0] == ["res"]
    print("PASS diff_fields: NaN == NaN for scalars and tensors; NaN vs 2.0 and int vs float differ")


def raw_audit(root, data_dir, pack):
    before = sorted((str(q), q.stat().st_mtime_ns, q.stat().st_size) for q in data_dir.rglob("*"))
    out = root / "raw_audit.tsv"
    p = subprocess.run([sys.executable, "scratchpad/raw_version_audit.py", "--pack", str(pack), "--data-dir",
                        str(data_dir), "--out", str(out), "--workers", "2"], cwd=REPO, env=os.environ.copy(),
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    lines = out.read_text().splitlines()
    rows = {r[0]: dict(zip(lines[0].split("\t"), r)) for r in (x.split("\t") for x in lines[1:])}
    pdbs = sorted({s.split("_")[0] for s in CATS})
    assert sorted(rows) == pdbs, sorted(rows)
    r = rows["7lll"]
    assert (r["flat_exists"], r["bucket_exists"], r["identical"]) == ("True", "True", "False"), r
    assert (r["flat_last_revision"], r["bucket_last_revision"]) == ("2021-03-04", "2018-05-06"), r  # loop / single
    assert (rows["7ggg"]["flat_exists"], rows["7ggg"]["bucket_exists"]) == ("False", "False")
    assert all((rows[q]["flat_exists"], rows[q]["bucket_exists"], rows[q]["identical"]) == ("True", "False", "")
               for q in pdbs if q not in ("7lll", "7ggg"))
    assert sorted((str(q), q.stat().st_mtime_ns, q.stat().st_size) for q in data_dir.rglob("*")) == before
    gz = root / "hist.cif.gz"
    with gzip.open(gz, "wt") as f:
        f.write("data_X\n#\nloop_\n_atom_site.id\n1\n#\n" + HIST_LOOP + "loop_\n_other.a\n1\n")
    assert R.last_revision(gz) == "2021-03-04" and R.last_revision(data_dir / "raw" / "7aaa.cif") == "absent"
    H = "_pdbx_audit_revision_history."
    loop = lambda cols, body: "loop_\n" + "".join(f"{H}{c}\n" for c in cols) + body
    edge = {  # CIF edge cases of the streamed parser (reviewer probe + text-field value)
        "cols_any_order": (loop(["revision_date", "ordinal"], "2020-01-01 1\n2022-02-02 2\n#\n"), "2022-02-02"),
        "quoted_dates": (loop(["ordinal", "revision_date"], "1 '2020-01-01'\n2 \"2023-03-03\"\n#\n"), "2023-03-03"),
        "next_category": (loop(["ordinal", "revision_date"], "1 2020-01-01\n2 2021-01-01\n_next.a 1\n"), "2021-01-01"),
        "eof": (loop(["ordinal", "revision_date"], "1 2020-01-01\n2 2021-01-01"), "2021-01-01"),
        "text_field_elsewhere": (f"_struct.title\n;\nloop_\n{H}revision_date 1999-01-01\n;\n"
                                 + loop(["ordinal", "revision_date"], "1 2020-01-01\n#\n"), "2020-01-01"),
        "null_in_loop": (loop(["ordinal", "revision_date"], "1 2020-01-01\n2 ?\n3 .\n#\n"), "2020-01-01"),
        "null_single": (f"{H}ordinal 1\n{H}revision_date ?\n#\n", "absent"),
        "tab_single": (f"{H}ordinal\t1\n{H}revision_date\t2020-01-01\n#\n", "2020-01-01"),
        "value_next_line": (f"{H}ordinal 1\n{H}revision_date\n2020-01-01\n#\n", "2020-01-01"),
        "wrapped_row": (loop(["ordinal", "data_content_type", "revision_date"], "1 'Structure model'\n2020-01-01\n"
                                                                              "2 'Structure model' 2021-06-07\n#\n"),
                        "2021-06-07"),
        "apostrophe_value": (loop(["ordinal", "data_content_type", "revision_date"], "1 'O'Brien model' 2020-01-01\n#\n"),
                             "2020-01-01"),
        "text_field_value": (loop(["ordinal", "data_content_type", "revision_date"],
                                  "1\n;\nmulti _x loop_\nline\n;\n2020-05-05 # comment\n#\n"), "2020-05-05"),
    }
    for k, (body, want) in edge.items():
        q = root / f"edge_{k}.cif"
        q.write_text("data_X\n#\n" + body)
        assert R.last_revision(q) == want, (k, R.last_revision(q), want)
    print(f"PASS raw_version_audit: {len(rows)} of {len(pdbs)} ids; 7lll flat != bucket, last revision loop 2021-03-04 / "
          f"single 2018-05-06 (+ gz between other loops, absent, {len(edge)} CIF edge cases); 7ggg none; rest flat only; "
          f"data tree unchanged ({len(before)} paths)")


def main():
    unit_diff_fields()
    root = pathlib.Path(tempfile.mkdtemp())
    data_dir, pack, audit, olds = build(root)
    unit_old_to_new(data_dir, olds)
    all_stems(root, data_dir, pack, audit)
    stems_mode(root, data_dir, pack, audit)
    raw_audit(root, data_dir, pack)
    print("ALL PASS")


if __name__ == "__main__":
    main()
