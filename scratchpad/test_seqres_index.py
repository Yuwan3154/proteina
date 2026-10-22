"""precompute_seqres_index maps residue_id -> label_seq_id exactly as processing builds residue_id (T8b).

Synthetic mmCIF 9zzz, run through the REAL read_pdb_to_dataframe + protein_to_pyg:
  chain B (auth A): ALA 10, GLY 11, HETATM MSE 12 (processing drops it -> gap), SER 13, THR 13A (biopandas 0.5.1 drops
    insertion codes -> residue_id B:THR:13:), unobserved label 6-7, LEU 16 -> label_seq_id [1, 2, 4, 5, 8], status ok;
  chain C (auth C): ALA 20, LYS 21, LYS 21A (same resname + insertion -> ONE graph residue covering label 3 and 4) -> merged;
  a HOH HETATM with label_seq_id '.' (object-dtype column, must not crash).
Then cmd_part + cmd_merge on a 2-chain pack written like build_pack.py (object-dtype stems in idx.npz).
Hand-made mappings: nonmonotone, collide, no_label_seq, unmapped.
Usage: python scratchpad/test_seqres_index.py   (CPU, temp dir only)
"""

import argparse
import io
import json
import os
import tempfile

import numpy as np
import torch

from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe
from proteinfoundation.utils.precompute_seqres_index import chain_positions, cmd_merge, cmd_part, residue_label_seq

HEADER = ["data_9ZZZ", "_entry.id 9ZZZ", "#", "loop_"] + [f"_atom_site.{c}" for c in (
    "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id label_seq_id "
    "pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv pdbx_formal_charge auth_seq_id auth_comp_id "
    "auth_asym_id auth_atom_id pdbx_PDB_model_num").split()]
# (record, atom, element, comp, label_asym, label_seq, auth_seq, ins, auth_asym)
ATOMS = [("ATOM", "CA", "C", "ALA", "B", "1", 10, "?", "A"), ("ATOM", "CA", "C", "GLY", "B", "2", 11, "?", "A"),
         ("HETATM", "CA", "C", "MSE", "B", "3", 12, "?", "A"), ("ATOM", "CA", "C", "SER", "B", "4", 13, "?", "A"),
         ("ATOM", "CA", "C", "THR", "B", "5", 13, "A", "A"), ("ATOM", "CA", "C", "LEU", "B", "8", 16, "?", "A"),
         ("ATOM", "CA", "C", "ALA", "C", "1", 20, "?", "C"), ("ATOM", "CA", "C", "LYS", "C", "3", 21, "?", "C"),
         ("ATOM", "CA", "C", "LYS", "C", "4", 21, "A", "C"), ("HETATM", "O", "O", "HOH", "D", ".", 101, "?", "A")]


def write_cif(path):
    lines = list(HEADER)
    for k, (rec, atom, el, comp, lasym, lseq, aseq, ins, aasym) in enumerate(ATOMS):
        lines.append(f"{rec} {k + 1} {el} {atom} . {comp} {lasym} 1 {lseq} {ins} {3.8 * k:.3f} 0.000 0.000 1.00 0.00 ? "
                     f"{aseq} {comp} {aasym} {atom} 1")
    lines.append("#")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "pdb_train", "raw"))
    path = os.path.join(d, "pdb_train", "raw", "9zzz.cif")
    write_cif(path)
    mapping, ins = residue_label_seq(path)
    assert ins["B:THR:13:"] == {"A"} and ins["B:SER:13:"] == {""} and ins["C:LYS:21:"] == {"", "A"}, dict(ins)

    gb = protein_to_pyg(df=read_pdb_to_dataframe(path=path), path=path, chain_selection=["B"], keep_insertions=True)
    rids_b = list(gb.residue_id)
    assert rids_b == ["B:ALA:10:", "B:GLY:11:", "B:SER:13:", "B:THR:13:", "B:LEU:16:"], rids_b
    status, pos, n_merged = chain_positions(rids_b, mapping)
    assert status == "ok" and pos.tolist() == [1, 2, 4, 5, 8] and n_merged == 0, (status, pos, n_merged)
    print(f"PASS chain B: {rids_b} -> label_seq_id {pos.tolist()} (MSE + unobserved 6-7 are gaps)")

    gc = protein_to_pyg(df=read_pdb_to_dataframe(path=path), path=path, chain_selection=["C"], keep_insertions=True)
    rids_c = list(gc.residue_id)
    assert rids_c == ["C:ALA:20:", "C:LYS:21:"], rids_c
    assert chain_positions(rids_c, mapping) == ("merged", None, 1)
    print(f"PASS chain C: LYS 21 + 21A collapse into {rids_c[1]} -> merged (label_seq {sorted(mapping[rids_c[1]])})")

    status, pos, _ = chain_positions(["X:ALA:1:", "X:GLY:2:", "X:SER:3:"], {"X:ALA:1:": {3}, "X:GLY:2:": {1}, "X:SER:3:": {2}})
    assert status == "nonmonotone" and pos.tolist() == [3, 1, 2], (status, pos)
    assert chain_positions(["X:VAL:2:", "X:ILE:2:"], {"X:VAL:2:": {2}, "X:ILE:2:": {2}}) == ("collide", None, 0)
    assert chain_positions(["X:ALA:1:"], {"X:ALA:1:": {None}}) == ("no_label_seq", None, 0)
    assert chain_positions(["X:ALA:1:", "X:CYS:9:"], {"X:ALA:1:": {1}}) == ("unmapped", None, 0)
    print("PASS statuses: nonmonotone (positions kept), collide, no_label_seq, unmapped")

    pack = os.path.join(d, "t.pack")
    blobs = []
    for stem, g in (("9zzz_B", gb), ("9zzz_C", gc)):
        buf = io.BytesIO()
        torch.save({"residue_id": list(g.residue_id), "coords": g.coords}, buf)
        blobs.append((stem, buf.getvalue()))
    with open(pack, "wb") as fh:
        for _, b in blobs:
            fh.write(b)
    lengths = [len(b) for _, b in blobs]
    np.savez(pack + ".idx.npz", stems=np.array([s for s, _ in blobs], dtype=object),
             offsets=np.cumsum([0] + lengths[:-1]).astype(np.int64), lengths=np.array(lengths, dtype=np.int64))
    json.dump({}, open(os.path.join(d, "pdb_train", "shard_manifest.json"), "w"))
    stems = os.path.join(d, "stems.txt")
    open(stems, "w").write("9zzz_B\n9zzz_C\n")
    parts = os.path.join(d, "parts")
    os.makedirs(parts)
    cmd_part(argparse.Namespace(pack=pack, data_dir=os.path.join(d, "pdb_train"), stems=stems, part=0, n_parts=1,
                                out_dir=parts))
    out = os.path.join(d, "seqres.pt")
    cmd_merge(argparse.Namespace(stems=stems, parts_dir=parts, out=out))
    t = torch.load(out, weights_only=False)
    assert t["stems"] == ["9zzz_B", "9zzz_C"] and t["status"] == ["ok", "merged"], (t["stems"], t["status"])
    assert t["offset"].tolist() == [0, 5, 5], t["offset"]
    assert t["label_seq_flat"].tolist() == [1, 2, 4, 5, 8], t["label_seq_flat"]
    assert t["audit"]["merged_residue_ids"] == 1 and t["audit"]["gapped_ok"] == 1, t["audit"]
    assert t["audit"]["ok_with_insertion_codes"] == 1, t["audit"]  # chain B: THR 13A (distinct resname, never merged)
    print(f"PASS cmd_part + cmd_merge on a 2-chain pack: audit {t['audit']}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
