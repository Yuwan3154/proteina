"""D36 W4 (Q6) staging (CPU): every selected template -> <root>/templates/<pid>/<template_id>.pdb, single chain A.
PDB sources (proteina cg2all, colabfold AF2, c2c) are copied byte-for-byte; mmCIF sources (AF3 model, the native) are
written as their protein chain (model 1, residues whose name maps to the 20 amino acids, no H/D), renamed A.
Gates per template: sequence == the af2rank query (scorer's own reference extraction from the native cif) and CA count
== query length (so skip_template_alignment stays 1:1). Fixed label = TM(template, native), USalign -TMscore 5
normalised by the native, via step 5's own run_reference_vs_paths_dir2. Writes <root>/templates_manifest.csv.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys

from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import (
    _detect_chain_in_cif, get_ca_coords_from_structure, get_sequence_from_structure)
from proteinfoundation.prediction_pipeline.proteina_analysis import run_reference_vs_paths_dir2

AA3 = {"ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
       "THR", "VAL", "TRP", "TYR"}


class ChainProtein(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.get_resname().strip() in AA3

    def accept_atom(self, atom):
        return atom.element not in ("H", "D")


def cif_chain_to_pdb(cif, chain_hint, out):
    s = MMCIFParser(QUIET=True).get_structure("x", cif)
    cid = _detect_chain_in_cif(cif, chain_hint)
    model = next(iter(s))
    for m in list(s)[1:]:
        s.detach_child(m.id)
    for c in list(model):
        if c.id != cid:
            model.detach_child(c.id)
    ch = model[cid]
    if cid != "A":
        ch.id = "A"
    io = PDBIO()
    io.set_structure(s)
    io.save(out, ChainProtein("A"))
    return cid


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--natives", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    args = ap.parse_args()
    usalign = shutil.which("USalign")
    assert usalign, "USalign not on PATH"
    sel = list(csv.DictReader(open(args.selected)))
    by_pid = {}
    for r in sel:
        by_pid.setdefault(r["pid"], []).append(r)
    out_rows = []
    for p, rows in by_pid.items():
        assert len({r["template_id"] for r in rows}) == len(rows), f"{p}: duplicate template_id"
        entry, chain = p.rsplit("_", 1)
        native = os.path.join(args.natives, f"{entry}.cif")
        ref_chain = _detect_chain_in_cif(native, chain)
        ref_seq = get_sequence_from_structure(native, ref_chain)
        tdir = os.path.join(args.root, "templates", p)
        os.makedirs(tdir, exist_ok=True)
        staged = []
        for r in rows:
            src = r["path"] if r["host"] == "Engaging" else os.path.join(args.root, "raw_sc", r["path"].lstrip("/"))
            assert os.path.exists(src), src
            out = os.path.join(tdir, f"{r['template_id']}.pdb")
            if src.endswith(".cif"):
                used_chain = cif_chain_to_pdb(src, r["chain"], out)
            else:
                chains = [c.id for c in next(iter(PDBParser(QUIET=True).get_structure("x", src)))]
                assert chains == ["A"], f"{src}: chains {chains}"
                shutil.copyfile(src, out)
                used_chain = "A"
            seq = get_sequence_from_structure(out, "A")
            n_ca = get_ca_coords_from_structure(out, "A").shape[0]
            assert seq == ref_seq, f"{p} {r['template_id']}: sequence differs from the query ({len(seq)} vs {len(ref_seq)})"
            assert n_ca == len(ref_seq), f"{p} {r['template_id']}: {n_ca} CA vs query {len(ref_seq)}"
            staged.append((r, out, src, used_chain))
        tm = run_reference_vs_paths_dir2(native, [x[1] for x in staged], chain1=chain, usalign_path=usalign)
        for r, out, src, used_chain in staged:
            name = os.path.basename(out)
            assert name in tm, f"{p} {name}: no USalign row"
            out_rows.append({"pid": p, "template_id": r["template_id"], "source": r["source"], "path": out,
                             "src_path": src, "src_host": r["host"], "src_chain": used_chain, "md5": md5(out),
                             "src_md5": md5(src), "L": len(ref_seq), "tm_native": round(tm[name]["tms"], 5),
                             "stored_tm": r["stored_tm"], "bin": r["bin"]})
        nat = [x for x in out_rows if x["pid"] == p and x["template_id"] == "native"]
        assert len(nat) == 1 and nat[0]["tm_native"] > 0.999, (p, nat)
        print(f"[stage] {p}: {len(staged)} templates, L {len(ref_seq)}", flush=True)
    fields = list(out_rows[0])
    with open(os.path.join(args.root, "templates_manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    assert len(out_rows) == len(sel), (len(out_rows), len(sel))
    d = sorted(abs(float(x["tm_native"]) - float(x["stored_tm"])) for x in out_rows)
    print(f"[stage] {len(out_rows)} templates staged; |label - stored TM|: median {d[len(d) // 2]:.5f} max {d[-1]:.5f}")
    print("D36W4_STAGE_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
