"""D36 W4 candidate eligibility (CPU, Biopython only; no openfold/proteinfoundation package import): for every
candidate template, its protein-chain sequence read from the structure file must equal the af2rank query (the native's
sequence). Mirrors af2rank_openfold_scorer.get_sequence_from_structure / _detect_chain_in_cif and reuses
cif_chain_mapping.py by file path. The stage job's sequence gate stays authoritative.
Also mirrors the stage's two structural gates: CA count (get_ca_coords_from_structure) == query length, and for PDB
sources the chain list == ["A"]. eligible = seq_match and ca_ok and chains_ok.
Writes one row per candidate: pid, template_id, source, n_res, n_ca, query_len, native_n_ca, seq_match, ca_ok, chains_ok,
eligible.
"""
import argparse
import csv
import importlib.util
import os
import sys
from multiprocessing import Pool

from Bio.PDB import MMCIFParser, PDBParser

_SPEC = importlib.util.spec_from_file_location(
    "cif_chain_mapping", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proteinfoundation",
                                      "prediction_pipeline", "cif_chain_mapping.py"))
ccm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ccm)
AA = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K',
      'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
      'TRP': 'W', 'TYR': 'Y'}


def parse(path):
    return (MMCIFParser if path.lower().endswith(".cif") else PDBParser)(QUIET=True).get_structure("x", path)


def detect_chain(path, hint):
    if not path.lower().endswith(".cif"):
        return hint
    avail = [c.id for c in next(iter(parse(path)))]
    r = ccm.resolve_biopython_chain_for_structure(path, hint)
    return r if r in avail else (avail[0] if avail else "A")


def chain_of(s, chain):
    target = None
    for model in s:
        for c in model:
            if c.id == chain:
                target = c
                break
        if target is not None:
            break
    if target is None:
        target = next(iter(next(iter(s))))
    return target


def sequence(s, chain):
    out = []
    for res in chain_of(s, chain):
        n = res.get_resname().strip()
        if n in AA:
            out.append(AA[n])
        elif res.id[0] == " ":
            out.append("X")
    return "".join(out)


def n_ca(s, chain):
    return sum(1 for res in chain_of(s, chain) if (res.get_resname().strip() in AA or res.id[0] == " ") and "CA" in res)


def check(job):
    r, (query, nat_ca) = job
    st = parse(r["path"])
    ch = detect_chain(r["path"], r["chain"])
    seq, nca = sequence(st, ch), n_ca(st, ch)
    chains_ok = r["path"].lower().endswith(".cif") or [c.id for c in next(iter(st))] == ["A"]
    m, ca_ok = seq == query, nca == len(query)
    return {"pid": r["pid"], "template_id": r["template_id"], "source": r["source"], "n_res": len(seq), "n_ca": nca,
            "query_len": len(query), "native_n_ca": nat_ca, "seq_match": int(m), "ca_ok": int(ca_ok),
            "chains_ok": int(chains_ok), "eligible": int(m and ca_ok and chains_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="csv with pid, template_id, source, host, path, chain")
    ap.add_argument("--host", required=True, help="only rows with this host")
    ap.add_argument("--natives", required=True, help="dir with <ENTRY>.cif")
    ap.add_argument("--path-prefix", nargs=2, default=None, metavar=("FROM", "TO"), help="rewrite a path prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.candidates)) if r["host"] == args.host]
    if args.path_prefix:
        a, b = args.path_prefix
        for r in rows:
            assert r["path"].startswith(a), r["path"]
            r["path"] = b + r["path"][len(a):]
    query = {}
    for p in sorted({r["pid"] for r in rows}):
        entry, chain = p.rsplit("_", 1)
        nat = os.path.join(args.natives, f"{entry}.cif")
        st, ch = parse(nat), detect_chain(nat, chain)
        query[p] = (sequence(st, ch), n_ca(st, ch))
    with Pool(args.workers) as pool:
        out = pool.map(check, [(r, query[r["pid"]]) for r in rows], chunksize=64)
    assert len(out) == len(rows)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    bad_nat = sorted(p for p, (q, c) in query.items() if c != len(q))
    print(f"[seqcheck] host {args.host}: {len(out)} candidates, {sum(o['seq_match'] for o in out)} sequence match, "
          f"{sum(o['eligible'] for o in out)} eligible, {len(query)} targets; natives with CA count != query length: "
          f"{bad_nat}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
