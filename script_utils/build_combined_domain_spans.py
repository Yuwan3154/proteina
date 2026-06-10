"""Build a combined PDB+AFDB domain-spans pickle for DomainCropTransform.

Output schema: ``{graph_id: {topology(CAT3): [(start, end), ...]}}`` where start/end
are inclusive residue numbers matched against ``graph.residue_pdb_idx`` (PDB author
numbering; AFDB 1..L sequential). One representative (largest) domain is stored per
(chain, topology); discontinuous domains keep all their segments.

PDB spans: read from CATHLabelTransform's already-parsed cathid_to_segment_mapping
(cath-b segments, keyed to graph.id = ``pdb_Chain`` via SIFTS pdb_chain_cath_uniprot).
AFDB spans: awk-scan the 128 GB TED domain-summary TSV (col1=domain id, col4=chopping,
col14=CATH), filtered to the labeled AFDB stems' uniprots, mapped to v6 stems.

Run on Engaging in cue_openfold with PYTHONPATH set to the worktree.
"""

import argparse
import os
import pathlib
import pickle
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_INT_RE = re.compile(r"-?\d+")


def parse_int(s):
    m = _INT_RE.search(str(s))
    return int(m.group()) if m else None


def cat3(code):
    parts = code.split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else None


def parse_chopping(chopping):
    """Parse a TED chopping string like ``11-41_290-389`` -> [(11,41),(290,389)]."""
    segs = []
    for part in chopping.split("_"):
        if "-" not in part:
            continue
        a, b = part.rsplit("-", 1) if part.count("-") > 1 else part.split("-", 1)
        si, ei = parse_int(a), parse_int(b)
        if si is None or ei is None:
            continue
        if ei < si:
            si, ei = ei, si
        segs.append((si, ei))
    return segs


def all_domains_per_topo(per_topo):
    """per_topo: {topo: [(segments, length), ...]} -> {topo: [segments, ...]}.

    Keeps ALL distinct same-topology domains (deduped by segment-set), largest first.
    """
    out = {}
    for topo, domains in per_topo.items():
        seen = set()
        doms = []
        for segs, _length in sorted(domains, key=lambda d: (-d[1], d[0])):
            key = tuple(segs)
            if key in seen:
                continue
            seen.add(key)
            doms.append(segs)
        out[topo] = doms
    return out


def build_pdb_spans(cathdata_dir):
    from proteinfoundation.datasets.transforms import CATHLabelTransform
    from proteinfoundation.utils.ff_utils.pdb_utils import extract_cath_code_by_level

    clt = CATHLabelTransform(root_dir=cathdata_dir)
    out = {}
    for stem, cath_ids in clt.pdbchain_to_cathid_mapping.items():
        if "_" not in stem:
            continue
        stem_chain = stem.split("_", 1)[1]
        per_topo = defaultdict(list)
        for cid in cath_ids:
            code = clt.cathid_to_cathcode_mapping.get(cid)
            if not code:
                continue
            topo = extract_cath_code_by_level(code, "T")
            segs = []
            for seg_chain, s, e in clt.cathid_to_segment_mapping.get(cid, []):
                if seg_chain != stem_chain:
                    continue
                si, ei = parse_int(s), parse_int(e)
                if si is None or ei is None:
                    continue
                if ei < si:
                    si, ei = ei, si
                segs.append((si, ei))
            if segs:
                length = sum(e - s + 1 for s, e in segs)
                per_topo[topo].append((segs, length))
        if per_topo:
            out[stem] = all_domains_per_topo(per_topo)
    return out


def build_afdb_spans(afdb_chain_to_cat_path, ted_tsv, workdir):
    with open(afdb_chain_to_cat_path, "rb") as f:
        a2c = pickle.load(f)
    uni2stem = {}
    for stem in a2c:
        if not str(stem).startswith("AF-") or "-F1-" not in str(stem):
            continue
        uni = str(stem).split("-")[1]
        uni2stem.setdefault(uni, stem)

    os.makedirs(workdir, exist_ok=True)
    whitelist = os.path.join(workdir, "afdb_span_uni.txt")
    with open(whitelist, "w") as f:
        f.write("\n".join(uni2stem.keys()))
    out_tsv = os.path.join(workdir, "afdb_span_rows.tsv")
    prog = (
        'BEGIN{while((getline u < "%s")>0) T[u]=1} '
        '$14!="-"{split($1,a,"-"); if(a[2] in T) print a[2]"\\t"$4"\\t"$14}'
    ) % whitelist
    subprocess.run(
        ["bash", "-c", f"LC_ALL=C awk -F'\\t' '{prog}' {ted_tsv} > {out_tsv}"], check=True
    )

    spans = defaultdict(lambda: defaultdict(list))
    with open(out_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            uni, chopping, cath_field = parts[0], parts[1], parts[2]
            stem = uni2stem.get(uni)
            if not stem:
                continue
            segs = parse_chopping(chopping)
            if not segs:
                continue
            length = sum(e - s + 1 for s, e in segs)
            for cath in cath_field.split(","):
                topo = cat3(cath)
                if topo:
                    spans[stem][topo].append((segs, length))
    return {stem: all_domains_per_topo(tm) for stem, tm in spans.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cathdata-dir", required=True)
    ap.add_argument("--afdb-chain-to-cat", required=True)
    ap.add_argument("--ted-tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb-out", default=None)
    ap.add_argument("--afdb-out", default=None)
    ap.add_argument("--workdir", default="/tmp/domain_spans")
    args = ap.parse_args()

    pdb_spans = build_pdb_spans(args.cathdata_dir)
    print(f"PDB spans: {len(pdb_spans)} chains", flush=True)
    if args.pdb_out:
        with open(args.pdb_out, "wb") as f:
            pickle.dump(pdb_spans, f, protocol=pickle.HIGHEST_PROTOCOL)

    afdb_spans = build_afdb_spans(args.afdb_chain_to_cat, args.ted_tsv, args.workdir)
    print(f"AFDB spans: {len(afdb_spans)} chains", flush=True)
    if args.afdb_out:
        with open(args.afdb_out, "wb") as f:
            pickle.dump(afdb_spans, f, protocol=pickle.HIGHEST_PROTOCOL)

    overlap = set(pdb_spans) & set(afdb_spans)
    print(f"key overlap PDB/AFDB: {len(overlap)} (expect 0)", flush=True)
    combined = {**pdb_spans, **afdb_spans}
    print(f"combined spans: {len(combined)} chains", flush=True)
    multi = sum(1 for tm in combined.values() for doms in tm.values() if len(doms) > 1)
    over = sum(1 for tm in combined.values() for doms in tm.values()
               for segs in doms if sum(e - s + 1 for s, e in segs) > 320)
    print(f"(stem,topo) with >1 domain: {multi}; domains >320 residues: {over}", flush=True)
    with open(args.out, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
