"""Audit the T8 build parts before merge: every part present and stamped with the same gate, skip reasons totalled over
all parts (no_f2c_maps / usalign_failed / load failures must be 0), chains and template rows per part, and chains of the
chain list that are in no part. Usage (PYTHONPATH=repo): python scratchpad/t8_audit_parts.py --parts-dir D --n-parts N
--chain-list TXT
T8b (--final-selection parts): --override-chains TXT --template-pack-tag A --override-pack-tag B also checks the pack
precedence on every template row (override-listed chains carry only B keys, the rest only A keys) and the template-source
stamp; --strict exits 1 on any failure class (missing parts' chains, no_f2c_maps*, usalign_failed, load_failed,
precedence violations, more than one gate / source stamp)."""

import argparse
import os
import sys
from collections import Counter

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--n-parts", type=int, required=True)
    ap.add_argument("--chain-list", required=True)
    ap.add_argument("--override-chains", default="")
    ap.add_argument("--template-pack-tag", default="")
    ap.add_argument("--override-pack-tag", default="")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    reasons, gates, n_chain, n_rows, n_nat, seen = Counter(), Counter(), 0, 0, 0, set()
    override = {l.strip() for l in open(args.override_chains) if l.strip()} if args.override_chains else None
    sources, prec = Counter(), Counter()
    per_part = []
    for p in range(args.n_parts):
        d = torch.load(os.path.join(args.parts_dir, f"part_{p:04d}.pt"), map_location="cpu", weights_only=False)
        gates[d.get("template_gate", "MISSING")] += 1
        rows = sum(len(c["rows"]) for c in d["chains"])
        nat = sum(c["native"] is not None for c in d["chains"])
        per_part.append((p, len(d["chains"]), nat, rows))
        n_chain += len(d["chains"])
        n_nat += nat
        n_rows += rows
        seen.update(c["stem"] for c in d["chains"])
        sources[d.get("template_source", "MISSING")] += 1
        if override is not None:
            for c in d["chains"]:
                want = args.override_pack_tag if c["stem"] in override else args.template_pack_tag
                for row in c["rows"]:
                    prec["ok" if str(row[3]).split("/")[0] == want else "violation"] += 1
                if c["stem"] in override:
                    prec["override_chains_seen"] += 1
                    prec["override_chains_with_rows"] += bool(c["rows"])
        for l in open(os.path.join(args.parts_dir, f"skips_{p:04d}.tsv")):
            reasons[l.rstrip("\n").split("\t")[2].split(":")[0]] += 1
    ids = [l.strip() for l in open(args.chain_list) if l.strip()]
    print(f"parts {args.n_parts}; gate stamps {dict(gates)}")
    print(f"chains in parts {n_chain} (with native row {n_nat}); chain list {len(ids)}; missing from parts "
          f"{len(set(ids) - seen)}; template rows {n_rows} ({n_rows / max(n_nat, 1):.1f} per chain with a native)")
    print("skips by reason:", dict(reasons.most_common()))
    rs = sorted(r for *_, r in per_part)
    print(f"rows per part min {rs[0]} median {rs[len(rs) // 2]} max {rs[-1]}")
    if override is not None:
        print(f"template-source stamps: {len(sources)} distinct, parts per stamp {sorted(sources.values())}")
        print(f"pack precedence over {prec['ok'] + prec['violation']} template rows: {dict(prec)}")
    if args.strict:
        bad = {"missing_from_parts": len(set(ids) - seen), "gate_stamps": len(gates) != 1,
               "no_f2c_maps": reasons["no_f2c_maps"] + reasons["no_f2c_map_for_draw"],
               "usalign_failed": reasons["usalign_failed"], "load_failed": reasons["load_failed"],
               "no_template_rows": n_rows == 0}
        if override is not None:
            bad.update(precedence_violations=prec["violation"], source_stamps=len(sources) != 1 or "MISSING" in sources)
        bad = {k: v for k, v in bad.items() if v}
        print("STRICT AUDIT " + ("PASS" if not bad else f"FAIL {bad}"))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
