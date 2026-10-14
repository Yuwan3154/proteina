"""Audit the T8 build parts before merge: every part present and stamped with the same gate, skip reasons totalled over
all parts (no_f2c_maps / usalign_failed / load failures must be 0), chains and template rows per part, and chains of the
chain list that are in no part. Usage (PYTHONPATH=repo): python scratchpad/t8_audit_parts.py --parts-dir D --n-parts N
--chain-list TXT"""

import argparse
import os
from collections import Counter

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--n-parts", type=int, required=True)
    ap.add_argument("--chain-list", required=True)
    args = ap.parse_args()
    reasons, gates, n_chain, n_rows, n_nat, seen = Counter(), Counter(), 0, 0, 0, set()
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
        for l in open(os.path.join(args.parts_dir, f"skips_{p:04d}.tsv")):
            reasons[l.rstrip("\n").split("\t")[2].split(":")[0]] += 1
    ids = [l.strip() for l in open(args.chain_list) if l.strip()]
    print(f"parts {args.n_parts}; gate stamps {dict(gates)}")
    print(f"chains in parts {n_chain} (with native row {n_nat}); chain list {len(ids)}; missing from parts "
          f"{len(set(ids) - seen)}; template rows {n_rows} ({n_rows / max(n_nat, 1):.1f} per chain with a native)")
    print("skips by reason:", dict(reasons.most_common()))
    rs = sorted(r for *_, r in per_part)
    print(f"rows per part min {rs[0]} median {rs[len(rs) // 2]} max {rs[-1]}")


if __name__ == "__main__":
    main()
