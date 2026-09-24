"""Compare a ConFind-built and a CB-8-built synthetic-index part over the same chains.

Must be IDENTICAL: chain set, native presence, template rows (tm, rewind, rung), DSSP runs, element count T,
element-of-residue maps and USalign alignments -- none of these depend on the contact definition.
Must DIFFER somewhere: the element contact bytes / structural bytes (else the ConFind source was not used).

Usage: python scratchpad/compare_index_parts.py CONFIND_PART.pt CB8_PART.pt
"""

import sys

import numpy as np
import torch

a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
print(f"[defs] confind part: {a['contact_def']!r}\n       cb8 part:     {b['contact_def']!r}")
ca = {c["stem"]: c for c in a["chains"]}
cb = {c["stem"]: c for c in b["chains"]}
bad = []
assert sorted(ca) == sorted(cb), "chain sets differ"
n_rows = n_ref_diff = n_struct_diff = n_native = 0
dens_cf, dens_cb = [], []
for s in ca:
    x, y = ca[s], cb[s]
    if (x["native"] is None) != (y["native"] is None):
        bad.append(f"{s}: native presence differs")
        continue
    pairs = []
    if x["native"] is not None:
        n_native += 1
        pairs.append(("native", x["native"], y["native"]))
    if len(x["rows"]) != len(y["rows"]):
        bad.append(f"{s}: {len(x['rows'])} vs {len(y['rows'])} template rows")
        continue
    for rx, ry in zip(x["rows"], y["rows"]):
        if rx[1:5] != ry[1:5]:  # tm, rewind, rung k, alignment bytes
            bad.append(f"{s}: template row meta/alignment differs (rung {rx[3]} vs {ry[3]})")
        pairs.append((f"tpl{rx[3]}", rx[0], ry[0]))
    for tag, u, v in pairs:
        if [list(r) for r in u[0]] != [list(r) for r in v[0]] or u[2] != v[2]:
            bad.append(f"{s} {tag}: DSSP runs or T differ")
            continue
        if tag == "native" and not np.array_equal(u[5], v[5]):
            bad.append(f"{s} native: element-of-residue map differs")
        n_rows += 1
        n_ref_diff += u[1] != v[1]
        n_struct_diff += u[3] != v[3]
        if u[2] > 0:
            dens_cf.append(np.frombuffer(u[1], dtype=np.uint8).mean())
            dens_cb.append(np.frombuffer(v[1], dtype=np.uint8).mean())

print(f"[compare] {len(ca)} chains, {n_native} natives, {n_rows} rows compared")
print(f"[compare] element-contact bytes differ in {n_ref_diff}/{n_rows} rows; structural bytes differ in "
      f"{n_struct_diff}/{n_rows}; mean element-contact density ConFind {np.mean(dens_cf):.3f} vs CB-8 {np.mean(dens_cb):.3f}")
for line in bad[:20]:
    print(f"[MISMATCH] {line}")
ok = not bad and n_rows > 0 and (n_ref_diff + n_struct_diff) > 0
print(f"[{'PASS' if ok else 'FAIL'}] invariants identical: {not bad}; contact source changed something: "
      f"{(n_ref_diff + n_struct_diff) > 0}; rows compared: {n_rows}")
sys.exit(0 if ok else 1)
