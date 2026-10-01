# T7 (3): best-by-validation-sampling vs latest tri ckpt on the SAME 99 queries x 8 (same templates, seed 0).
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

Q = "/Users/Chenxi/.claude/jobs/2c2943b0/tmp/t7q_out"
L = "/Users/Chenxi/.claude/jobs/2c2943b0/tmp/t7ql_out"
ARMS = {"CB-8": (f"{Q}/final_cb8_tri.jsonl", f"{L}/latest_cb8_tri.jsonl"),
        "ConFind": (f"{Q}/final18120_cf_tri.jsonl", f"{L}/latest_cf_tri.jsonl")}
M = [("P@L", "tri_contact_precision_at_L"), ("P@L/5", "tri_contact_precision_at_L5"),
     ("LR P@L/5", "tri_contact_long_range_precision_at_L5"), ("F1@0.5", "tri_contact_f1"), ("TM", "tm")]
for name, (fb, fl) in ARMS.items():
    B = {(r["stem"], r["sample_index"]): r for r in map(json.loads, open(fb))}
    Lt = {(r["stem"], r["sample_index"]): r for r in map(json.loads, open(fl))}
    assert B.keys() == Lt.keys() and len(B) == 792
    assert all(B[k]["ref_id"] == Lt[k]["ref_id"] for k in B), "templates differ"
    print(f"== {name}: best ckpt step {B[next(iter(B))]['tri_file'] and ''}vs latest (792 samples, 99 queries; same templates)")
    print(f"   {'metric':9s} {'best med':>9s} {'latest med':>11s} {'best mean':>10s} {'latest mean':>12s}  per-query mean: latest better/worse, Wilcoxon p")
    for lab, k in M:
        b = np.array([B[x][k] for x in B]); l = np.array([Lt[x][k] for x in B])
        qb, ql = defaultdict(list), defaultdict(list)
        for x in B:
            qb[x[0]].append(B[x][k]); ql[x[0]].append(Lt[x][k])
        d = np.array([np.mean(ql[s]) - np.mean(qb[s]) for s in qb])
        print(f"   {lab:9s} {np.median(b):9.3f} {np.median(l):11.3f} {b.mean():10.3f} {l.mean():12.3f}  "
              f"{(d > 0).sum()}/{(d < 0).sum()}, p={wilcoxon(d).pvalue:.1e}")
    tb = np.array([B[x]["tm"] for x in B]); tl = np.array([Lt[x]["tm"] for x in B])
    print(f"   TM>=0.5: best {(tb >= 0.5).sum()} vs latest {(tl >= 0.5).sum()} of 792; mirror rate {np.mean([B[x]['is_mirrored'] for x in B]):.3f} vs {np.mean([Lt[x]['is_mirrored'] for x in B]):.3f}")
