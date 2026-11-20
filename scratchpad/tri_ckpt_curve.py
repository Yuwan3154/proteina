# T7-NEWEST: tri extra-training effect, c2c held fixed per definition (CB-8: tbeta final; ConFind: twin final 29,000).
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

T = "/Users/Chenxi/.claude/jobs/2c2943b0/tmp"
ARMS = {"CB-8": [("78,772 best", f"{T}/t7q_out/final_cb8_tri.jsonl"), ("92,700", f"{T}/t7ql_out/latest_cb8_tri.jsonl"),
                 ("106,750", f"{T}/t7qn_out/all_t7qn_cb8.jsonl"),
                 ("141,030 newest", f"{T}/t7qx_out/all_t7qx_cb8.jsonl")],
        "ConFind": [("11,460 best", f"{T}/t7qn_out/all_t7q_cf29000.jsonl"), ("24,872", f"{T}/t7qn_out/all_t7ql_cf29000.jsonl"),
                    ("48,150", f"{T}/t7qn_out/all_t7qn_cf29000.jsonl"),
                    ("60,505 newest", f"{T}/t7qx_out/all_t7qx_cf29000.jsonl")]}
M = [("TM", "tm"), ("P@L", "tri_contact_precision_at_L"), ("LR P@L/5", "tri_contact_long_range_precision_at_L5"), ("F1@0.5", "tri_contact_f1")]
for name, arms in ARMS.items():
    R = []
    for lab, f in arms:
        rows = {(r["stem"], r["sample_index"]): r for r in map(json.loads, open(f))}
        assert len(rows) == 792, (f, len(rows))
        R.append((lab, rows))
    keys = R[0][1].keys()
    assert all(r.keys() == keys for _, r in R) and all(R[0][1][k]["ref_id"] == r[k]["ref_id"] for _, r in R for k in keys)
    steps = {r[next(iter(keys))]["global_step"] for _, r in R}
    print(f"== {name} tri ckpts (c2c step {steps}; same templates asserted; 99 queries x 8)")
    print("   ckpt            " + "  ".join(f"{m:>9s}" for m, _ in M) + "   TM>=0.5   vs first ckpt (per-query mean TM): better/worse, p")
    q0 = defaultdict(list)
    for k in keys: q0[k[0]].append(R[0][1][k]["tm"])
    for lab, rows in R:
        med = [np.median([rows[k][key] for k in keys]) for _, key in M]
        n5 = sum(rows[k]["tm"] >= 0.5 for k in keys)
        q = defaultdict(list)
        for k in keys: q[k[0]].append(rows[k]["tm"])
        d = np.array([np.mean(q[s]) - np.mean(q0[s]) for s in q0])
        cmp = "" if lab == R[0][0] else f"{(d > 0).sum()}/{(d < 0).sum()}, median d {np.median(d):+.3f}, p {wilcoxon(d).pvalue:.2g}"
        print(f"   {lab:15s} " + "  ".join(f"{v:9.3f}" for v in med) + f"   {n5:4d}      {cmp}")
