"""Pick example rows for the deck by a FIXED rule (user 2026-10-08), from the full 195-query set.

Categories (success = per-query mean TM >= 0.5): both succeed / ConFind succeeds & CB-8 fails / both fail. Within a
category: the 2 queries whose CB-8-vs-ConFind gap (ConFind - CB-8 per-query mean TM) is closest to that category's
median gap. Per query and definition: the sample whose TM is nearest that query's median TM (ties -> lower index).
Usage: python t7_pick_examples.py CB8_TRI.jsonl CF_TRI.jsonl OUT.json
"""

import json
import sys
from collections import defaultdict

import numpy as np

cb, cf, out = sys.argv[1:4]
rows = {d: defaultdict(list) for d in ("CB8", "CF")}
for d, f in (("CB8", cb), ("CF", cf)):
    for r in map(json.loads, open(f)):
        rows[d][r["stem"]].append(r)
stems = sorted(set(rows["CB8"]) & set(rows["CF"]))
assert len(stems) == 195
mean = {d: {s: float(np.mean([r["tm"] for r in rows[d][s]])) for s in stems} for d in rows}
cats = {"both_succeed": [s for s in stems if mean["CB8"][s] >= 0.5 and mean["CF"][s] >= 0.5],
        "confind_only": [s for s in stems if mean["CB8"][s] < 0.5 and mean["CF"][s] >= 0.5],
        "both_fail": [s for s in stems if mean["CB8"][s] < 0.5 and mean["CF"][s] < 0.5]}
pick = {}
for c, ss in cats.items():
    gap = {s: mean["CF"][s] - mean["CB8"][s] for s in ss}
    med = float(np.median(list(gap.values())))
    chosen = sorted(ss, key=lambda s: (abs(gap[s] - med), s))[:2]
    pick[c] = []
    for s in chosen:
        ent = {"stem": s, "L": rows["CB8"][s][0]["L"], "ref_id": rows["CB8"][s][0]["ref_id"]}
        for d in ("CB8", "CF"):
            rs = sorted(rows[d][s], key=lambda r: r["sample_index"])
            m = float(np.median([r["tm"] for r in rs]))
            r = min(rs, key=lambda r: (abs(r["tm"] - m), r["sample_index"]))
            ent[d] = {"sample_index": r["sample_index"], "tm": r["tm"], "query_mean_tm": mean[d][s], "maps_dir": r["maps_dir"],
                      "tri_file": r["tri_file"], "ckpt": r["ckpt"], "dataset": r["dataset"]}
        assert rows["CB8"][s][0]["ref_id"] == rows["CF"][s][0]["ref_id"]
        pick[c].append(ent)
    print(f"{c}: {len(ss)} queries, median gap {med:+.3f}; picked {[(e['stem'], round(e['CB8']['tm'], 3), round(e['CF']['tm'], 3)) for e in pick[c]]}")
json.dump(pick, open(out, "w"), indent=1)
