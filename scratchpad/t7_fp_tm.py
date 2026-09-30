import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

fp = defaultdict(dict)
for l in open("fp_distance_best.jsonl"):
    r = json.loads(l); fp[r["label"]][(r["stem"], r["sample_index"])] = r
for lab, f in (("CB8", "final_cb8_tri.jsonl"), ("CONFIND", "final18120_cf_tri.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    J = [(r, fp[lab][(r["stem"], r["sample_index"])]) for r in rows]
    assert len(J) == 792
    tm = np.array([r["tm"] for r, _ in J])
    q50 = np.array([np.nan if x["fp_dist"]["q50"] is None else x["fp_dist"]["q50"] for _, x in J])
    q10 = np.array([np.nan if x["fp_dist"]["q10"] is None else x["fp_dist"]["q10"] for _, x in J])
    nfp = np.array([x["n_fp"] for _, x in J]); L = np.array([x["L"] for _, x in J])
    ok = ~np.isnan(q50)
    print(f"{lab}: samples with FP {ok.sum()}/792; FP/L median {np.median(nfp/L):.2f}; FP-dist q10 median {np.nanmedian(q10):.2f}, q50 median {np.nanmedian(q50):.2f}")
    for name, v in (("FP median dist", q50), ("FP q10 dist", q10), ("FP count / L", nfp / L)):
        rho = spearmanr(v[ok], tm[ok])[0]
        by = defaultdict(list)
        for (r, _), a, t, o in zip(J, v, tm, ok):
            if o: by[r["stem"]].append((a, t))
        w = [spearmanr(*zip(*p))[0] for p in by.values() if len({a for a, _ in p}) >= 3 and len({t for _, t in p}) >= 3]
        print(f"   {name:15s} vs TM: pooled rho {rho:+.2f}; within-query rho median {np.median(w):+.2f} (n={len(w)} queries, {sum(x>0 for x in w)} positive)")
