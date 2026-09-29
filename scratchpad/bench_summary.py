"""Summarise the 195-chain native-map benchmark arms for the training-status report (user 2026-09-29).

Same metrics as stageB_compare.py on the pre-registered primary set (stageA_val_1percluster.txt): TM quartiles,
count at TM >= 0.5, median proper RMSD, median dist_mae, mirror rate (mean refl_sign); plus, per twin arm, the paired
Wilcoxon against the previous twin arm and against the CB-8 reference arm. Writes one JSON the stdlib-only report
builder renders.

Usage: python scratchpad/bench_summary.py <lists_dir> <out.json> REF=path.jsonl ARM=path.jsonl [ARM=path.jsonl ...]
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

lists, out = sys.argv[1], sys.argv[2]
specs = [a.split("=", 1) for a in sys.argv[3:]]
assert specs[0][0] == "REF" and all(k == "ARM" for k, _ in specs[1:]), "usage: REF=... then ARM=..."


def load(path):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    by = {r["stem"]: r for r in rows}
    assert len(by) == len(rows) == 195, f"{path}: {len(rows)} rows, {len(by)} chains (expected 195, n_seeds 1)"
    assert all(r["map_source"] == "native" and r["weights"] == "ema" for r in rows), f"{path}: not native/EMA"
    return by


ref = load(specs[0][1])
arms = [load(p) for _, p in specs[1:]]
chains = [ln.strip() for ln in open(os.path.join(lists, "stageA_val_1percluster.txt")) if ln.strip()]
chains = [s for s in chains if s in ref and all(s in a for a in arms)]
assert len(chains) == 191, len(chains)
for a in arms:
    assert all(a[s]["seq_fp"] == ref[s]["seq_fp"] for s in chains), "sequence fingerprints differ"


def col(by, k):
    return np.array([by[s][k] for s in chains], dtype=float)


def stats(by):
    tm = col(by, "tm")
    q = np.percentile(tm, [25, 50, 75])
    r0 = by[chains[0]]
    return dict(step=int(r0["global_step"]), contact_def=r0["contact_def"], tm_q25=float(q[0]), tm_median=float(q[1]),
                tm_q75=float(q[2]), n_tm05=int((tm >= 0.5).sum()), rmsd_median=float(np.median(col(by, "rmsd_proper"))),
                distmae_median=float(np.median(col(by, "dist_mae"))), mirror=float(col(by, "refl_sign").mean()))


def paired(a, b):
    ta, tb = col(a, "tm"), col(b, "tm")
    d = tb - ta
    return dict(better=int((d > 0).sum()), worse=int((d < 0).sum()), median_d=float(np.median(d)),
                p=float(wilcoxon(ta, tb).pvalue) if np.any(d != 0) else 1.0)


res = dict(n_chains=len(chains), chain_set="primary 191 (max384 val, 1/cluster)", ref=stats(ref), arms=[])
for i, a in enumerate(arms):
    e = stats(a)
    e["vs_prev"] = paired(arms[i - 1], a) if i else None
    e["vs_ref"] = paired(ref, a)
    res["arms"].append(e)
json.dump(res, open(out, "w"), indent=1)
print(f"wrote {out}: ref step {res['ref']['step']}, {len(arms)} arms, {len(chains)} chains")
for e in res["arms"]:
    print(f"  step {e['step']:>6} TM med {e['tm_median']:.3f} n>=0.5 {e['n_tm05']} mirror {e['mirror']:.3f} "
          f"vs_ref {e['vs_ref']['better']}/{e['vs_ref']['worse']} p={e['vs_ref']['p']:.2g}")
