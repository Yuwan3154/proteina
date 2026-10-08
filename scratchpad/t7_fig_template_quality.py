"""Deck figure G (user 2026-10-08): template quality (TM template vs native) vs final prediction quality (TM output vs
native), full 195-query set, both definitions (same template per query). Per-query mean over the 8 samples; Spearman
per definition. Usage: python t7_fig_template_quality.py TEMPLATE_TM.tsv CB8.jsonl CF.jsonl OUT.png"""

import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

tsv, cb, cf, out = sys.argv[1:5]
tt = {}
for i, l in enumerate(open(tsv)):
    f = l.rstrip("\n").split("\t")
    if i and f[5] != "nan":
        tt[f[0]] = float(f[5])
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#9aa0a6", zorder=1)
for lab, f, c in (("CB-8", cb, "#3a6ea5"), ("ConFind", cf, "#c46b2d")):
    q = defaultdict(list)
    for r in map(json.loads, open(f)):
        q[r["stem"]].append(r["tm"])
    stems = sorted(set(q) & set(tt))
    x = np.array([tt[s] for s in stems]); y = np.array([np.mean(q[s]) for s in stems])
    rho = spearmanr(x, y)[0]
    above = int((y > x).sum())
    ax.scatter(x, y, s=22, color=c, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=2,
               label=f"{lab}: Spearman {rho:+.2f}; output beats template on {above}/{len(stems)}")
    print(f"{lab}: n={len(stems)} rho={rho:+.3f} median template {np.median(x):.3f} median output {np.median(y):.3f} output>template {above}")
ax.set_xlim(0.4, 1.0); ax.set_ylim(0, 1.0)
ax.set_xticks(np.arange(0.4, 1.01, 0.1)); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel("template TM to native (the reference the tri saw)"); ax.set_ylabel("final TM to native (mean of 8 samples)")
ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(out, dpi=200); print("WROTE", out)
