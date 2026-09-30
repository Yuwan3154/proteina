"""T7 figure E (user 2026-09-30): TM-score at matched sampled-map quality, CB-8 vs ConFind.

Bins each tri sample by a map-quality metric (each definition scored against its own native map) and plots the median
TM per bin, so the two pipelines are compared at EQUAL map quality. Palette as t7_figures.py.

Usage: python scratchpad/t7_fig_matched.py CB8_TRI.jsonl CONFIND_TRI.jsonl OUT.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"CB-8": "#2a78d6", "ConFind": "#eb6834"}
MARK = {"CB-8": "o", "ConFind": "s"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
EDGES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
METRICS = (("tri_contact_f1", "F1 at 0.5 (what the c2c sees)"), ("tri_contact_precision_at_L", "P@L"))
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

data = {"CB-8": [json.loads(l) for l in open(sys.argv[1])], "ConFind": [json.loads(l) for l in open(sys.argv[2])]}
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
fig.subplots_adjust(wspace=0.12)
centers = list(range(len(EDGES) - 1))  # evenly spaced categorical bins (bin widths differ)
labels = [f"{a:.1f}-{min(b, 1.0):.1f}" for a, b in zip(EDGES[:-1], EDGES[1:])]
for ax, (key, name) in zip(axs, METRICS):
    for d, rows in data.items():
        med, ns = [], []
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            t = [r["tm"] for r in rows if a <= r[key] < b]
            med.append(np.median(t) if t else np.nan)
            ns.append(len(t))
        ax.plot(centers, med, color=COL[d], lw=2, marker=MARK[d], ms=8, label=d)
    ax.set(xlim=(-0.5, len(centers) - 0.5), ylim=(0, 1), xticks=centers, yticks=np.arange(0, 1.01, 0.2),
           xlabel=f"sampled-map {name}, binned")
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"Median TM per {name.split(' (')[0]} bin", fontsize=10.5, color=INK, loc="left")
axs[0].set_ylabel("TM-score vs native (median)")
axs[0].legend(frameon=False, loc="lower right")
fig.text(0.01, -0.04, "Each map scored against its own definition's native map; 792 samples per pipeline, 99 queries.",
         color=MUTED, fontsize=9)
fig.savefig(sys.argv[3])
print(f"[figure] {sys.argv[3]}")
