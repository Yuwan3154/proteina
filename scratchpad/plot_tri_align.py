"""Tri QxT alignment head curves (user 2026-10-08): per-epoch train/validation precision@Q of the head and the SAME metric on
the position-only (diagonal) baseline, plus the alignment loss, for the CB-8 tri and the ConFind tri fine-tune.

precision@Q: per sample, the top-Q scored (query residue, template element) cells, Q = the number of query residues that
truly align to some element, scored as the fraction that are true; the baseline scores each cell by -|i - element midpoint|.
Y axes step on the 1/2/2.5/5 x 10^k ladder (nice_axis_reticker). Reads tri_epochs_<run>.json from a report snapshot.
Usage: python plot_tri_align.py SNAPSHOT_DIR OUT.png
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# the 1/2/2.5/5 x 10^k tick ladder of nice_axis_reticker.py (copied: that module runs its re-ticking script on import)
def nice_axis(lo, hi, target=5):
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    s = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * 1.0000001), 10 * mag) if raw > 0 else 1.0
    a = math.floor(lo / s + 1e-9) * s
    b = math.ceil(hi / s - 1e-9) * s
    return a, (b if b > a else a + s), s


def decimals(step):
    if step >= 1:
        return 0
    d = -math.floor(math.log10(step))
    return int(d + 1 if abs(step * 10 ** d - 2.5) < 1e-9 else d)


snap, out = sys.argv[1], sys.argv[2]
RUNS = [("tri_cb8synth_v5", "CB-8 tri (from scratch)"), ("tri_confindsynth_ft", "ConFind tri (fine-tune; its own epoch counter)")]
COL = {"head": "#2a78d6", "base": "#8a8f98"}


def series(rows, key):
    pts = [(r["epoch"], r[key]) for r in rows if r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def ticks(ax, vals):
    lo, hi, s = nice_axis(min(vals), max(vals))
    ax.set_ylim(lo, hi)
    n = int(round((hi - lo) / s))
    ax.set_yticks([lo + k * s for k in range(n + 1)])
    ax.set_yticklabels([f"{lo + k * s:.{decimals(s)}f}" for k in range(n + 1)])


fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
for c, (rid, title) in enumerate(RUNS):
    rows = json.load(open(os.path.join(snap, f"tri_epochs_{rid}.json")))
    ax = axes[0, c]
    vals = []
    for key, lab, color, ls in (("train_align_p", "head, train", COL["head"], "-"), ("val_align_p", "head, validation", COL["head"], "--"),
                                ("train_align_pos", "position-only baseline, train", COL["base"], "-"),
                                ("val_align_pos", "position-only baseline, validation", COL["base"], "--")):
        x, y = series(rows, key)
        assert x, f"{rid}: no '{key}' values in the snapshot"
        ax.plot(x, y, ls, color=color, lw=1.8 if color == COL["head"] else 1.4, label=lab)
        vals += y
        print(f"{rid} {key}: {len(x)} epochs, first {y[0]:.3f} (ep {x[0]}), last {y[-1]:.3f} (ep {x[-1]})")
    ticks(ax, vals)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("alignment precision@Q")
    ax.legend(fontsize=9, frameon=False, loc="lower center", ncol=2)
    ax = axes[1, c]
    vals = []
    for key, lab, ls in (("train_align", "train", "-"), ("val_align", "validation", "--")):
        x, y = series(rows, key)
        assert x, f"{rid}: no '{key}' values in the snapshot"
        ax.plot(x, y, ls, color=COL["head"], lw=1.8, label=lab)
        vals += y
    ticks(ax, vals)
    ax.set_ylabel("alignment loss (softmax CE)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
for ax in axes.flat:
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("[done]", out)
