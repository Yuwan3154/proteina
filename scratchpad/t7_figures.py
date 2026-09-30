"""T7 slide figures (user 2026-09-30): CB-8 vs ConFind on the same (query, template) pairs.

Reads the per-sample table written by stageB_pairs_analyze.py --out (def, stem, ref_id, sample_index, tm, rmsd_proper,
refl_sign, tri_* ...) and, optionally, each definition's native-map JSONL (the ceiling). Writes static PNGs for slides.
Palette: reference categorical slots 1-2 (blue CB-8, orange ConFind) + gray ceiling; marker shape as second encoding.

Usage: python scratchpad/t7_figures.py MERGED.tsv OUT_DIR [CB8_NATIVE=a.jsonl] [CONFIND_NATIVE=b.jsonl] [SUBSET=list]
"""

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

COL = {"CB8": "#2a78d6", "CONFIND": "#eb6834"}
MARK = {"CB8": "o", "CONFIND": "s"}
NAME = {"CB8": "CB-8", "CONFIND": "ConFind"}
INK, MUTED, GRID, CEIL = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(tsv, subset):
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    by = defaultdict(list)
    for r in rows:
        if subset is None or r["stem"] in subset:
            by[(r["def"], r["stem"])].append(r)
    return by


def f(r, k):
    v = r.get(k)
    return float(v) if v not in (None, "", "None") else np.nan


def main():
    tsv, out = sys.argv[1], sys.argv[2]
    kv = dict(a.split("=", 1) for a in sys.argv[3:])
    subset = {ln.split()[0] for ln in open(kv["SUBSET"]) if ln.strip()} if "SUBSET" in kv else None
    os.makedirs(out, exist_ok=True)
    by = load(tsv, subset)
    stems = sorted({s for d, s in by if d == "CB8"} & {s for d, s in by if d == "CONFIND"})
    nat = {d: {json.loads(l)["stem"]: json.loads(l)["tm"] for l in open(kv[f"{d}_NATIVE"])}
           for d in ("CB8", "CONFIND") if f"{d}_NATIVE" in kv}
    qm = {d: np.array([np.mean([f(r, "tm") for r in by[(d, s)]]) for s in stems]) for d in ("CB8", "CONFIND")}

    # A: per-query mean TM, CB-8 (x) vs ConFind (y)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=0)
    ax.scatter(qm["CB8"], qm["CONFIND"], s=26, color=MUTED, edgecolor="white", linewidth=0.8, zorder=2)
    nb = int((qm["CONFIND"] > qm["CB8"]).sum())
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, 1.01, 0.2),
           xlabel="CB-8 pipeline: mean TM over 8 samples", ylabel="ConFind pipeline: mean TM over 8 samples")
    ax.set_title(f"Same query-template pairs ({len(stems)} queries)\nConFind higher on {nb}, CB-8 higher on "
                 f"{len(stems) - nb}", fontsize=11, color=INK, loc="left")
    fig.savefig(os.path.join(out, "A_pairs_query_mean_tm.png"))
    plt.close(fig)

    # B: TM distributions, tri-sampled (per sample) vs native ceiling
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for i, d in enumerate(("CB8", "CONFIND")):
        tm = np.array([f(r, "tm") for s in stems for r in by[(d, s)]])
        bp = ax.boxplot(tm, positions=[i * 2.6], widths=0.6, patch_artist=True, showfliers=False)
        for p in bp["boxes"]:
            p.set(facecolor=COL[d], alpha=0.85, edgecolor=COL[d])
        for k in ("whiskers", "caps", "medians"):
            for p in bp[k]:
                p.set(color=INK if k == "medians" else COL[d], lw=1.5)
        if d in nat:
            nt = np.array([nat[d][s] for s in stems if s in nat[d]])
            bq = ax.boxplot(nt, positions=[i * 2.6 + 1.0], widths=0.45, patch_artist=True, showfliers=False)
            for p in bq["boxes"]:
                p.set(facecolor="white", edgecolor=CEIL, hatch="///")
            for k in ("whiskers", "caps", "medians"):
                for p in bq[k]:
                    p.set(color=CEIL, lw=1.2)
    ax.set_xticks([0, 1.0, 2.6, 3.6], ["CB-8\nsampled", "CB-8\nnative map", "ConFind\nsampled", "ConFind\nnative map"])
    ax.set(ylim=(0, 1), yticks=np.arange(0, 1.01, 0.2), ylabel="TM-score vs native")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Structure quality from tri-sampled maps vs each pipeline's native-map ceiling", fontsize=11,
                 color=INK, loc="left")
    fig.savefig(os.path.join(out, "B_tm_distributions.png"))
    plt.close(fig)

    # C: TM vs sampled-map P@L, one panel per definition
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    within = {}
    for ax, d in zip(axs, ("CB8", "CONFIND")):
        x = np.array([f(r, "tri_contact_precision_at_L") for s in stems for r in by[(d, s)]])
        y = np.array([f(r, "tm") for s in stems for r in by[(d, s)]])
        ax.scatter(x, y, s=10, color=COL[d], alpha=0.35, marker=MARK[d], linewidth=0)
        mx = [np.mean([f(r, "tri_contact_precision_at_L") for r in by[(d, s)]]) for s in stems]
        ax.scatter(mx, qm[d], s=22, color=COL[d], edgecolor="white", linewidth=0.8, marker=MARK[d])
        w = []
        for s in stems:
            xs = np.array([f(r, "tri_contact_precision_at_L") for r in by[(d, s)]])
            ts = np.array([f(r, "tm") for r in by[(d, s)]])
            if len(np.unique(xs)) >= 3 and len(np.unique(ts)) >= 2:
                w.append(spearmanr(xs, ts).statistic)
        within[d] = np.array(w)
        rp = spearmanr(x, y).statistic
        rb = spearmanr(mx, qm[d]).statistic
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, 1.01, 0.2),
               xlabel=f"sampled-map P@L vs {NAME[d]} native map")
        ax.set_title(f"{NAME[d]}\nSpearman rho: pooled {rp:+.2f}, between-query {rb:+.2f}\nwithin-query median "
                     f"{np.median(w) if w else float('nan'):+.2f} (n={len(w)} queries)", fontsize=9.5, color=INK,
                     loc="left")
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("TM-score vs native")
    fig.text(0.01, -0.03, "faint: one tri sample each; solid: per-query means. rho = Spearman.", color=MUTED,
             fontsize=9)
    fig.savefig(os.path.join(out, "C_tm_vs_pal.png"))
    plt.close(fig)

    # D: distribution of within-query Spearman rho
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, d in enumerate(("CB8", "CONFIND")):
        w = within[d]
        if len(w):
            jit = (np.random.default_rng(0).random(len(w)) - 0.5) * 0.3
            ax.scatter(np.full(len(w), i) + jit, w, s=16, color=COL[d], alpha=0.7, marker=MARK[d], linewidth=0)
            ax.plot([i - 0.25, i + 0.25], [np.median(w)] * 2, color=INK, lw=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(xticks=[0, 1], xticklabels=["CB-8", "ConFind"], ylim=(-1, 1), yticks=np.arange(-1, 1.01, 0.5),
           ylabel="within-query Spearman rho\n(TM vs sampled-map P@L, 8 samples)")
    ax.set_title("Does a better sampled map give a better structure for the SAME query?", fontsize=10, color=INK,
                 loc="left")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(out, "D_within_query_rho.png"))
    plt.close(fig)
    print(f"[figures] {len(stems)} queries -> {out}: A_pairs_query_mean_tm, B_tm_distributions, C_tm_vs_pal, "
          f"D_within_query_rho")


if __name__ == "__main__":
    main()
