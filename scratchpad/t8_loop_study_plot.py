"""T8 loop-element feature review figures (for the user's review, 2026-10-09 directive) from t8_loop_feature_study.py output.

1. Elements per chain with loops as elements (all / helix+strand / loop) vs the 96 cap; the fraction of chains truncated.
2. Element lengths by type (loops have one token, so their length reaches the model only through the length feature).
3. Every pair feature by element-type pair (H-H, H-E, E-E, H-L, E-L, L-L), sequence-ADJACENT pairs (elem_sep 1) apart:
   contact_max / contact_frac / seq_gap / the 4 circuit channels.
4. Orientation (cos between element axes) under the two loop-axis options, for pairs involving a loop, by loop length.
Usage: python t8_loop_study_plot.py STUDY.npz OUT_DIR
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAME = {0: "L", 1: "H", 2: "E"}
PAIRS = ["H-H", "H-E", "E-E", "H-L", "E-L", "L-L"]
BLUE, GREY = "#2a78d6", "#8a8f98"


def nice_axis(lo, hi, target=5):
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    s = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * 1.0000001), 10 * mag) if raw > 0 else 1.0
    a = math.floor(lo / s + 1e-9) * s
    b = math.ceil(hi / s - 1e-9) * s
    return a, (b if b > a else a + s), s


def ticks(ax, vals):
    lo, hi, s = nice_axis(min(vals), max(vals))
    n = int(round((hi - lo) / s))
    ax.set_ylim(lo, hi)
    ax.set_yticks([lo + k * s for k in range(n + 1)])


def pair_name(a, b):
    x, y = sorted([NAME[int(a)], NAME[int(b)]], key="HEL".index)
    return f"{x}-{y}"


def main(study, out_dir):
    z = np.load(study)
    C, E, P = z["chains"], z["elements"], z["pairs"]
    cols = {c: i for i, c in enumerate(z["pair_cols"])}
    os.makedirs(out_dir, exist_ok=True)
    L, T, nH, nE, nL = C[:, 1], C[:, 2], C[:, 3], C[:, 4], C[:, 5]
    lines = [f"chains {len(C)} (skipped {len(z['skipped'])}), elements {len(E)}, element pairs {len(P)}",
             f"elements per chain (loops included): median {np.median(T):.0f}, p95 {np.percentile(T, 95):.0f}, p99 "
             f"{np.percentile(T, 99):.0f}, max {T.max():.0f}; > 96 (truncated): {int((T > 96).sum())} chains "
             f"({100 * (T > 96).mean():.2f}%); helix+strand only: median {np.median(nH + nE):.0f}, p95 {np.percentile(nH + nE, 95):.0f}",
             f"loops per chain median {np.median(nL):.0f}; chain length median {np.median(L):.0f}, max {L.max():.0f}"]
    for t in (0, 1, 2):
        ln = E[E[:, 1] == t, 2]
        lines.append(f"{NAME[t]} element length: median {np.median(ln):.0f}, p5 {np.percentile(ln, 5):.0f}, p95 "
                     f"{np.percentile(ln, 95):.0f}; 1-residue {100 * (ln == 1).mean():.1f}%, <=2 {100 * (ln <= 2).mean():.1f}%")
    names = np.array([pair_name(a, b) for a, b in zip(P[:, cols["type_a"]], P[:, cols["type_b"]])])
    adj = P[:, cols["elem_sep"]] == 1
    feats = ["contact_max", "contact_frac", "seq_gap", "circ_series", "circ_contains", "circ_inside", "circ_cross"]
    lines.append("pair features, mean over all element pairs pooled across chains (pair-weighted), adjacent | non-adjacent:")
    for pn in PAIRS:
        m = names == pn
        if not m.any():
            continue
        lines.append(f"  {pn}: n {int((m & adj).sum())} | {int((m & ~adj).sum())}; " + "; ".join(
            f"{f} {P[m & adj, cols[f]].mean() if (m & adj).any() else float('nan'):.3f} | {P[m & ~adj, cols[f]].mean():.3f}"
            for f in feats))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(0, max(T.max(), 100) + 5, 4) + 0.5  # edges at 4k + 0.5, so 96 | 97 falls on the edge 96.5
    ax[0].hist(T, bins=bins, color=BLUE, label="all elements (loops included)")
    ax[0].hist(nH + nE, bins=bins, color=GREY, alpha=0.7, label="helix + strand only (old axis)")
    ax[0].axvline(96, color="black", lw=1, ls="--")
    ax[0].text(97, ax[0].get_ylim()[1] * 0.9, "cap 96", fontsize=9)
    ax[0].set_xlabel("elements per chain")
    ax[0].set_ylabel("chains")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].scatter(L, T, s=4, color=BLUE)
    ax[1].axhline(96, color="black", lw=1, ls="--")
    ax[1].set_xlabel("chain length (residues)")
    ax[1].set_ylabel("elements (loops included)")
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loop_study_counts.png"), dpi=140)

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for k, f in enumerate(feats + ["elem_len"]):
        a = ax.flat[k]
        if f == "elem_len":
            for t, c in ((1, "#4c8bf5"), (2, "#e8a33d"), (0, GREY)):
                ln = E[E[:, 1] == t, 2]
                a.hist(np.clip(ln, 0, 40), bins=np.arange(0.5, 41.5, 1), histtype="step", lw=1.6, color=c, label=NAME[t])
            a.set_xlabel("element length (clipped at 40)")
            a.set_ylabel("elements")
            a.legend(frameon=False, fontsize=9)
            continue
        xs = np.arange(len(PAIRS))
        nan = float("nan")  # an empty group draws no bar (a 0 bar would read as a real mean of 0)
        va = [P[(names == pn) & adj, cols[f]].mean() if ((names == pn) & adj).any() else nan for pn in PAIRS]
        vn = [P[(names == pn) & ~adj, cols[f]].mean() if ((names == pn) & ~adj).any() else nan for pn in PAIRS]
        a.bar(xs - 0.2, va, 0.4, color=BLUE, label="sequence-adjacent")
        a.bar(xs + 0.2, vn, 0.4, color=GREY, label="non-adjacent")
        a.set_xticks(xs)
        a.set_xticklabels(PAIRS)
        a.set_title(f"mean {f} (pair-weighted; missing bar = no pairs)", fontsize=9)
        ticks(a, [v for v in va + vn if v == v] + [0])
        if k == 0:
            a.legend(frameon=False, fontsize=8)
    for a in ax.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loop_study_pair_features.png"), dpi=130)

    # orientation: pairs with a loop, by that loop's length (the shorter loop if both are loops)
    ta, tb = P[:, cols["type_a"]], P[:, cols["type_b"]]
    la, lb = P[:, cols["len_a"]], P[:, cols["len_b"]]
    has_loop = (ta == 0) | (tb == 0)
    loop_len = np.where((ta == 0) & (tb == 0), np.minimum(la, lb), np.where(ta == 0, la, lb))
    # a 1-residue loop has < 2 CA, so it has NO axis under either option (cos 0): its own panel, so its spike at 0 is
    # not read as geometry of the end_to_end option
    groups = [("helix/strand pairs (unchanged)", ~has_loop), ("1-residue loop (no axis in either mode)", has_loop & (loop_len == 1)),
              ("2-residue loop", has_loop & (loop_len == 2)), ("loop of 3-5", has_loop & (loop_len >= 3) & (loop_len <= 5)),
              ("loop of >= 6", has_loop & (loop_len >= 6))]
    fig, ax = plt.subplots(1, 5, figsize=(23, 4.3))
    for a, (lab, m) in zip(ax, groups):
        col = "cos_zero" if lab.startswith("helix") else "cos_e2e"
        a.hist(P[m, cols[col]], bins=np.linspace(-1, 1, 41), color=GREY if lab.startswith("helix") else BLUE)
        z0 = float((P[m, cols[col]] == 0).mean()) if m.any() else float("nan")
        a.set_title(f"{lab}: n {int(m.sum())}, exactly 0: {100 * z0:.1f}%"
                    + ("" if lab.startswith("helix") else "\nend_to_end axis ('zero' puts every loop pair at 0)"), fontsize=9)
        a.set_xlabel("cosine between element axes")
        a.set_ylabel("element pairs")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        if not lab.startswith("helix"):
            lines.append(f"orientation, {lab}: end_to_end |cos| median {np.median(np.abs(P[m, cols['cos_e2e']])):.3f}, "
                         f"exactly 0 {100 * z0:.1f}% (n={int(m.sum())})")
    open(os.path.join(out_dir, "loop_study_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loop_study_orientation.png"), dpi=130)
    print(f"[plot] -> {out_dir}")


if __name__ == "__main__":
    assert len(sys.argv) == 3, __doc__
    main(*sys.argv[1:])
