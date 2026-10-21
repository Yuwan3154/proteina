"""Validation-sampling P@L median and validation contact loss vs optimizer step, CB-8 tri (tri_cb8synth_v5) and ConFind
fine-tune (tri_confindsynth_ft), from the full unsampled W&B export (runcompare agent, 2026-10-20). Each run is scored
against ITS OWN contact label, so each sits in its own panel: read the trend within a panel, never across.
Usage: python scratchpad/run_compare_curves_fig.py IN_DIR OUT.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"tri_cb8synth_v5": "#2a78d6", "tri_confindsynth_ft": "#eb6834"}
NAME = {"tri_cb8synth_v5": "CB-8 tri (from scratch)", "tri_confindsynth_ft": "ConFind tri fine-tune"}
# the fine-tune counts its own steps; its 71,950 pretrain steps (old tri_full384) precede x = 0
XLAB = {"tri_cb8synth_v5": "Optimizer step (thousands)", "tri_confindsynth_ft": "Fine-tune step (thousands), after 71,950 pretrain steps"}
INK, MUTED, GRID = "#1b2330", "#52514e", "#e6e5e1"
BIN = 10_000


def binned(x, y):
    edges = np.arange(0, x.max() + BIN, BIN)
    k = np.digitize(x, edges) - 1
    cx, cy = [], []
    for b in np.unique(k):
        m = k == b
        if m.sum() >= 3:
            cx.append(x[m].mean())
            cy.append(y[m].mean())
    return np.array(cx), np.array(cy)


def main():
    src, out = sys.argv[1], sys.argv[2]
    runs = list(COL)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, constrained_layout=True)
    for j, run in enumerate(runs):
        s = pd.read_csv(f"{src}/{run}_sampling.csv")
        v = pd.read_csv(f"{src}/{run}_valpass.csv")
        x, y = s["gs"].to_numpy(float) / 1000, s["validation_sampling/contact_precision_at_L_median"].to_numpy(float)
        ax = axes[0, j]
        ax.plot(x, y, "o", ms=4, color=COL[run], alpha=0.35, mec="none")
        bx, by = binned(x * 1000, y)
        ax.plot(bx / 1000, by, "-", lw=2.5, color=COL[run])
        ax.set_ylim(0.0, 0.5)
        ax.set_yticks(np.arange(0.0, 0.51, 0.1))
        ax.set_title(NAME[run], color=INK, fontsize=16, loc="left")
        if j == 0:
            ax.set_ylabel("Sampling P@L, median of 32 chains", color=INK, fontsize=13)
        xv, yv = v["gs"].to_numpy(float) / 1000, v["validation_loss/contact_map_loss_epoch"].to_numpy(float)
        ax = axes[1, j]
        ax.plot(xv, yv, "o", ms=2, color=COL[run], alpha=0.15, mec="none")
        bx, by = binned(xv * 1000, yv)
        ax.plot(bx / 1000, by, "-", lw=2.5, color=COL[run])
        ax.set_ylim(0.025, 0.050)
        ax.set_yticks(np.arange(0.025, 0.0501, 0.005))
        if j == 0:
            ax.set_ylabel("Validation contact loss", color=INK, fontsize=13)
        ax.set_xlabel(XLAB[run], color=INK, fontsize=13)
    for ax in axes.flat:
        ax.set_xlim(0, 150)
        ax.set_xticks(np.arange(0, 151, 25))
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    fig.savefig(out, dpi=150, facecolor="#ffffff")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
