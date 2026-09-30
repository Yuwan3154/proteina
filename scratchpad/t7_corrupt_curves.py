"""T7 (a): degradation of each c2c under matched balanced corruption of its OWN native map (99-query subset).

Per rate: TM median and TM>=0.5 count over (chain, draw) rows, plus a per-chain paired comparison of the two c2cs (chain
mean TM, ConFind minus CB-8, sign test). Rate 0 is the plain native arm (one deterministic row per chain).
Usage: python scratchpad/t7_corrupt_curves.py CB8:<rate>=<jsonl> ... CF:<rate>=<jsonl> ... [--fig out.png]
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--fig")
    a = ap.parse_args()
    data = defaultdict(dict)
    for s in a.arms:
        k, f = s.split("=", 1)
        model, rate = k.split(":")
        rows = [json.loads(l) for l in open(f) if l.strip()]
        assert all(abs((r.get("corrupt_rate") or 0.0) - float(rate)) < 1e-9 for r in rows), f"{f}: corrupt_rate != {rate}"
        by = defaultdict(list)
        for r in rows:
            by[r["stem"]].append(r["tm"])
        data[model][float(rate)] = by
    rates = sorted(set.intersection(*[set(d) for d in data.values()]))
    stems = sorted(set.intersection(*[set(b) for d in data.values() for b in d.values()]))
    print(f"chains in all arms: {len(stems)}; rates {rates}")
    print(f"{'rate':>5} | {'model':>5} {'rows':>5} {'TM med':>7} {'TM>=0.5':>9} | CF-CB8 chain-mean: med diff, CF better/worse, sign p")
    curves = defaultdict(list)
    for rt in rates:
        cm = {}
        for m in ("CB8", "CF"):
            tms = np.array([t for s in stems for t in data[m][rt][s]])
            cm[m] = np.array([np.mean(data[m][rt][s]) for s in stems])
            curves[m].append((rt, np.median(tms), (tms >= 0.5).mean()))
            print(f"{rt:5.2f} | {m:>5} {len(tms):5d} {np.median(tms):7.3f} {(tms >= 0.5).sum():4d} ({(tms >= 0.5).mean():4.0%})", end="")
            print(" |" if m == "CB8" else "", end="\n" if m == "CB8" else "")
        d = cm["CF"] - cm["CB8"]
        nb, nw = int((d > 0).sum()), int((d < 0).sum())
        print(f"      {np.median(d):+.3f}, {nb}/{nw}, p={binomtest(nb, nb + nw).pvalue:.1e}")
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for m, c, lab in (("CB8", "#3a6ea5", "CB-8 c2c (tbeta final)"), ("CF", "#c46b2d", "ConFind c2c (twin)")):
            r, med, frac = zip(*curves[m])
            ax[0].plot(r, med, "-o", color=c, lw=2, ms=8, label=lab)
            ax[1].plot(r, frac, "-o", color=c, lw=2, ms=8, label=lab)
        for x, yl in zip(ax, ("median TM", "fraction TM >= 0.5")):
            x.set_xlabel("balanced corruption rate (precision = recall = 1 - rate)")
            x.set_ylabel(yl); x.set_ylim(0, 1); x.set_yticks(np.arange(0, 1.01, 0.2)); x.grid(alpha=0.3)
        ax[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(a.fig, dpi=200)
        print(f"WROTE {a.fig}")


if __name__ == "__main__":
    main()
