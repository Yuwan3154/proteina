"""Extract the validation chains with NO close training fold -- the honest evaluation subset.

90.2% of validation clusters have a same-fold training neighbour (TM >= 0.5), so aggregate
validation numbers are dominated by fold recall. The chains BELOW that threshold are the ones where
the model must actually thread onto a topology it has not memorised, and they are the set worth
reporting.

⛔ Chains with NO hit at all never appear in hits.tsv, so they must be recovered by differencing
against the written query list -- not by filtering hits.tsv, which would silently drop the most
novel chains of all.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="/orcd/scratch/orcd/011/chenxiou/valleak_all/hits.tsv")
    ap.add_argument("--query_dir", default="/orcd/scratch/orcd/011/chenxiou/valleak_all/query")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="TM below which a chain counts as novel-fold (0.5 = standard same-fold cut)")
    ap.add_argument("--out", default="/orcd/scratch/orcd/011/chenxiou/novel_fold_val_chains.txt")
    args = ap.parse_args()

    written = set()
    for root, _, files in os.walk(args.query_dir):
        for f in files:
            if f.endswith(".pdb"):
                written.add(f[:-4])

    best = {}
    with open(args.hits) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            q, t = f[0], f[1]
            if q == t:
                continue
            try:
                qtm = float(f[3])
            except ValueError:
                continue
            stem = q[:-4] if q.endswith(".pdb") else q
            if stem not in best or qtm > best[stem][1]:
                best[stem] = (t, qtm)

    novel = []
    for c in sorted(written):
        if c not in best:
            novel.append((c, None, 0.0))
        elif best[c][1] < args.threshold:
            novel.append((c, best[c][0], best[c][1]))

    with open(args.out, "w") as fh:
        for c, _, _ in novel:
            fh.write(c + "\n")

    print(f"validation chains searched : {len(written)}")
    print(f"novel-fold (TM < {args.threshold} or no hit) : {len(novel)}  "
          f"({100.0*len(novel)/max(len(written),1):.1f}%)")
    print(f"\n{'chain':>9} {'nearest training':>17} {'TM':>7}")
    for c, t, tm in novel:
        print(f"{c:>9} {(t or '-- none --'):>17} {tm:>7.3f}")
    print(f"\nwrote {args.out}")
    print("\n⚠️ This is the subset where generation quality means threading, not recall. It is also")
    print("   a LOWER bound on novelty: only 13152 of 259k training chains were searched, so some of")
    print("   these may yet have a close neighbour in the unsearched remainder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
