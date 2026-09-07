"""Is generation quality FOLD RECALL? Join per-chain TM against distance to the nearest training fold.

12/12 probed validation chains have a same-fold training neighbour, so the headline TM 0.91-0.96
cannot be read as generalisation. This asks the sharper question: within the validation set, does
generation quality TRACK how close the nearest training fold is?

  strong positive relation -> the model is largely recalling folds it has seen; the number to
      report is performance on the chains with the MOST DISTANT training neighbour.
  no relation             -> quality is driven by something else (length, contact density), and
      contamination inflates the absolute level without explaining the spread.

⛔ Reports the per-chain table and threshold COUNTS, not just a correlation. With n in the tens a
single chain moves a correlation coefficient, and the distribution here is bimodal by construction
(mirrored chains sit near TM 0.3, correct ones near 0.95), so a pooled r would mostly measure the
mirror rate rather than any homology effect. Mirrored chains are therefore reported SEPARATELY.
"""

import argparse
import csv
import os
import sys


def load_hits(path):
    """chain -> best TM to a training chain, normalised by the QUERY (never max)."""
    best = {}
    with open(path) as fh:
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
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_csv", required=True, help="gen_results.csv from gen_c2c_structures.py")
    ap.add_argument("--hits", required=True, help="foldseek hits.tsv, val queries vs training")
    args = ap.parse_args()

    best = load_hits(args.hits)
    rows = []
    with open(args.gen_csv) as fh:
        for r in csv.DictReader(fh):
            cid = r["chain_id"]
            if cid not in best or not r["tm"]:
                continue
            rows.append((cid, float(r["tm"]), int(r["mirrored"]), best[cid][1], best[cid][0],
                         int(r["length"])))
    if not rows:
        print("no chains in common between the probe and the leakage search")
        return 1

    rows.sort(key=lambda x: x[3])
    print(f"{'chain':>9} {'L':>5} {'gen TM':>8} {'nearest train':>14} {'train TM':>9}  flag")
    for cid, tm, mir, htm, ht, L in rows:
        print(f"{cid:>9} {L:>5} {tm:>8.3f} {ht:>14} {htm:>9.3f}  {'MIRROR' if mir else ''}")

    ok = [r for r in rows if not r[2]]
    mir = [r for r in rows if r[2]]
    print(f"\ncorrect-handed n={len(ok)}   mirrored n={len(mir)}")

    # ⭐ The comparison that matters: among CORRECT-handed chains only (so the bimodal mirror split
    # cannot drive it), split at the median training-TM and compare generation quality.
    if len(ok) >= 4:
        s = sorted(ok, key=lambda x: x[3])
        half = len(s) // 2
        near, far = s[half:], s[:half]
        mn = sum(r[1] for r in near) / len(near)
        mf = sum(r[1] for r in far) / len(far)
        print(f"  nearest-training-fold ABOVE median (n={len(near)}, train TM "
              f"{near[0][3]:.3f}-{near[-1][3]:.3f}): mean gen TM {mn:.3f}")
        print(f"  nearest-training-fold BELOW median (n={len(far)}, train TM "
              f"{far[0][3]:.3f}-{far[-1][3]:.3f}): mean gen TM {mf:.3f}")
        print(f"  difference: {mn - mf:+.3f}")
        print("  ⚠️ A positive difference means quality tracks homolog proximity, i.e. fold recall.")
        print("     n is small -- read this as a direction to test, not an effect size.")

    if mir:
        tms = sorted(r[3] for r in mir)
        oks = sorted(r[3] for r in ok)
        print(f"\nnearest-training-fold TM, mirrored: {[round(v, 3) for v in tms]}")
        print(f"nearest-training-fold TM, correct : {[round(v, 3) for v in oks]}")
        print("  ⚠️ Overlapping ranges => handedness failure is NOT explained by fold novelty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
