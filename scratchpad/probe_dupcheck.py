"""Are the captured probe samples actually distinct, or duplicated by self-conditioning?

The trainer forwards the SAME batch twice per step when self-conditioning is on, and the capture
hooks every nn forward without deduping. This project has been bitten by exactly that before (an
n=30 that was really n=18). If the 150 samples contain duplicate (query, reference) pairs with
identical features, the effective sample count is smaller than reported.

Splitting is by query chain, so duplicates cannot leak across train/test -- but they would inflate
the apparent n and unevenly weight whichever chains got duplicated.
"""

import glob
import hashlib
import os
import sys
from collections import Counter

import torch


def main():
    d = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(d, "s*.pt")))
    pairs, fhashes, rows = Counter(), Counter(), []
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        key = (s["query"], s["ref"])
        pairs[key] += 1
        h = hashlib.md5(s["feat"].numpy().tobytes()).hexdigest()
        fhashes[h] += 1
        rows.append((s["query"], s["ref"], int(s["L"]), int(s["T"]), int(s["Q"]),
                     int(s["gt"].sum()), h))

    n = len(rows)
    uq = len({r[0] for r in rows})
    upair = len(pairs)
    ufeat = len(fhashes)
    dup_pairs = {k: v for k, v in pairs.items() if v > 1}
    dup_feats = {k: v for k, v in fhashes.items() if v > 1}

    print(f"samples on disk        : {n}")
    print(f"unique query chains    : {uq}")
    print(f"unique (query, ref)    : {upair}")
    print(f"unique FEATURE tensors : {ufeat}   <- the decisive number")
    print()
    print(f"(query,ref) pairs appearing >1x : {len(dup_pairs)}  "
          f"covering {sum(dup_pairs.values())} samples")
    print(f"IDENTICAL feature tensors >1x   : {len(dup_feats)}  "
          f"covering {sum(dup_feats.values())} samples")
    if dup_feats:
        print("\n  => self-conditioning DOUBLE-COUNTING CONFIRMED: identical features stored twice.")
        print(f"  => effective distinct examples = {ufeat}, not {n}")
    else:
        print("\n  => no identical feature tensors: samples are genuinely distinct forwards")
        print("     (repeat queries are same-chain-different-reference, or a different noise level t)")

    cells = sum(r[2] * r[3] for r in rows)
    pos = sum(r[5] for r in rows)
    print(f"\ncells (L*T) total : {cells:,}")
    print(f"positive cells    : {pos:,}  ({pos / max(cells,1):.2%})")
    print(f"L median {sorted(r[2] for r in rows)[n//2]}   T median {sorted(r[3] for r in rows)[n//2]}")

    print("\n  most repeated (query, ref) pairs:")
    for k, v in pairs.most_common(5):
        print(f"    {k[0]:12s} -> {k[1]:12s}  x{v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
