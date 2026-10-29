"""Row-level torch.equal of a (small) topology index against another (production) index, matched by row id.

Every row of --a whose id is in --b is compared field by field (runs, element contact map, structural features,
alignment target, he_size, row_tm, row_rewind, row_usalign_tm, row_is_native). Prints per-field counts and, for the
fp16 structural features, the max |diff| and the differing channels (no tolerance: a difference is reported, never
accepted here). Exit 1 if any compared row differs, if fewer rows than --min-rows were compared (non-vacuous), or if
--require-all and some row of --a is absent from --b. --natives-only compares native rows only.
Usage: python scratchpad/xrun_index_compare.py --a SMALL.pt --b PROD.pt [--natives-only] [--require-all] [--min-rows N]
"""

import argparse
import sys
from collections import Counter

import torch

from proteinfoundation.datasets.sse_topology import STRUCTURAL_PAIR_FEATURES


def load(path):
    idx = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return idx, {s: i for i, s in enumerate(idx["ids"])}


def sl(idx, flat, off, r):
    return idx[flat][int(idx[off][r]):int(idx[off][r + 1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--natives-only", action="store_true")
    ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--require-b-complete", action="store_true",
                    help="also every row of b under a native stem of a must be present in a (a rebuild dropped nothing)")
    a = ap.parse_args()
    A, ia = load(a.a)
    B, ib = load(a.b)
    cnt, bad, absent = Counter(), [], []
    max_feat = 0.0
    feat_ch = Counter()
    scal = {}  # scalar field -> max |a - b| over differing rows
    nan_side = {}  # scalar field -> Counter of which side is NaN on the differing rows
    n_struct = len(STRUCTURAL_PAIR_FEATURES)
    for s, ra in ia.items():
        nat = bool(A["row_is_native"][ra])
        if a.natives_only and not nat:
            continue
        rb = ib.get(s)
        if rb is None:
            absent.append(s)
            continue
        cnt["compared_native" if nat else "compared_template"] += 1
        diff = []
        for name, flat, off in (("runs", "runs_flat", "runs_offset"), ("he", "he_flat", "he_offset"),
                                ("feat", "feat_flat", "feat_offset"), ("align", "align_flat", "align_offset")):
            x, y = sl(A, flat, off, ra), sl(B, flat, off, rb)
            if x.shape != y.shape or not torch.equal(x, y):
                diff.append(name)
                if name == "feat" and x.shape == y.shape:
                    d = (x.float() - y.float()).abs()
                    max_feat = max(max_feat, float(d.max()))
                    for c in torch.nonzero(d.reshape(-1, n_struct).amax(0) > 0).flatten().tolist():
                        feat_ch[STRUCTURAL_PAIR_FEATURES[c]] += 1
        for k in ("he_size", "row_tm", "row_rewind", "row_usalign_tm", "row_is_native"):
            if not torch.equal(A[k][ra], B[k][rb]):
                diff.append(k)
                scal[k] = max(scal.get(k, 0.0), abs(float(A[k][ra]) - float(B[k][rb])))
                xa, xb = float(A[k][ra]), float(B[k][rb])
                nan_side[k] = nan_side.get(k, Counter()) + Counter({f"a_nan={xa != xa},b_nan={xb != xb}": 1})
        for d in diff:
            cnt[f"differs:{d}"] += 1
        if diff:
            bad.append((s, diff))
    n = cnt["compared_native"] + cnt["compared_template"]
    dropped = []
    if a.require_b_complete:
        stems = {s for s, r in ia.items() if bool(A["row_is_native"][r])}
        dropped = [s for s in ib if s.split("@")[0] in stems and s not in ia]
        print(f"[compare] rows of b under a's {len(stems)} natives missing from a: {len(dropped)} {dropped[:10]}")
    print(f"[compare] {a.a} vs {a.b}: {dict(cnt)}; absent from b: {len(absent)}")
    if feat_ch:
        print(f"[compare] feat max|diff| {max_feat:.6g}; rows differing per channel {dict(feat_ch)}")
    if scal:
        print(f"[compare] scalar fields max|diff| {scal}")
        print(f"[compare] scalar fields NaN side on differing rows { {k: dict(v) for k, v in nan_side.items()} }")
    for s, d in bad[:20]:
        print(f"[compare] DIFF {s}: {d}")
    for s in absent[:10]:
        print(f"[compare] ABSENT {s}")
    ok = not bad and n >= max(a.min_rows, 1) and not (a.require_all and absent) and not dropped
    print(f"[compare] {'PASS' if ok else 'FAIL'}: {n} rows compared (min {a.min_rows}), {len(bad)} differ, "
          f"{len(absent)} absent{' (required present)' if a.require_all else ''}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
