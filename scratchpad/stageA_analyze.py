"""Stage A analysis (Directive B): does the old ConFind tri model need its two topology-reference CA features?

Reads each pass's samples.jsonl (one sample per chain, seeded identically across passes) and reports, on
the pre-registered chain sets (primary 191 max384-val clusters, the 144 sequence-clean subset, the novel
fold chains):
  * per-pass median/mean of the per-chain contact metrics;
  * OLD model FULL vs ZERO-CA, paired per chain, per arm (Wilcoxon signed-rank on P@L);
  * OLD ZERO-CA non-self vs the SAME model's own no-topology floor (the user judges "broken" from this);
  * each model's lift over its OWN no-reference convention (old: single MASK token; new: variable-length
    masked reference -- the user's 2026-09-22 caveat), plus the new model's single-token floor for contrast.
Control: the OLD mask arm feeds all-zero pair features, so FULL and ZERO-CA must agree EXACTLY there.
⛔ OLD and NEW P@L are scored on different contact definitions (ConFind vs CB-8 A) and are never compared
directly here; the structure-level comparison is Stage B.

Usage: python scratchpad/stageA_analyze.py <stageA_dir> <lists_dir>
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

PASSES = ("old_full_self", "old_full_nonself", "old_full_mask", "old_zero_self", "old_zero_nonself",
          "old_zero_mask", "new_self", "new_nonself", "new_mask_variable", "new_mask_single")
METRICS = ("contact_precision_at_L", "contact_precision_at_L5", "contact_long_range_precision_at_L5")
SETS = (("primary 191 (max384 val, 1/cluster)", "stageA_val_1percluster.txt"),
        ("sequence-clean 144", "stageA_primary_seqclean.txt"),
        ("novel-fold", "stageA_novel.txt"))


def load_pass(d):
    rows = [json.loads(ln) for ln in open(os.path.join(d, "samples.jsonl")) if ln.strip()]
    by = {r["stem"]: r["metrics"] for r in rows}
    assert len(by) == len(rows), f"{d}: a chain was sampled more than once"
    return by


root, lists = sys.argv[1], sys.argv[2]
union = [ln.strip() for ln in open(os.path.join(lists, "stageA_run_union.txt")) if ln.strip()]
P = {}
for p in PASSES:
    d = os.path.join(root, p)
    if not os.path.isfile(os.path.join(d, "samples.jsonl")):
        print(f"[missing] {p}")
        continue
    P[p] = load_pass(d)
    assert sorted(P[p]) == sorted(union), f"{p}: sampled chains != run list ({len(P[p])} vs {len(union)})"
print(f"[passes] {len(P)} of {len(PASSES)} present, each covering all {len(union)} listed chains")

if "old_full_mask" in P and "old_zero_mask" in P:
    dmax = max(abs(P["old_full_mask"][s][m] - P["old_zero_mask"][s][m]) for s in union for m in METRICS
               if m in P["old_full_mask"][s])
    print(f"[control] old mask arm, FULL vs ZERO-CA, max |diff| over {len(union)} chains x metrics = {dmax:.3g} "
          f"({'PASS: identical' if dmax == 0 else 'FAIL: the ablation touched a path it should not'})")


def vals(p, chains, m="contact_precision_at_L"):
    return np.array([P[p][s].get(m, np.nan) for s in chains], dtype=float)


def paired(a, b, chains, m="contact_precision_at_L"):
    x, y = vals(a, chains, m), vals(b, chains, m)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    d = y - x
    p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return int(ok.sum()), float(np.median(d)), float(np.mean(d)), int((d < 0).sum()), int((d > 0).sum()), p


for name, f in SETS:
    chains = [ln.strip() for ln in open(os.path.join(lists, f)) if ln.strip()]
    print(f"\n==== {name}: n={len(chains)} ====")
    print(f"{'pass':20s} " + " ".join(f"{m.replace('contact_', '')[:24]:>26s}" for m in METRICS))
    for p in PASSES:
        if p in P:
            cells = []
            for m in METRICS:
                v = vals(p, chains, m)
                cells.append(f"{np.nanmedian(v):.3f} (mean {np.nanmean(v):.3f})")
            print(f"{p:20s} " + " ".join(f"{c:>26s}" for c in cells))
    print("paired, P@L (b - a):   n   median d   mean d   b<a   b>a   Wilcoxon p")
    for a, b in (("old_full_self", "old_zero_self"), ("old_full_nonself", "old_zero_nonself"),
                 ("old_full_mask", "old_zero_nonself"),
                 ("new_mask_variable", "new_nonself"), ("new_mask_single", "new_mask_variable")):
        if a in P and b in P:
            n, md, mn, lo, hi, p = paired(a, b, chains)
            print(f"  {a} -> {b}: {n:4d} {md:+9.3f} {mn:+8.3f} {lo:5d} {hi:5d}   {p:.4g}")
