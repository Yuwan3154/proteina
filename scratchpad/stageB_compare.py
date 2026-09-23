"""Stage B comparison (Directive B): structure quality from tri-sampled maps, ConFind pipeline vs CB-8 pipeline.

Each input is one stageB_c2c_from_maps.py JSONL (n_seeds 1, one row per chain), passed as LABEL=path. Rows
are joined on the chain id and must agree on the sequence fingerprint. Reported on the pre-registered chain
sets (primary 191, sequence-clean 144, novel-fold): per arm the TM distribution (USalign, native-normalised),
the count at TM >= 0.5 (the standard same-fold threshold), proper RMSD, dist_mae and the mirror rate; then
every requested pair of arms paired per chain (Wilcoxon signed-rank on TM, exact McNemar on refl-sign).
⛔ Arms from different contact definitions ARE comparable here -- that is the point of Stage B -- but each
c2c's own native-map ceiling should be read beside it.

Usage: python scratchpad/stageB_compare.py <lists_dir> LABEL=path.jsonl ... [--pairs A:B,C:D]
"""

import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

SETS = (("primary 191 (max384 val, 1/cluster)", "stageA_val_1percluster.txt"),
        ("sequence-clean 144", "stageA_primary_seqclean.txt"),
        ("novel-fold", "stageA_novel.txt"))

lists = sys.argv[1]
args = [a for a in sys.argv[2:] if not a.startswith("--pairs")]
pairs = [p.split(":") for a in sys.argv[2:] if a.startswith("--pairs") for p in a.split("=", 1)[1].split(",")]
arms = {}
for a in args:
    label, path = a.split("=", 1)
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    by = {r["stem"]: r for r in rows}
    assert len(by) == len(rows), f"{label}: more than one row per chain (use n_seeds 1)"
    arms[label] = by
    r0 = rows[0]
    print(f"[arm] {label}: {len(by)} chains  ckpt step {r0['global_step']}  weights {r0['weights']}  "
          f"map {r0['map_source']}  def {r0['contact_def']}")
for a, b in pairs:
    for s in set(arms[a]) & set(arms[b]):
        assert arms[a][s]["seq_fp"] == arms[b][s]["seq_fp"], f"{s}: sequence differs between {a} and {b}"


def col(label, chains, k):
    return np.array([arms[label][s][k] for s in chains], dtype=float)


for name, f in SETS:
    chains = [ln.strip() for ln in open(os.path.join(lists, f)) if ln.strip()]
    chains = [s for s in chains if all(s in arms[lab] for lab in arms)]
    print(f"\n==== {name}: n={len(chains)} (present in every arm) ====")
    print(f"{'arm':28s} {'TM q25':>7s} {'median':>7s} {'q75':>7s} {'TM>=0.5':>8s} {'RMSDp med':>9s} "
          f"{'distMAE med':>11s} {'mirror':>7s}")
    for lab in arms:
        tm = col(lab, chains, "tm")
        q = np.percentile(tm, [25, 50, 75])
        print(f"{lab:28s} {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f} {int((tm >= 0.5).sum()):8d} "
              f"{np.median(col(lab, chains, 'rmsd_proper')):9.2f} {np.median(col(lab, chains, 'dist_mae')):11.2f} "
              f"{col(lab, chains, 'refl_sign').mean():7.3f}")
    for a, b in pairs:
        ta, tb = col(a, chains, "tm"), col(b, chains, "tm")
        d = tb - ta
        pw = wilcoxon(ta, tb).pvalue if np.any(d != 0) else 1.0
        ra, rb = col(a, chains, "refl_sign"), col(b, chains, "refl_sign")
        n01, n10 = int(((ra == 0) & (rb == 1)).sum()), int(((ra == 1) & (rb == 0)).sum())
        pm = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        print(f"  {a} -> {b}: TM median d {np.median(d):+.3f} mean d {np.mean(d):+.3f}  "
              f"b better {int((d > 0).sum())} / worse {int((d < 0).sum())}  Wilcoxon p={pw:.4g}; "
              f"mirror only-b {n01} only-a {n10} McNemar p={pm:.4g}")
