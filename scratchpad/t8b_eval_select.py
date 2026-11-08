"""T8b eval, selection-gap breakdown of arm (b) (sse_ens24 + ContactEBM): how much of its TM comes from SELECTING among
samples vs from the extra SSE classifiers vs from the refinement, against the oracle ceilings.

Per chain, from the c2c rows of the three 8-sample T8b arms and t8b_eval_cebm.py's cebm.jsonl (e25_pool + pool_labels =
the clean ContactEBM energy at t_eval 0.25 of every pool map, in pool order):
  mean8_hbond     arm (a): mean TM of the 8 production-SSE samples (= a random single sample, in expectation)
  pick8_hbond     the lowest-energy map among the 8 h_bond samples only (ContactEBM selection, no extra classifiers)
  pick24          the lowest-energy map of the 24-pool (= the sample cebm_pick folded; consistency-checked below)
  minE            arm (b): pick24 refined (cebm_minE row)
  best8_hbond / best24   oracle: the best TM among the 8 h_bond / all 24 samples
Statistics as t8b_eval_score.py (per chain, paired, 95% percentile bootstrap over chains, N_BOOT 2000, default_rng(0)).
Sets: clean (gt_mismatch 0) and all.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

N_BOOT = 2000
MODES = ("h_bond", "tmalign", "tmalign_smooth")


def boot_ci(x, rng):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", default="/orcd/scratch/orcd/011/chenxiou/t8b_eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chains = sorted(l.strip() for l in open(os.path.join(a.e, "pool", "chains_t7.txt")) if l.strip())
    assert len(chains) == 98, len(chains)
    tm = defaultdict(dict)            # (mode, sample_index) per stem -> TM
    mism = defaultdict(int)
    for m in MODES:
        for k in range(4):
            for l in open(os.path.join(a.e, "stageB", f"t8b_t7_{m}_c{k}.jsonl")):
                if l.strip():
                    r = json.loads(l)
                    key = (m, int(r["sample_index"]))
                    assert key not in tm[r["stem"]], (r["stem"], key)
                    tm[r["stem"]][key] = float(r["tm"])
                    mism[r["stem"]] += int(r["gt_mismatch"])
    cebm_tm, meta = defaultdict(dict), {}
    for k in range(4):
        for l in open(os.path.join(a.e, "stageB", f"t8b_t7_cebm_c{k}.jsonl")):
            if l.strip():
                r = json.loads(l)
                assert int(r["sample_index"]) not in cebm_tm[r["stem"]], (r["stem"], r["sample_index"])
                cebm_tm[r["stem"]][int(r["sample_index"])] = float(r["tm"])
                mism[r["stem"]] += int(r["gt_mismatch"])   # same clean-set definition as t8b_eval_score.py
        for l in open(os.path.join(a.e, "dumps", f"t8b_t7_cebm_c{k}", "cebm.jsonl")):
            if l.strip():
                r = json.loads(l)
                assert r["stem"] not in meta, r["stem"]
                meta[r["stem"]] = r
    rows = {}
    n_pick_mismatch = []
    n_tie8 = 0
    for st in chains:
        t, mt = tm[st], meta[st]
        assert len(t) == 24 and sorted(cebm_tm[st]) == [0, 1, 2], (st, len(t), sorted(cebm_tm[st]))
        labels = mt["pool_labels"]
        e = np.array(mt["e25_pool"])
        assert len(labels) == 24 and len(e) == 24
        pool_tm = np.array([t[(lab.split("#")[0], int(lab.split("#")[1]))] for lab in labels])
        h = [i for i, lab in enumerate(labels) if lab.startswith("h_bond#")]
        assert len(h) == 8
        p24 = int(np.argmin(e))
        assert p24 == mt["pick_idx"], (st, p24, mt["pick_idx"])
        p8 = h[int(np.argmin(e[h]))]
        n_tie8 += int((e[h] == e[h].min()).sum() > 1)   # e25_pool is rounded to 4 decimals
        # cebm_pick folded the same map with the same c2c seed: its TM should equal the pool row's (GPU nondeterminism aside)
        n_pick_mismatch.append(abs(cebm_tm[st][1] - pool_tm[p24]))
        rows[st] = {"mean8_hbond": pool_tm[h].mean(), "pick8_hbond": pool_tm[p8], "pick24": pool_tm[p24],
                    "minE": cebm_tm[st][0], "best8_hbond": pool_tm[h].max(), "best24": pool_tm.max(),
                    "mode": labels[p24].split("#")[0]}
    out = [f"consistency: |TM(cebm_pick row) - TM(pool row of the pick)| max {max(n_pick_mismatch):.4f}, "
           f"mean {np.mean(n_pick_mismatch):.4f} over {len(chains)} chains (same map + seed -> ~0 expected)",
           f"h_bond-only pick: {n_tie8} of {len(chains)} chains have a tied minimum energy (rounded values)",
           "mean8_hbond equals t8b_eval_score.py t8b_hbond; its CI differs slightly (RNG draws consumed in a different order)", ""]
    clean = [st for st in chains if mism[st] == 0]
    cols = ["mean8_hbond", "pick8_hbond", "pick24", "minE", "best8_hbond", "best24"]
    pairs = [("pick8_hbond", "mean8_hbond"), ("pick24", "pick8_hbond"), ("minE", "pick24"), ("minE", "mean8_hbond"),
             ("best8_hbond", "pick8_hbond"), ("best24", "pick24")]
    for name, S in (("clean", clean), ("all", chains)):
        rng = np.random.default_rng(0)
        out.append(f"## set {name}: {len(S)} chains")
        out.append(f"{'quantity':<12} {'chains>=0.5':>11}   mean TM [95% CI]")
        v = {c: np.array([rows[st][c] for st in S]) for c in cols}
        for c in cols:
            lo, hi = boot_ci(v[c], rng)
            out.append(f"{c:<12} {int((v[c] >= 0.5).sum()):>11}   {v[c].mean():.3f} [{lo:.3f}, {hi:.3f}]")
        out.append("paired B - A   mean [95% CI]   chains B>A / ties / n")
        for b_, a_ in pairs:
            d = v[b_] - v[a_]
            lo, hi = boot_ci(d, rng)
            out.append(f"{b_:>12} - {a_:<12} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   "
                       f"{int((d > 0).sum())}/{int((d == 0).sum())}/{len(d)}")
        modes = defaultdict(int)
        for st in S:
            modes[rows[st]["mode"]] += 1
        out.append("pick24 source classifier: " + ", ".join(f"{m} {modes[m]}" for m in MODES))
        out.append("")
    text = "\n".join(out)
    print(text)
    open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
