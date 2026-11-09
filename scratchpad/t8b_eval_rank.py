"""T8b eval: WITHIN-TARGET ranking quality of the ContactEBM pick (user 2026-11-08 "Do analysis").

Per chain, over its 24-pool (8 T8b samples x h_bond / tmalign / tmalign_smooth template SSE; c2c TM from stageB, clean
ContactEBM energy at t_eval 0.25 from cebm.jsonl e25_pool, lower = better):
  rho24 / rho8    Spearman(-energy, TM) over the 24 pool / the 8 h_bond samples (average ranks for ties)
  spread24        TM standard deviation over the 24 (a pool without spread cannot be ranked)
  pick_pct        fraction of the OTHER 23 with TM strictly below the pick's TM (1 = the pick is the unique best)
  regret          best24 - TM(pick24);  random_regret = best24 - mean24 (what a random pick loses in expectation)
  captured        (TM(pick24) - mean24) / (best24 - mean24): share of the possible gain the pick realises
Reports distributions (quartiles), counts, both sets (clean = gt_mismatch 0; all), regret vs spread / length / classifiers,
and the largest misses. Sources as t8b_eval_select.py (run t8b_eval_score.py first: it holds the provenance asserts).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

MODES = ("h_bond", "tmalign", "tmalign_smooth")


def q(x):
    if len(x) == 0:
        return "n/a (all undefined)"
    return f"median {np.median(x):+.3f} [IQR {np.percentile(x, 25):+.3f}, {np.percentile(x, 75):+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", default="/orcd/scratch/orcd/011/chenxiou/t8b_eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chains = sorted(l.strip() for l in open(os.path.join(a.e, "pool", "chains_t7.txt")) if l.strip())
    assert len(chains) == 98, len(chains)
    tm, L, mism = defaultdict(dict), {}, defaultdict(int)
    for m in MODES:
        for k in range(4):
            for l in open(os.path.join(a.e, "stageB", f"t8b_t7_{m}_c{k}.jsonl")):
                if l.strip():
                    r = json.loads(l)
                    key = (m, int(r["sample_index"]))
                    assert key not in tm[r["stem"]], (r["stem"], key)
                    tm[r["stem"]][key] = float(r["tm"])
                    L[r["stem"]] = int(r["L"])
                    mism[r["stem"]] += int(r["gt_mismatch"])
    meta = {}
    for k in range(4):
        for l in open(os.path.join(a.e, "dumps", f"t8b_t7_cebm_c{k}", "cebm.jsonl")):
            if l.strip():
                r = json.loads(l)
                assert r["stem"] not in meta, r["stem"]
                meta[r["stem"]] = r
    rows = {}
    for st in chains:
        mt = meta[st]
        labels, e = mt["pool_labels"], np.array(mt["e25_pool"])
        assert len(labels) == 24 and len(set(labels)) == 24 and len(tm[st]) == 24
        t = np.array([tm[st][(lab.split("#")[0], int(lab.split("#")[1]))] for lab in labels])
        h = np.array([lab.startswith("h_bond#") for lab in labels])
        p = int(np.argmin(e))
        assert p == mt["pick_idx"]
        best = int(np.argmax(t))
        rows[st] = {
            "L": L[st], "rho24": spearmanr(-e, t)[0], "rho8": spearmanr(-e[h], t[h])[0], "spread24": t.std(),
            "pick_pct": float((t < t[p]).sum() / (len(t) - 1)), "n_above": int((t > t[p]).sum()), "pick": t[p], "best": t[best], "mean": t.mean(),
            "regret": t[best] - t[p], "random_regret": t[best] - t.mean(),
            "captured": (t[p] - t.mean()) / (t[best] - t.mean()) if t[best] > t.mean() else np.nan,
            "pick_mode": labels[p].split("#")[0], "best_mode": labels[best].split("#")[0],
            "mode_means": {m: t[[lab.startswith(m + "#") for lab in labels]].mean() for m in MODES},
        }
    out = []
    for name, S in (("clean", [st for st in chains if mism[st] == 0]), ("all", chains)):
        R = [rows[st] for st in S]
        v = lambda k: np.array([r[k] for r in R], dtype=float)
        rho24, rho8 = v("rho24"), v("rho8")
        out.append(f"## set {name}: {len(S)} chains")
        out.append(f"rho24 (Spearman -E vs TM, 24 pool): {q(rho24[~np.isnan(rho24)])}; > 0 on {int((rho24 > 0).sum())}, "
                   f"> 0.5 on {int((rho24 > 0.5).sum())}, < 0 on {int((rho24 < 0).sum())}, undefined {int(np.isnan(rho24).sum())}")
        out.append(f"rho8  (h_bond 8 only):              {q(rho8[~np.isnan(rho8)])}; > 0 on {int((rho8 > 0).sum())}, "
                   f"< 0 on {int((rho8 < 0).sum())}, undefined {int(np.isnan(rho8).sum())}")
        pp = v("pick_pct")
        out.append(f"pick position in the 24 (fraction below it): {q(pp)}; pick = best on {int((v('n_above') == 0).sum())}, "
                   f"top 3 on {int((v('n_above') < 3).sum())} (ties favour the pick), fewer than half of the other 23 below it on "
                   f"{int((pp < 0.5).sum())}")
        reg, rreg, cap = v("regret"), v("random_regret"), v("captured")
        out.append(f"regret (best24 - pick): mean {reg.mean():.3f}, {q(reg)}; random-pick regret mean {rreg.mean():.3f} "
                   f"-> the pick removes {1 - reg.mean() / rreg.mean():.0%} of the random regret")
        out.append(f"captured share of (best - mean): {q(cap[~np.isnan(cap)])}")
        sp = v("spread24")
        lo, hi = np.percentile(sp, [33.3, 66.7])
        for lab, msk in (("low", sp <= lo), ("mid", (sp > lo) & (sp <= hi)), ("high", sp > hi)):
            if not msk.any():
                continue
            out.append(f"  spread24 tercile {lab:<4} (n {int(msk.sum()):>2}, spread {sp[msk].min():.3f}-{sp[msk].max():.3f}): "
                       f"rho24 median {np.nanmedian(rho24[msk]):+.2f}, regret mean {reg[msk].mean():.3f}, "
                       f"random regret {rreg[msk].mean():.3f}")
        Ls = v("L")
        for lab, msk in (("L<=150", Ls <= 150), ("150<L<=250", (Ls > 150) & (Ls <= 250)), ("L>250", Ls > 250)):
            if msk.any():
                out.append(f"  {lab:<11} (n {int(msk.sum()):>2}): rho24 median {np.nanmedian(rho24[msk]):+.2f}, "
                           f"regret mean {reg[msk].mean():.3f}, pick {v('pick')[msk].mean():.3f}, best {v('best')[msk].mean():.3f}")
        out.append(f"  Spearman over chains: regret vs spread24 {spearmanr(reg, sp)[0]:+.2f}, regret vs L {spearmanr(reg, Ls)[0]:+.2f}, "
                   f"rho24 vs spread24 {spearmanr(rho24, sp, nan_policy='omit')[0]:+.2f}")
        pm = defaultdict(int)
        bm = defaultdict(int)
        for r in R:
            pm[r["pick_mode"]] += 1
            bm[r["best_mode"]] += 1
        out.append("  pick classifier: " + ", ".join(f"{m} {pm[m]}" for m in MODES)
                   + " | best-of-24 classifier: " + ", ".join(f"{m} {bm[m]}" for m in MODES))
        mm = {m: np.array([r["mode_means"][m] for r in R]) for m in MODES}
        bestmode = np.array([max(MODES, key=lambda m: r["mode_means"][m]) for r in R])
        out.append("  classifier with the highest mean-of-8 per chain: "
                   + ", ".join(f"{m} {int((bestmode == m).sum())}" for m in MODES)
                   + f"; oracle per-chain classifier choice mean {np.max(np.stack([mm[m] for m in MODES]), 0).mean():.3f} "
                   + f"vs h_bond {mm['h_bond'].mean():.3f} vs ContactEBM pick24 {v('pick').mean():.3f}")
        out.append("  largest misses (regret): stem L spread rho24 pick best pick_mode best_mode changed")
        for st in sorted(S, key=lambda s: -rows[s]["regret"])[:10]:
            r = rows[st]
            out.append(f"    {st:<8} {r['L']:>4} {r['spread24']:.3f} {r['rho24']:+.2f} {r['pick']:.3f} {r['best']:.3f} "
                       f"{r['pick_mode']:<15} {r['best_mode']:<15} {'yes' if mism[st] else 'no'}")
        out.append("")
    text = "\n".join(out)
    print(text)
    open(a.out, "w").write(text + "\n")
    with open(os.path.splitext(a.out)[0] + "_per_chain.tsv", "w") as fh:
        keys = ["L", "rho24", "rho8", "spread24", "pick_pct", "pick", "best", "mean", "regret", "random_regret", "captured",
                "pick_mode", "best_mode"]
        fh.write("stem\tchanged\t" + "\t".join(keys) + "\n")
        for st in chains:
            fh.write(f"{st}\t{int(mism[st] > 0)}\t" + "\t".join(str(rows[st][k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
