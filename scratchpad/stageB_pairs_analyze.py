"""T7 / Stage B analysis (user 2026-09-29/30): CB-8 vs ConFind on the SAME (query, template) pairs, and how well
structure quality tracks the sampled contact map's P@L.

Inputs are stageB_c2c_from_maps.py JSONLs: per definition one tri-map arm (k samples per query, each row carrying the
tri sample's metrics as tri_* and its template as ref_id) and optionally the native-map arm (its ceiling).

Reported per pre-registered set (primary 191, sequence-clean 144, novel-fold):
  1. per definition: TM quartiles over samples, TM >= 0.5 count, median proper RMSD, mirror rate, per-query mean TM;
     native ceiling beside it when given;
  2. CB-8 vs ConFind, one value per query (mean over its k samples): Wilcoxon on TM, better / worse counts;
  3. structure vs map quality, per definition: Spearman(TM, x) and Spearman(RMSD, x) for x in the tri metrics --
     POOLED over all samples, BETWEEN queries (per-query means), and WITHIN query (distribution of per-query rho over
     queries whose k maps have >= 3 distinct x values; the count is reported). Pooled rho mixes target difficulty into
     the answer, so the within-query distribution is the one that says whether a better map gives a better structure.

Usage: python scratchpad/stageB_pairs_analyze.py <lists_dir> CB8_TRI=a.jsonl CONFIND_TRI=b.jsonl
          [CB8_NATIVE=c.jsonl] [CONFIND_NATIVE=d.jsonl] [--k 8] [--out merged.tsv]
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, wilcoxon

SETS = (("primary 191 (max384 val, 1/cluster)", "stageA_val_1percluster.txt"),
        ("sequence-clean 144", "stageA_primary_seqclean.txt"),
        ("novel-fold", "stageA_novel.txt"))
# thresholded (> 0.5) precision / recall / F1 are what the c2c actually sees; P@L ranks by probability
XS = ("tri_contact_precision_at_L", "tri_contact_precision_at_L5", "tri_contact_long_range_precision_at_L5",
      "tri_contact_precision", "tri_contact_recall", "tri_contact_f1", "density_ratio")
DEFS = ("CB8", "CONFIND")


def load(path, tri):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    for r in rows:
        assert r["map_source"] == ("tri" if tri else "native"), f"{path}: map_source {r['map_source']}"
        assert r["weights"] == "ema", f"{path}: weights {r['weights']}"
        if tri and r.get("density_native"):
            r["density_ratio"] = r["density_tri"] / r["density_native"]
    return rows


def main():
    lists = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a and not a.startswith("--"))
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 8
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None

    tri, nat = {}, {}
    for d in DEFS:
        rows = load(kv[f"{d}_TRI"], tri=True)
        by = defaultdict(list)
        for r in rows:
            by[r["stem"]].append(r)
        for st, rs in by.items():
            idx = sorted(r["sample_index"] for r in rs)
            assert idx == list(range(k)), f"{d} {st}: sample indices {idx}, expected 0..{k - 1}"
            refs = {r["ref_id"] for r in rs}
            assert len(refs) == 1 and next(iter(refs)), f"{d} {st}: templates {refs}"
        tri[d] = by
        if f"{d}_NATIVE" in kv:
            nat[d] = {r["stem"]: r for r in load(kv[f"{d}_NATIVE"], tri=False)}
        print(f"[arm] {d}: {len(rows)} tri rows over {len(by)} queries, {k} samples each; contact_def "
              f"{rows[0]['contact_def']}; c2c step {rows[0]['global_step']}"
              + (f"; native ceiling {len(nat[d])} rows" if d in nat else ""))

    common = sorted(set(tri["CB8"]) & set(tri["CONFIND"]))
    for st in common:
        a, b = tri["CB8"][st][0], tri["CONFIND"][st][0]
        assert a["seq_fp"] == b["seq_fp"], f"{st}: sequence differs between definitions"
        assert a["ref_id"] == b["ref_id"], f"{st}: template {a['ref_id']} (CB-8) != {b['ref_id']} (ConFind)"
    print(f"[pairs] {len(common)} queries present in both, SAME template in both (asserted)")

    if out:
        with open(out, "w") as fh:
            cols = ["def", "stem", "ref_id", "sample_index", "tm", "rmsd_proper", "refl_sign", *XS]
            fh.write("\t".join(cols) + "\n")
            for d in DEFS:
                for st in common:
                    for r in sorted(tri[d][st], key=lambda r: r["sample_index"]):
                        fh.write("\t".join(str(r.get(c) if c != "def" else d) for c in cols) + "\n")
        print(f"[out] per-sample table -> {out}")

    for name, f in SETS:
        chains = [ln.strip() for ln in open(os.path.join(lists, f)) if ln.strip()]
        chains = [s for s in chains if s in common]
        print(f"\n==== {name}: {len(chains)} queries x {k} samples ====")
        print(f"{'arm':22s} {'TM q25':>7s} {'median':>7s} {'q75':>7s} {'TM>=0.5':>9s} {'RMSDp med':>9s} {'mirror':>7s}"
              f" {'query-mean TM med':>18s}")
        qmean = {}
        for d in DEFS:
            tm = np.array([r["tm"] for s in chains for r in tri[d][s]])
            rm = np.array([r["rmsd_proper"] for s in chains for r in tri[d][s]])
            mi = np.array([r["refl_sign"] for s in chains for r in tri[d][s]])
            qmean[d] = np.array([np.mean([r["tm"] for r in tri[d][s]]) for s in chains])
            q = np.percentile(tm, [25, 50, 75])
            print(f"{d + ' tri':22s} {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f} {int((tm >= 0.5).sum()):5d}/{len(tm):<4d}"
                  f"{np.median(rm):9.2f} {mi.mean():7.3f} {np.median(qmean[d]):18.3f}")
            if d in nat:
                nt = np.array([nat[d][s]["tm"] for s in chains if s in nat[d]])
                nq = np.percentile(nt, [25, 50, 75])
                print(f"{d + ' native (ceiling)':22s} {nq[0]:7.3f} {nq[1]:7.3f} {nq[2]:7.3f} "
                      f"{int((nt >= 0.5).sum()):5d}/{len(nt):<4d}")
        dq = qmean["CONFIND"] - qmean["CB8"]
        p = wilcoxon(qmean["CB8"], qmean["CONFIND"]).pvalue if np.any(dq != 0) else 1.0
        print(f"  CB8 -> CONFIND (per-query mean TM): median d {np.median(dq):+.3f}  ConFind better "
              f"{int((dq > 0).sum())} / worse {int((dq < 0).sum())}  Wilcoxon p={p:.3g}")

        for d in DEFS:
            print(f"  -- {d}: structure vs sampled-map quality (Spearman rho) --")
            print(f"     {'x':40s} {'pooled TM':>10s} {'between-q TM':>13s} {'within-q TM median [IQR] (n, frac>0)':>40s}"
                  f" {'pooled RMSD':>12s}")
            for x in XS:
                pts = [(r["tm"], r["rmsd_proper"], r.get(x)) for s in chains for r in tri[d][s]]
                pts = [t for t in pts if t[2] is not None]
                if len(pts) < 3:
                    continue
                tm, rm, xv = map(np.array, zip(*pts))
                if len(np.unique(xv)) < 2:
                    print(f"     {x:40s} constant across all samples ({xv[0]!r}) -- no correlation defined")
                    continue
                rp = spearmanr(xv, tm).statistic
                rr = spearmanr(xv, rm).statistic
                mq = [(np.mean([r["tm"] for r in tri[d][s]]), np.mean([r[x] for r in tri[d][s]])) for s in chains]
                rb = spearmanr(*zip(*mq)).statistic
                within = []
                for s in chains:
                    xs = np.array([r[x] for r in tri[d][s]])
                    ts = np.array([r["tm"] for r in tri[d][s]])
                    if len(np.unique(xs)) >= 3 and len(np.unique(ts)) >= 2:
                        within.append(spearmanr(xs, ts).statistic)
                w = np.array(within)
                wtxt = (f"{np.median(w):+.2f} [{np.percentile(w, 25):+.2f},{np.percentile(w, 75):+.2f}] "
                        f"(n={len(w)}, {np.mean(w > 0):.2f})" if len(w) else "n/a (no query with spread)")
                print(f"     {x:40s} {rp:+10.3f} {rb:+13.3f} {wtxt:>40s} {rr:+12.3f}")


if __name__ == "__main__":
    main()
