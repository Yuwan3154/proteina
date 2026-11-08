"""T8b eval scoring on the T7-NEWEST pairs (user 2026-11-07 "Evaluate T8b" + 2026-11-08 "Both: 70 primary + 98").

Arms (c2c = ConFind twin 29,000 for T8b and ConFind, tbeta 29,620 for CB-8; TM native-normalised vs the OLD pdb_train
native, as T7):
  cb8          CB-8 TopoRef 141,030, 8 samples per chain (T7-NEWEST rows; chunk 0 re-run without 1ffk_G)
  cf           ConFind FT TopoRef 60,505, 8 samples (same)
  t8b_hbond    T8b final (step 450,000), production template SSE, 8 samples  = arm (a)
  t8b_tmalign / t8b_smooth   same, template SSE by TMalign / smoothed TMalign, 8 samples each
  t8b_pool24   mean over the 24 maps of the three classifiers
  cebm_minE    arm (b): sse_ens24 + ContactEBM pick + reverse refinement minE, ONE structure per chain
  cebm_pick / cebm_final   the unrefined ContactEBM pick / the refinement's final map (diagnostics)
Sets: clean = chains whose T8b rows all have gt_mismatch 0 (control (b) passed; primary, 70 expected), all = every chain.
Statistics: per-chain mean over samples, then mean over chains; paired differences per chain; 95% percentile bootstrap
over chains, N_BOOT 2000, default_rng(0) (as xrun_score.py / ProteinEBMalign synth_compare.py @89ea726).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

N_BOOT = 2000
CEBM_TAGS = {0: "cebm_minE", 1: "cebm_pick", 2: "cebm_final"}   # t8b_eval_cebm.py sample_index -> output


def boot_ci(x, rng):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def read(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    assert rows, f"no rows in {paths}"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", default="/orcd/scratch/orcd/011/chenxiou/t8b_eval")
    ap.add_argument("--t7", default="/orcd/scratch/orcd/011/chenxiou/t7/x141030_out")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sb = os.path.join(a.e, "stageB")
    chains = sorted(l.strip() for l in open(os.path.join(a.e, "pool", "chains_t7.txt")) if l.strip())
    assert len(chains) == 98, len(chains)
    C2C_STEP = {"cb8": 29620, "cf": 29000, "t8b_hbond": 29000, "t8b_tmalign": 29000, "t8b_smooth": 29000, "cebm": 29000}
    src = {
        "cb8": [f"{sb}/base_cb8_141030_c0_no1ffk.jsonl"] + [f"{a.t7}/B_t7qx_cb8_c0{k}.jsonl" for k in (1, 2, 3)],
        "cf": [f"{sb}/base_cf60505_c0_no1ffk.jsonl"] + [f"{a.t7}/B_t7qx_cf29000_c0{k}.jsonl" for k in (1, 2, 3)],
        "t8b_hbond": [f"{sb}/t8b_t7_h_bond_c{k}.jsonl" for k in range(4)],
        "t8b_tmalign": [f"{sb}/t8b_t7_tmalign_c{k}.jsonl" for k in range(4)],
        "t8b_smooth": [f"{sb}/t8b_t7_tmalign_smooth_c{k}.jsonl" for k in range(4)],
        "cebm": [f"{sb}/t8b_t7_cebm_c{k}.jsonl" for k in range(4)],
    }
    tm = defaultdict(dict)            # arm -> stem -> [tm]
    ref, seed, mism = {}, defaultdict(set), defaultdict(int)
    prov = defaultdict(set)           # arm -> {(c2c ckpt, global_step, steps, sigma_data)}
    sidx = defaultdict(list)          # (name, stem) -> sample indices
    for arm, paths in src.items():
        for r in read(paths):
            st = r["stem"]
            if st not in chains:
                assert arm in ("cb8", "cf") and st == "1ffk_G", f"{arm}: unexpected chain {st}"
                continue
            assert r["ref_id"], f"{arm} {st}: no ref_id"
            assert int(r["global_step"]) == C2C_STEP[arm], f"{arm}: c2c global_step {r['global_step']}"
            prov[arm].add((r["ckpt"], r["global_step"], r["steps"], r["sigma_data"]))
            assert ref.setdefault(st, r["ref_id"]) == r["ref_id"], f"{arm} {st}: ref_id {r['ref_id']} != {ref[st]}"
            seed[st].add(r["torch_seed"])
            if arm.startswith("t8b") or arm == "cebm":
                mism[st] += int(r["gt_mismatch"])
            name = CEBM_TAGS[r["sample_index"]] if arm == "cebm" else arm
            tm[name].setdefault(st, []).append(float(r["tm"]))
            sidx[(name, st)].append(int(r["sample_index"]) if arm != "cebm" else 0)
    for st in chains:
        assert len(seed[st]) == 1, f"{st}: c2c seeds differ across arms {seed[st]} (pairing broken)"
    for name, d in tm.items():
        want = 1 if name.startswith("cebm") else 8
        bad = [st for st in chains if len(d.get(st, [])) != want or sorted(sidx[(name, st)]) != list(range(want))]
        assert not bad, f"{name}: chains without exactly {want} rows: {bad[:10]}"
    tm["t8b_pool24"] = {st: tm["t8b_hbond"][st] + tm["t8b_tmalign"][st] + tm["t8b_smooth"][st] for st in chains}
    clean = [st for st in chains if mism[st] == 0]
    arms = ["cb8", "cf", "t8b_hbond", "t8b_tmalign", "t8b_smooth", "t8b_pool24", "cebm_minE", "cebm_pick", "cebm_final"]
    pairs = [("cf", "cb8"), ("t8b_hbond", "cf"), ("t8b_hbond", "cb8"), ("cebm_minE", "t8b_hbond"), ("cebm_minE", "cf"),
             ("cebm_minE", "cebm_pick"), ("t8b_tmalign", "t8b_hbond"), ("t8b_smooth", "t8b_hbond"),
             ("t8b_pool24", "t8b_hbond")]
    for arm, pv in prov.items():
        assert len(pv) == 1, f"{arm}: rows from several c2c runs {pv}"
    out = [f"c2c per arm: {arm} {pv}" for arm, pv in sorted(prov.items())]
    out += ["tri: CB-8 141,030 / ConFind 60,505 (T7 x141030 rows) / T8b last-EMA step 450,000 (t8b_eval dumps)", ""]
    for set_name, S in (("clean", clean), ("all", chains)):
        rng = np.random.default_rng(0)
        out.append(f"## set {set_name}: {len(S)} chains (gt_mismatch 0: {len(clean)} of {len(chains)})")
        out.append(f"{'arm':<12} {'samples':>8} {'TM>=0.5':>9} {'chains mean>=0.5':>17}   mean TM [95% CI]")
        means = {}
        for arm in arms:
            m = np.array([np.mean(tm[arm][st]) for st in S])
            means[arm] = m
            allv = np.concatenate([tm[arm][st] for st in S])
            lo, hi = boot_ci(m, rng)
            out.append(f"{arm:<12} {len(allv):>8} {int((allv >= 0.5).sum()):>9} {int((m >= 0.5).sum()):>17}   "
                       f"{m.mean():.3f} [{lo:.3f}, {hi:.3f}]")
        out.append("paired difference B - A (per chain)   mean [95% CI]   chains B>A / n")
        for b_, a_ in pairs:
            d = means[b_] - means[a_]
            lo, hi = boot_ci(d, rng)
            out.append(f"{b_:>12} - {a_:<12} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   {int((d > 0).sum())}/{len(d)}")
        out.append("")
    out.append("changed (gt_mismatch > 0) chains: " + ",".join(st for st in chains if mism[st]))
    text = "\n".join(out)
    print(text)
    open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
