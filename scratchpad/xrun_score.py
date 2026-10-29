"""Cross-run TopoRef comparison, stage D scoring (Engaging CPU): per (model, arm, chain, sample) alignment P@Q (+ the
position-only baseline), contact P@L and c2c TM; per-arm tables with paired bootstrap CIs over chains.

P@Q = the training metric (model_trainer_base._topology_aux_losses): per sample, Q = residues with an aligned target
element, the top-Q scored (residue, element) cells of the alignment logits (LAST network call of the trajectory), the
fraction that are true. Baseline = the same with score -|query position - element midpoint| (query position = the
SEQRES index for T8b, arange(L) otherwise; element midpoint = topology_he_pos_raw), as logged in training. Samples with
Q = 0 are EXCLUDED (the logged training metric averaged in 0s of batches with no aligned row: "undiluted", user 2026-10-28). H/E-only variant: the rows restricted to
residues whose target element is a helix or a strand (types from the reference row's DSSP runs).
Targets: template arms = the eval index's USalign target (TopologyReferenceTransform._align_for, the training reader);
self arm = each residue's own element in the native row's runs (an element beyond the shown T = none).
Contact P@L = the trajectory's own contact_precision_at_L against the model's own native label (samples.jsonl).
TM = stageB_c2c_from_maps.py rows (USalign -TMscore 5, native-normalised), joined on (chain, sample_index).
Statistics: per-chain mean over samples, then mean over chains; differences paired per chain; 95% percentile bootstrap
over chains, N_BOOT 2000, default_rng(0) (ProteinEBMalign synth_compare.py:13,30,57-58 @89ea726).
Usage: python scratchpad/xrun_score.py --x-dir $S/toporef_xrun --out-dir $S/toporef_xrun/score
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from proteinfoundation.datasets.sse_topology import runs_to_spans
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform

N_BOOT = 2000                    # synth_compare.py:13
HE_TYPES = (1, 2)                # DSSP_HELIX, DSSP_STRAND (sse_topology)
TM_CUT = 0.5                     # user 2026-10-28 "threshold counts (TM>=0.5)"
MODELS = ("cb8", "t8", "t8b")
ARMS = ("self", "equal", "indel")
INDEX_FILE = {"cb8": "topology_index_cb8_synth.pt", "t8": "topology_index_confind_synth_t8.pt",
              "t8b": "topology_index_confind_synth_t8b.pt"}
ELEMENT_TYPES = {"cb8": (1, 2), "t8": (0, 1, 2), "t8b": (0, 1, 2)}


def elements_of(tf, row):
    """(element index per residue, element types) of index row `row`, elements = runs of the index's element types."""
    runs = tf._runs_for(row)
    spans = runs_to_spans(runs)
    keep = [i for i, (t, _) in enumerate(runs) if t in tf.element_types]
    L = sum(n for _, n in runs)
    elem = np.full(L, -1, dtype=np.int64)
    for a, i in enumerate(keep):
        s, e = spans[i]
        elem[s:e] = a
    return elem, np.asarray([runs[i][0] for i in keep])


def p_at_q(score, tgt, rows):
    """precision@Q over the rows `rows` (bool [L]); None when no row has an aligned target. Ties at the top-Q cutoff
    (the position score lives on a 0.5 grid) are resolved by their EXPECTED precision under random tie-breaking, so
    the value does not depend on torch.topk's unspecified tie order (training's topk picks one order arbitrarily)."""
    L, T = score.shape
    aligned = (tgt >= 0) & rows
    q = int(aligned.sum())
    if q == 0:
        return None
    A = np.zeros((L, T), dtype=bool)
    A[np.flatnonzero(aligned), tgt[aligned]] = True
    s = np.where(rows[:, None], score, -np.inf).reshape(-1)
    a = A.reshape(-1)
    q = min(q, s.size)
    thr = np.sort(s)[::-1][q - 1]
    above, tie = s > thr, s == thr
    return float((a[above].sum() + (q - above.sum()) * a[tie].sum() / tie.sum()) / q)


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    lo, hi = np.quantile(b, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default=",".join(MODELS), help="subset for a partial (smoke) scoring")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--chunks", default="", help="comma list of chunk ids (partial scoring); default every chunk")
    a = ap.parse_args()
    models, arms = a.models.split(","), a.arms.split(",")
    assert set(arms) <= set(ARMS) and set(models) <= set(MODELS), (models, arms)
    os.makedirs(a.out_dir, exist_ok=True)
    rec = []
    for m in models:
        tfs = {}
        for idx_arm in ("equal", "indel"):
            tf = TopologyReferenceTransform(index_path=os.path.join(a.x_dir, "index", f"{m}_{idx_arm}", INDEX_FILE[m]),
                                            reference_source="synthetic", sse_types=ELEMENT_TYPES[m],
                                            element_types=ELEMENT_TYPES[m])
            tf._ensure_loaded()
            tfs[idx_arm] = tf
        for arm in arms:
            tf = tfs["equal" if arm == "equal" else "indel"]
            tm_rows = {}
            ck = "|".join(a.chunks.split(",")) if a.chunks else r"\d+"
            pat = re.compile(rf"{m}_{arm}_c({ck})(\.jsonl)?$")
            fb = sorted(g for g in glob.glob(os.path.join(a.x_dir, "stageB", f"{m}_{arm}_c*.jsonl"))
                        if pat.match(os.path.basename(g)))
            fd = sorted(g for g in glob.glob(os.path.join(a.x_dir, "dumps", f"{m}_{arm}_c*"))
                        if pat.match(os.path.basename(g)))
            if a.chunks:  # a requested chunk with no files must not shrink the chain set silently
                want = set(a.chunks.split(","))
                got_b = {pat.match(os.path.basename(g)).group(1) for g in fb}
                got_d = {pat.match(os.path.basename(g)).group(1) for g in fd}
                assert got_b == want and got_d == want, (m, arm, sorted(want), sorted(got_b), sorted(got_d))
            for f in fb:
                for l in open(f):
                    r = json.loads(l)
                    k = (r["stem"], r["sample_index"])
                    assert k not in tm_rows, f"{f}: duplicate {k}"
                    tm_rows[k] = r["tm"]
            for d in fd:
                for l in open(os.path.join(d, "samples.jsonl")):
                    r = json.loads(l)
                    z = np.load(os.path.join(d, r["file"]))
                    L, T = int(z["L"]), int(z["he_tokens"].shape[0])
                    ref = str(z["ref_id"])
                    row = tf._id_to_row[ref]
                    if arm == "self":
                        assert ref == r["stem"], (m, arm, r["stem"], ref)
                        elem, types = elements_of(tf, row)
                        assert len(elem) == L, (r["stem"], len(elem), L)
                        tgt = np.where(elem < T, elem, -1)
                    else:
                        assert not bool(tf._index["row_is_native"][row]) and ref.startswith(r["stem"] + "@"), ref
                        a0, b0 = int(tf._index["align_offset"][row]), int(tf._index["align_offset"][row + 1])
                        assert b0 - a0 == L, (m, arm, r["stem"], ref, b0 - a0, L)  # else _align_for goes all-NONE
                        tgt = tf._align_for(row, L, T).numpy()
                        _, types = elements_of(tf, row)
                    assert int(tgt.max()) < T
                    assert len(types) >= T, (m, arm, r["stem"], len(types), T)
                    he_rows = (tgt >= 0) & np.isin(types[np.clip(tgt, 0, None)], HE_TYPES)
                    q_pos = (z["residue_seqres_idx"].astype(np.float64) if m == "t8b"
                             else np.arange(L, dtype=np.float64))
                    pos = -np.abs(q_pos[:, None] - z["he_pos_raw"][None, :T].astype(np.float64))
                    head = z["align_logits_last"].astype(np.float64)
                    allr = np.ones(L, dtype=bool)
                    key = (r["stem"], r["sample_index"])
                    assert key in tm_rows, f"{m} {arm} {key}: no c2c row"
                    rec.append(dict(model=m, arm=arm, stem=r["stem"], sample=r["sample_index"], ref_id=ref, L=L, T=T,
                                    pq=p_at_q(head, tgt, allr), pq_base=p_at_q(pos, tgt, allr),
                                    pq_he=p_at_q(head, tgt, he_rows), pq_he_base=p_at_q(pos, tgt, he_rows),
                                    pl=r["metrics"]["contact_precision_at_L"], tm=tm_rows[key]))
            n_dump = sum(1 for x in rec if x["model"] == m and x["arm"] == arm)
            assert n_dump == len(tm_rows), f"{m} {arm}: {n_dump} samples vs {len(tm_rows)} c2c rows"
    with open(os.path.join(a.out_dir, "samples.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in rec))
    # every (arm, chain) must carry the same template across models and the same sample count
    by = defaultdict(lambda: defaultdict(list))
    for x in rec:
        by[(x["arm"], x["stem"])][x["model"]].append(x)
    for (arm, s), d in by.items():
        assert set(d) == set(models), (arm, s, sorted(d))
        assert len({len(v) for v in d.values()}) == 1, (arm, s)
        assert len({x["ref_id"] for v in d.values() for x in v}) == 1, (arm, s, {m: v[0]["ref_id"] for m, v in d.items()})
    out = []
    for arm in arms:
        stems = sorted(s for (ar, s) in by if ar == arm)
        out.append(f"\n## arm {arm}: {len(stems)} chains, {len(by[(arm, stems[0])][models[0]])} samples each")
        per = {}
        for m in models:
            for k in ("pq", "pq_base", "pq_he", "pq_he_base", "pl", "tm"):
                vals = {}
                for s in stems:
                    v = [x[k] for x in by[(arm, s)][m] if x[k] is not None]
                    if v:
                        vals[s] = float(np.mean(v))
                per[(m, k)] = vals
            # head minus its position-only baseline, per chain (the within-model alignment gain)
            for k, base in (("pq_gain", "pq_base"), ("pq_he_gain", "pq_he_base")):
                h = per[(m, k.replace("_gain", ""))]
                per[(m, k)] = {s: h[s] - per[(m, base)][s] for s in h if s in per[(m, base)]}
            n_ex = sum(1 for s in stems for x in by[(arm, s)][m] if x["pq"] is None)
            n_ex_he = sum(1 for s in stems for x in by[(arm, s)][m] if x["pq_he"] is None)
            tms = [x["tm"] for s in stems for x in by[(arm, s)][m]]
            out.append(f"{m:>4}: samples TM>={TM_CUT} {sum(t >= TM_CUT for t in tms)}/{len(tms)}; "
                       f"chains mean TM>={TM_CUT} {sum(v >= TM_CUT for v in per[(m, 'tm')].values())}/{len(stems)}; "
                       f"samples with Q=0 excluded from P@Q {n_ex} (H/E-only {n_ex_he})")
        out.append("metric        model  chains   mean   [95% CI]")
        rng = np.random.default_rng(0)
        for k in ("pq", "pq_base", "pq_gain", "pq_he", "pq_he_base", "pq_he_gain", "pl", "tm"):
            for m in models:
                v = list(per[(m, k)].values())
                if v:
                    mu, lo, hi = boot_ci(v, rng)
                    out.append(f"{k:<13} {m:>5} {len(v):>7} {mu:6.3f}  [{lo:.3f}, {hi:.3f}]")
        out.append("paired difference (per chain, chains present in both)  mean [95% CI]  chains B>A / n")
        for k in ("pq", "pq_gain", "pq_he", "pq_he_gain", "pl", "tm"):
            for ma, mb in (("cb8", "t8"), ("cb8", "t8b"), ("t8", "t8b")):
                if ma not in models or mb not in models:
                    continue
                if k == "pl" and "cb8" in (ma, mb):
                    continue  # CB-8 P@L is scored on its own CB-8 label: not comparable (user 2026-10-28)
                common = sorted(set(per[(ma, k)]) & set(per[(mb, k)]))
                dlt = [per[(mb, k)][s] - per[(ma, k)][s] for s in common]
                if dlt:
                    mu, lo, hi = boot_ci(dlt, rng)
                    out.append(f"{k:<6} {mb}-{ma:<4} {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]  {sum(x > 0 for x in dlt)}/{len(dlt)}")
        with open(os.path.join(a.out_dir, f"per_chain_{arm}.tsv"), "w") as fh:
            keys = [(m, k) for m in models for k in ("pq", "pq_base", "pq_gain", "pq_he", "pq_he_base", "pq_he_gain", "pl", "tm")]
            fh.write("stem\t" + "\t".join(f"{m}_{k}" for m, k in keys) + "\n")
            for s in stems:
                fh.write(s + "\t" + "\t".join(f"{per[mk].get(s, float('nan')):.4f}" for mk in keys) + "\n")
    txt = "\n".join(out)
    open(os.path.join(a.out_dir, "summary.txt"), "w").write(txt + "\n")
    print(txt)
    print("XRUN_SCORE_DONE")


if __name__ == "__main__":
    main()
