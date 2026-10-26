"""T8b native ConFind maps: gate comparison and full-run check (CPU, Engaging compute node).

--gate: recomputed maps in <gate-dir>/processed vs the stored originals saved at prep (<gate-dir>/ref_maps.pt).
  Metrics = the S1 gate (job 23623562; precompute_synthetic_topology_index.py native check): contacts p >= 0.01
  over the full L x L map, Jaccard |a & b| / max(|a | b|, 1), max |p_new - p_stored|; cell agreement for contrast.
  python t8b_confind_compare.py --gate --gate-dir G --status-dir G_STATUS
--check: every cache row has a map with the properties measured on the stored originals (ref_maps.pt), and the
  status CSVs mark every cache basename processed.
  python t8b_confind_compare.py --check --cache C.json --status-tsv S --status-dir STATUS --ref-maps G/ref_maps.pt
--padfix-check: every recomputed pad-fix copy (cache rows, under --fix-tree) vs its ORIGINAL (same relative path under
  --tree): every key but contact_map_confind equal (tensors: dtype, shape, values, NaN == NaN), the new map has the
  original's dtype / shape (L, L), is finite, in [0, 1], exactly symmetric where the original is, differs from the stored
  one, and has a processed status row. A recomputed map identical to the stored one is not a failure: that stem is
  DROPPED from the fix (coordinator 2026-10-26) and recorded in --dropped-tsv (stem, reason); its copy stays in the fix
  tree, unused (the table keeps it unchanged). Reports max|dp| (all, row/col L-1, rest), how often the max cell is in
  row/col L-1, Jaccard; writes --out-tsv (kept stems) + --dropped-tsv only on a full pass.
  python t8b_confind_compare.py --padfix-check --cache C.json --tree ORIG/processed --fix-tree FIX/processed \
      --status-dir STATUS --out-tsv check_padfix.tsv --dropped-tsv dropped_padfix.tsv
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

THR = 0.01  # CONFIND_THRESHOLD of the S1 gate / the contact_method: confind transform
S1_REFERENCE = "S1 gate 23623562 (92 natives, bf16): Jaccard median 0.9988 / min 0.9894, max|dp| 0.0059"
BS1_REFERENCE = "diag 26829457: bs 1 vs stored UNPADDED June maps (n=100, L 385-504): max|dp| max 0.0078, Jaccard min 0.9940"
PADFIX_COLS = ["stem", "L", "max_dp", "max_dp_last", "max_dp_rest", "max_in_last", "jaccard"]


def map_metrics(new, ref):
    new, ref = new.float(), ref.float()
    a, b = new >= THR, ref >= THR
    jac = ((a & b).sum() / (a | b).sum().clamp_min(1)).item()
    agree = (a == b).float().mean().item()
    return jac, agree, (new - ref).abs().max().item()


def tensor_stats(m):
    f = m.float()
    finite = torch.isfinite(f)
    ff = f[finite]
    return {"dtype": str(m.dtype), "shape": tuple(m.shape), "n_nonfinite": int((~finite).sum()),
            "min": ff.min().item(), "max": ff.max().item(),
            "asym": (f - f.T).abs().max().item() if m.dim() == 2 and m.shape[0] == m.shape[1] else float("nan")}


def file_stats(path):
    g = torch.load(path, map_location="cpu", weights_only=False)
    m = getattr(g, "contact_map_confind", None)
    out = {"path": path, "L_coords": int(g.coords.shape[0]), "has": m is not None}
    if m is not None:
        out.update(tensor_stats(m))
    return out


def values_equal(x, y):
    if torch.is_tensor(x) or torch.is_tensor(y):
        return (torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape
                and bool(((x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)).all()))
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype
                and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return type(x) is type(y) and bool(x == y)


def padfix_one(job):
    """One recomputed copy vs its original: keys that differ besides the map, map stats, map deltas."""
    stem, new_path, orig_path = job
    dn = dict(torch.load(new_path, map_location="cpu", weights_only=False).items())
    do = dict(torch.load(orig_path, map_location="cpu", weights_only=False).items())
    m, r = dn.pop("contact_map_confind", None), do.pop("contact_map_confind", None)
    other = sorted(set(dn) ^ set(do)) + sorted(k for k in set(dn) & set(do) if not values_equal(dn[k], do[k]))
    L = int(dn["coords"].shape[0])
    out = {"stem": stem, "L": L, "other_diff": other, "has": m is not None and r is not None}
    if not out["has"]:
        return out
    out.update(tensor_stats(m))
    out["ref"] = tensor_stats(r)
    if m.shape != r.shape or tuple(m.shape) != (L, L):
        return out
    d = (m.float() - r.float()).abs()
    last = torch.zeros(L, L, dtype=torch.bool)
    last[L - 1, :] = True
    last[:, L - 1] = True
    out["max_dp"] = d.max().item()
    out["max_dp_last"] = d[last].max().item()
    out["max_dp_rest"] = d[~last].max().item() if L > 1 else 0.0
    out["max_in_last"] = out["max_dp_last"] >= out["max_dp_rest"]
    out["jaccard"] = map_metrics(m, r)[0]
    return out


def qstr(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
    return f"min {v[0]:.4g} p5 {q(0.05):.4g} p25 {q(0.25):.4g} median {q(0.5):.4g} p75 {q(0.75):.4g} p95 {q(0.95):.4g} max {v[-1]:.4g}"


def run_padfix_check(args):
    with open(args.cache) as fh:
        cache = json.load(fh)
    tree, fix = os.path.abspath(args.tree).rstrip("/"), os.path.abspath(args.fix_tree).rstrip("/")
    assert len(cache) > 0 and all(p.startswith(fix + "/") for p, _, _ in cache), "cache row outside --fix-tree"
    jobs = [(os.path.basename(p)[:-3], p, os.path.join(tree, os.path.relpath(p, fix))) for p, _, _ in cache]
    assert len({j[0] for j in jobs}) == len(jobs), "duplicate stems in the cache"
    done, failed = processed_names(args.status_dir)
    not_done = [s for s, _, _ in jobs if f"{s}.pt" not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(padfix_one, jobs, chunksize=16))
    assert len(res) == len(jobs)
    ok = [x for x in res if "max_dp" in x and x["max_dp"] > 0.0]
    dropped = [x for x in res if "max_dp" in x and x["max_dp"] == 0.0]
    bad = {"keys other than contact_map_confind differ": [(x["stem"], x["other_diff"]) for x in res if x["other_diff"]],
           "missing map (new or original)": [x["stem"] for x in res if not x["has"]],
           "shape != original / (L, L)": [x["stem"] for x in res if x["has"] and "max_dp" not in x],
           "dtype != original": [x["stem"] for x in res if x["has"] and x["dtype"] != x["ref"]["dtype"]],
           "non-finite": [x["stem"] for x in res if x["has"] and x["n_nonfinite"] > 0],
           "outside [0,1]": [x["stem"] for x in res if x["has"] and (x["min"] < 0.0 or x["max"] > 1.0)],
           "not symmetric (original is)": [x["stem"] for x in res if x["has"] and x["ref"]["asym"] == 0.0 and x["asym"] != 0.0],
           "not processed per status CSVs": [(s, failed.get(s + ".pt", "")) for s in not_done]}
    print(f"PADFIX CHECK n={len(res)} of {len(cache)} copies (L {min(x['L'] for x in res)}-{max(x['L'] for x in res)})",
          flush=True)
    for k, v in bad.items():
        print(f"  {k}: {len(v)} of {len(res)}" + (f" e.g. {v[:5]}" if v else ""), flush=True)
    print(f"  DROPPED (recomputed map identical to the stored one): {len(dropped)} of {len(res)}"
          + (f" e.g. {[x['stem'] for x in dropped[:5]]}" if dropped else ""), flush=True)
    if ok:
        n_last = sum(x["max_in_last"] for x in ok)
        print(f"  max|dp| all   : {qstr([x['max_dp'] for x in ok])}", flush=True)
        print(f"  max|dp| row/col L-1: {qstr([x['max_dp_last'] for x in ok])}", flush=True)
        print(f"  max|dp| rest  : {qstr([x['max_dp_rest'] for x in ok])}", flush=True)
        print(f"  max cell in row/col L-1: {n_last} of {len(ok)} ({n_last / len(ok):.2%})", flush=True)
        print(f"  Jaccard (p >= {THR}): {qstr([x['jaccard'] for x in ok])}", flush=True)
        print(f"  reference: {BS1_REFERENCE}", flush=True)
    n_bad = sum(len(v) for v in bad.values())
    assert n_bad == 0, f"PADFIX CHECK FAILED: {n_bad} violations (no check TSV written)"
    assert not os.path.exists(args.out_tsv) and not os.path.exists(args.dropped_tsv), "check / dropped TSV exists"
    with open(args.dropped_tsv + ".partial", "w", newline="") as fh:
        fh.write("stem\treason\n" + "".join(f"{x['stem']}\tmap_identical_to_stored\n" for x in sorted(dropped, key=lambda x: x["stem"])))
    os.replace(args.dropped_tsv + ".partial", args.dropped_tsv)
    with open(args.out_tsv + ".partial", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PADFIX_COLS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(ok, key=lambda x: x["stem"]))
    os.replace(args.out_tsv + ".partial", args.out_tsv)
    print(f"PADFIX CHECK OK: {len(ok)} copies differ from the stored map in contact_map_confind only -> {args.out_tsv}; "
          f"{len(dropped)} dropped -> {args.dropped_tsv}", flush=True)


def processed_names(status_dir):
    csvs = sorted(glob.glob(os.path.join(status_dir, "*.csv")))
    done, failed = set(), {}
    for c in csvs:
        with open(c, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["status"] == "processed":
                    done.add(r["file"])
                else:
                    failed[r["file"]] = r["error"]
    print(f"status: {len(csvs)} CSVs in {status_dir}, {len(done)} processed basenames, "
          f"{len(set(failed) - done)} failed-and-never-processed", flush=True)
    return done, {k: v for k, v in failed.items() if k not in done}


def run_gate(args):
    with open(os.path.join(args.gate_dir, "manifest.tsv"), newline="") as fh:
        man = list(csv.DictReader(fh, delimiter="\t"))
    ref = torch.load(os.path.join(args.gate_dir, "ref_maps.pt"), weights_only=False)
    assert len(man) == len(ref) > 0
    done, failed = processed_names(args.status_dir)
    not_done = [r["stem"] for r in man if f"{r['stem']}.pt" not in done]
    assert not not_done, f"{len(not_done)} gate files not processed by the run, e.g. {not_done[:5]} {failed}"
    res = []
    for r in man:
        new = torch.load(r["gate_path"], map_location="cpu", weights_only=False).contact_map_confind
        old = ref[r["stem"]]
        assert new.shape == old.shape == (int(r["L"]), int(r["L"])), (r["stem"], new.shape, old.shape)
        res.append((r["stem"], int(r["L"])) + map_metrics(new, old))
    assert len(res) == len(man)
    jac = [x[2] for x in res]
    agr = [x[3] for x in res]
    mdp = [x[4] for x in res]
    print(f"GATE n={len(res)} of {len(man)} (L {min(x[1] for x in res)}-{max(x[1] for x in res)}): "
          f"contact Jaccard (p >= {THR}) median {st.median(jac):.4f} min {min(jac):.4f} | "
          f"max|dp| median {st.median(mdp):.4f} max {max(mdp):.4f} | cell agreement mean {st.mean(agr):.5f}", flush=True)
    print("lowest-Jaccard 5: " + ", ".join(f"{s} L={L} J={j:.4f} max|dp|={d:.4f}"
                                           for s, L, j, _, d in sorted(res, key=lambda x: x[2])[:5]), flush=True)
    print(f"reference: {S1_REFERENCE}", flush=True)


def run_check(args):
    with open(args.cache) as fh:
        cache = json.load(fh)
    with open(args.status_tsv, newline="") as fh:
        changed = {r["stem"]: int(r["L_new"]) for r in csv.DictReader(fh, delimiter="\t") if r["status"] == "changed"}
    stems = {os.path.basename(p)[:-3]: (p, L) for p, L, _ in cache}
    assert len(changed) > 0, "no changed stems"
    assert len(stems) == len(cache) and set(stems) == set(changed), "cache != changed stems of the status TSV"

    # stored-map profile: only properties every stored original has become assertions
    ref = torch.load(args.ref_maps, weights_only=False)
    prof = [tensor_stats(m) for m in ref.values()]
    assert len(prof) > 0
    p_dtypes = {p["dtype"] for p in prof}
    p_square = all(len(p["shape"]) == 2 and p["shape"][0] == p["shape"][1] for p in prof)
    p_finite = all(p["n_nonfinite"] == 0 for p in prof)
    p_unit = all(0.0 <= p["min"] and p["max"] <= 1.0 for p in prof)
    p_sym = all(p["asym"] == 0.0 for p in prof)
    print(f"stored profile n={len(prof)}: dtypes {sorted(p_dtypes)}, square {p_square}, finite {p_finite}, "
          f"in [0,1] {p_unit} (min {min(p['min'] for p in prof):.4g}, max {max(p['max'] for p in prof):.4g}), "
          f"exactly symmetric {p_sym} (max|M-M^T| median {st.median(p['asym'] for p in prof):.4g} "
          f"max {max(p['asym'] for p in prof):.4g})", flush=True)

    done, failed = processed_names(args.status_dir)
    not_done = sorted(s for s in stems if f"{s}.pt" not in done)
    print(f"status: {len(stems) - len(not_done)} of {len(stems)} cache stems processed", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(file_stats, [stems[s][0] for s in sorted(stems)], chunksize=64))
    assert len(res) == len(stems)
    by = {os.path.basename(r["path"])[:-3]: r for r in res}
    with_map = [s for s in by if by[s]["has"]]
    bad = {"missing map": [s for s in by if not by[s]["has"]],
           "coords L != L_new": [s for s in by if by[s]["L_coords"] != changed[s]]}
    bad["shape != (L, L)"] = [s for s in with_map if by[s]["shape"] != (changed[s], changed[s])]
    bad["dtype not stored dtype"] = [s for s in with_map if by[s]["dtype"] not in p_dtypes]
    if p_finite:
        bad["non-finite"] = [s for s in with_map if by[s]["n_nonfinite"] > 0]
    if p_unit:
        bad["outside [0,1]"] = [s for s in with_map if by[s]["min"] < 0.0 or by[s]["max"] > 1.0]
    if p_sym:
        bad["not symmetric"] = [s for s in with_map if by[s]["asym"] != 0.0]
    asym = [by[s]["asym"] for s in with_map]
    print(f"CHECK n={len(res)} of {len(stems)} files, {len(with_map)} with contact_map_confind; "
          f"new max|M-M^T| median {st.median(asym) if asym else float('nan'):.4g} "
          f"max {max(asym) if asym else float('nan'):.4g}", flush=True)
    for k, v in bad.items():
        print(f"  {k}: {len(v)} of {len(res)}" + (f" e.g. {v[:5]}" if v else ""), flush=True)
    print(f"  not processed per status CSVs: {len(not_done)} of {len(stems)}"
          + (f" e.g. {[(s, failed.get(s + '.pt', '')) for s in not_done[:5]]}" if not_done else ""), flush=True)
    n_bad = sum(len(v) for v in bad.values()) + len(not_done)
    assert n_bad == 0, f"CHECK FAILED: {n_bad} violations"
    print(f"CHECK OK: {len(res)} of {len(stems)} changed stems have a valid map and a processed status row", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--padfix-check", action="store_true")
    ap.add_argument("--tree", help="--padfix-check: original processed/ root")
    ap.add_argument("--fix-tree", help="--padfix-check: fix tree processed/ root holding the recomputed copies")
    ap.add_argument("--out-tsv", help="--padfix-check: per-stem result TSV (written only on a full pass)")
    ap.add_argument("--dropped-tsv", help="--padfix-check: stems dropped from the fix (identical map), with the reason")
    ap.add_argument("--gate-dir")
    ap.add_argument("--status-dir", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--status-tsv")
    ap.add_argument("--ref-maps")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    assert args.gate + args.check + args.padfix_check == 1, "pick exactly one of --gate / --check / --padfix-check"
    if args.padfix_check:
        assert args.cache and args.tree and args.fix_tree and args.out_tsv and args.dropped_tsv
        run_padfix_check(args)
    elif args.gate:
        assert args.gate_dir
        run_gate(args)
    else:
        assert args.cache and args.status_tsv and args.ref_maps
        run_check(args)


if __name__ == "__main__":
    main()
