"""T8b native ConFind maps: gate comparison and full-run check (CPU, Engaging compute node).

--gate: recomputed maps in <gate-dir>/processed vs the stored originals saved at prep (<gate-dir>/ref_maps.pt).
  Metrics = the S1 gate (job 23623562; precompute_synthetic_topology_index.py native check): contacts p >= 0.01
  over the full L x L map, Jaccard |a & b| / max(|a | b|, 1), max |p_new - p_stored|; cell agreement for contrast.
  python t8b_confind_compare.py --gate --gate-dir G --status-dir G_STATUS
--check: every cache row has a map with the properties measured on the stored originals (ref_maps.pt), and the
  status CSVs mark every cache basename processed.
  python t8b_confind_compare.py --check --cache C.json --status-tsv S --status-dir STATUS --ref-maps G/ref_maps.pt
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from concurrent.futures import ProcessPoolExecutor

import torch

THR = 0.01  # CONFIND_THRESHOLD of the S1 gate / the contact_method: confind transform
S1_REFERENCE = "S1 gate 23623562 (92 natives, bf16): Jaccard median 0.9988 / min 0.9894, max|dp| 0.0059"


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
    ap.add_argument("--gate-dir")
    ap.add_argument("--status-dir", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--status-tsv")
    ap.add_argument("--ref-maps")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    assert args.gate != args.check, "pick exactly one of --gate / --check"
    if args.gate:
        assert args.gate_dir
        run_gate(args)
    else:
        assert args.cache and args.status_tsv and args.ref_maps
        run_check(args)


if __name__ == "__main__":
    main()
