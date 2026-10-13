"""Data availability under a CA-clash gate on the t2 templates (user 2026-10-13: "gate at 3.4 A because of the normal
Carbon VdW radius of 1.7 A ... We could relax the gate to at most 1 with such clashes. Calculate statistics on how gating
on this would impact data availability after filtering.").

clash = a non-bonded (|i - j| >= 2) CA-CA pair closer than --clash (3.4 A). Gates (the user's): G0 = no clash,
G1 = at most 1 clash. Also per TM bin (0.1 wide, as the generator's t2_prod_summary bins) and the TM span each chain's
selection covers before vs after gating.
For a seeded sample of chains (all in-range templates of each, TM [0.5, 0.9]): templates surviving each gate (template- and
chain-weighted), chains left with none, and the templates per chain the cap-64 selection would then get (gate FIRST, then
select evenly in TM value). Natives of the same chains are the control (same metric on resolved CAs).
Usage (PYTHONPATH=repo): python scratchpad/t8_clash_gate_stats.py --pack ROOT --chain-list TXT --native-pack PACK --out DIR
"""

import argparse
import io
import json
import os
import random
from multiprocessing import Pool

import numpy as np
import torch

from proteinfoundation.utils.t2_pack_reader import TemplatePack

CA = 1
_TP, _ARGS, _NAT = None, None, None


def counts(ca, ok, clash):
    c = ca[ok]
    rn = np.flatnonzero(ok)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    i, j = np.triu_indices(len(c), k=1)
    far = (rn[j] - rn[i]) >= 2
    near = (rn[j] - rn[i]) == 1
    return int((far & (d[i, j] < clash)).sum()), int((near & (d[i, j] < clash)).sum())


def chain_job(stem):
    tp, a = _TP, _ARGS
    o, ln = _NAT[stem]
    with open(a.native_pack, "rb") as fh:
        fh.seek(o)
        g = torch.load(io.BytesIO(fh.read(ln)), map_location="cpu", weights_only=False)
    n_nb, n_st = counts(g.coords[:, CA].numpy(), (g.coord_mask[:, CA] > 0.5).numpy(), a.clash)
    tm = tp.col["tm_native"]
    out = []
    for r in tp.rows(stem):
        t = float(tm[r])
        if not (a.tm_min <= t <= a.tm_max):
            continue
        rec = tp._read(r)
        full = np.zeros((rec["atom_mask"].shape[0], 37, 3), dtype=np.float32)
        full[rec["atom_mask"]] = rec["coords"]
        nb, stp = counts(full[:, CA], rec["atom_mask"][:, CA], a.clash)
        out.append((t, int(rec["draw"]), nb, stp))
    return stem, n_nb, n_st, out


def init(pack, args, nat):
    global _TP, _ARGS, _NAT
    _TP, _ARGS, _NAT = TemplatePack(pack), args, nat
    for k in _TP.col._z.files:
        _TP.col[k]


def evenly_in_tm(rows, cap, lo, hi):
    """Same rule as precompute_synthetic_topology_index.select_pack_rows, on (tm, draw) tuples."""
    if len(rows) <= cap:
        return len(rows)
    free = list(range(len(rows)))
    n = 0
    for t in np.linspace(lo, hi, cap):
        k = min(free, key=lambda i: (abs(rows[i][0] - t), rows[i][1]))
        free.remove(k)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--chain-list", required=True)
    ap.add_argument("--native-pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-chains", type=int, default=2000)
    ap.add_argument("--clash", type=float, default=3.4)
    ap.add_argument("--tm-min", type=float, default=0.5)
    ap.add_argument("--tm-max", type=float, default=0.9)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.native_pack + ".idx.npz", allow_pickle=True)
    nat = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    tp = TemplatePack(args.pack)
    have = set(str(s) for s in tp.stems)
    pool = [s.strip() for s in open(args.chain_list) if s.strip() in have and s.strip() in nat]
    sample = random.Random(args.seed).sample(pool, min(args.sample_chains, len(pool)))
    with Pool(args.workers, initializer=init, initargs=(args.pack, args, nat)) as p:
        res = p.map(chain_job, sample, chunksize=4)

    gates = {"G0": lambda nb, st: nb == 0, "G1": lambda nb, st: nb <= 1}
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]

    def span(rows):
        return (max(r[0] for r in rows) - min(r[0] for r in rows)) if rows else 0.0
    rep = {"clash_A": args.clash, "chains": len(res), "templates_in_range": sum(len(o) for *_, o in res)}
    nat_nb = np.array([n for _, n, _, _ in res])
    nat_st = np.array([s for _, _, s, _ in res])
    rep["native_control"] = {"natives_with_any_clash": int((nat_nb > 0).sum()), "native_clash_pairs_max": int(nat_nb.max()),
                             "natives_with_step_lt_clash": int((nat_st > 0).sum())}
    tnb = np.array([x[2] for *_, o in res for x in o])
    rep["template_clash_pairs_hist"] = {str(k): int(v) for k, v in zip(*np.unique(np.minimum(tnb, 10), return_counts=True))}
    base_sel = np.array([evenly_in_tm(o, args.cap, args.tm_min, args.tm_max) for *_, o in res])
    for name, f in gates.items():
        surv = [[x for x in o if f(x[2], x[3])] for *_, o in res]
        n_in = np.array([len(o) for *_, o in res])
        n_s = np.array([len(s) for s in surv])
        sel = np.array([evenly_in_tm(s, args.cap, args.tm_min, args.tm_max) for s in surv])
        keep = n_in > 0
        rep[name] = {
            "templates_kept_template_weighted": round(float(n_s.sum() / max(n_in.sum(), 1)), 4),
            "templates_kept_chain_weighted": round(float(np.mean(n_s[keep] / n_in[keep])), 4),
            "chains_with_zero_left": int(((n_s == 0) & keep).sum()),
            "templates_left_per_chain": {q: float(np.quantile(n_s[keep], float(q))) for q in ("0", "0.05", "0.25", "0.5")},
            "selected_per_chain_before": {q: float(np.quantile(base_sel, float(q))) for q in ("0.05", "0.25", "0.5", "0.75")},
            "selected_per_chain_after": {q: float(np.quantile(sel, float(q))) for q in ("0.05", "0.25", "0.5", "0.75")},
            "selected_total_before_after": [int(base_sel.sum()), int(sel.sum())],
            "kept_fraction_by_tm_bin": {f"{lo:.1f}-{hi:.1f}": round(
                sum(1 for s_ in surv for x in s_ if lo <= x[0] < hi or (hi == 0.9 and x[0] == hi)) /
                max(sum(1 for *_, o in res for x in o if lo <= x[0] < hi or (hi == 0.9 and x[0] == hi)), 1), 4)
                for lo, hi in bins},
            "tm_span_per_chain_before": {q: round(float(np.quantile([span(o) for *_, o in res if o], float(q))), 3)
                                         for q in ("0.05", "0.5")},
            "tm_span_per_chain_after": {q: round(float(np.quantile([span(s_) for s_ in surv if s_] or [0], float(q))), 3)
                                        for q in ("0.05", "0.5")},
        }
    with open(os.path.join(args.out, "clash_gate_stats.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
