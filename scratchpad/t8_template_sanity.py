"""Sanity check of a t2 template pack before T8 training (user 2026-10-13: "run a thorough sanity check on the template
data before starting the real training job and sign it off on the code and log").

Layer 1 -- the whole index (every template row): chain coverage vs the chain list; per-chain counts, in-range (TM) and
after the cap-64 selection; record layout (nbytes == layout(L, A)), shard offsets inside the shard files and
non-overlapping; value ranges (L, A <= 37 L, TM in [0, 1]); one L_native per chain; draw ids unique per chain.
Layer 2 -- a seeded sample of chains, every SELECTED template decoded: finite coordinates; atom_mask count == A; N/CA/C
present; residue_index contiguous 1..L; orig_idx in [-1, L_native) and strictly increasing over its non -1 entries;
broken CA-CA steps by the GENERATOR's own rule (of-t2deep t2_stage_b.n_broken_steps: > 4.0 A, except a step joining two
native-adjacent residues whose native step is itself broken; more than 1 = dropped, the user's "badly broken");
non-bonded CA-CA < 3.0 A (clashes); L_native
vs the native in the training pack (a mismatch = a template of a different native: reported as provenance, and the
broken-step rule is then not applied); sequence identity to the native at orig_idx positions (REPORTED only: templates may
be redesigned); USalign TM(template, native) recomputed on a sub-sample vs the stored tm_native.
Hard FAIL only for: corrupt layout / offsets, out-of-range indices, non-finite coordinates, missing backbone atoms on
residues, and templates breaking the user's broken-step rule. Everything else is reported for sign-off.
Usage: python scratchpad/t8_template_sanity.py --pack ROOT --chain-list TXT --native-pack PACK --out DIR
"""

import argparse
import io
import json
import os
import random
import subprocess
import tempfile
from collections import Counter

import numpy as np
import torch

from proteinfoundation.utils.precompute_synthetic_topology_index import select_pack_rows, write_ca_pdb
from proteinfoundation.utils.t2_pack_reader import TemplatePack, layout

CA = 1
BROKEN_CA_CA = 4.0   # of-t2deep prune_work/t2_indel/t2_stage_b.py n_broken_steps (the generator's gate)
CLASH_CA_CA = 3.0    # reported only (no gate)


def n_broken_steps(ca, orig, nat_break):
    """t2_stage_b.n_broken_steps, verbatim logic: template CA-CA steps > 4.0 A, except steps joining two native-adjacent
    residues whose native step is itself broken."""
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    adj = (orig[:-1] >= 0) & (orig[1:] == orig[:-1] + 1)
    native_broken = np.zeros(len(d), bool)
    native_broken[adj] = nat_break[orig[:-1][adj]]
    return int(((d > BROKEN_CA_CA) & ~native_broken).sum())


def placeholder_ca(ca, ca_ok):
    """of-t2deep prune_work/t2_indel/t2_native.load_npz, CA part verbatim: a residue without a CA gets an interpolated
    placeholder (terminal runs extended along the last resolved CA-CA direction in 3.8 A steps) -- the native the
    generator computed its break flags on."""
    pos = ca.copy()
    if not ca_ok.all():
        idx = np.arange(len(ca_ok))
        ok = np.flatnonzero(ca_ok)
        for k in range(3):
            pos[~ca_ok, k] = np.interp(idx[~ca_ok], ok, pos[ok, k])
        for lo, hi, step in ((ok[0], ok[1] if len(ok) > 1 else ok[0], -1), (ok[-1], ok[-2] if len(ok) > 1 else ok[-1], 1)):
            d = pos[lo] - pos[hi]
            d = d / max(float(np.linalg.norm(d)), 1e-6)
            run = range(lo + step, -1 if step < 0 else len(ca_ok), step)
            for j in run:
                if ca_ok[j]:
                    break
                pos[j] = pos[lo] + d * 3.8 * abs(j - lo)
    return pos


def quantiles(x, qs=(0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)):
    x = np.asarray(x, dtype=np.float64)
    return {str(q): round(float(np.quantile(x, q)), 4) for q in qs} if x.size else {}


def usalign_tm(a_pdb, b_pdb, exe):
    out = subprocess.run([exe, a_pdb, b_pdb, "-outfmt", "2"], capture_output=True, text=True, check=True).stdout
    line = [l for l in out.splitlines() if l and not l.startswith("#")][0].split("\t")
    return float(line[3])  # TM normalised by the second structure (the native)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--chain-list", required=True)
    ap.add_argument("--native-pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-chains", type=int, default=500)
    ap.add_argument("--usalign-templates", type=int, default=300)
    ap.add_argument("--tm-min", type=float, default=0.5)
    ap.add_argument("--tm-max", type=float, default=0.9)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rep, fails = {}, []

    tp = TemplatePack(args.pack)
    c = tp.col
    want = [l.strip() for l in open(args.chain_list) if l.strip()]
    have = set(str(s) for s in tp.stems)
    missing = [s for s in want if s not in have]
    extra = sorted(have - set(want))
    rep["coverage"] = {"chain_list": len(want), "in_pack": len(want) - len(missing), "missing": len(missing),
                       "pack_chains_not_in_list": len(extra), "missing_examples": missing[:20]}

    L, A, nb, off, sh = (c[k].astype(np.int64) for k in ("L", "A", "nbytes", "off", "shard"))
    tm = c["tm_native"].astype(np.float64)
    n = len(L)
    exp_nb = np.array([layout(int(l), int(a))["nbytes"] for l, a in zip(L, A)], dtype=np.int64)
    bad_layout = int((exp_nb != nb).sum())
    bad_range = int(((L <= 0) | (A <= 0) | (A > 37 * L)).sum())
    bad_tm = int(((tm < 0) | (tm > 1) | ~np.isfinite(tm)).sum())
    overlap = out_of_file = 0
    for s in np.unique(sh):
        m = sh == s
        o, e = off[m], off[m] + nb[m]
        order = np.argsort(o)
        overlap += int((o[order][1:] < e[order][:-1]).sum())
        size = os.path.getsize(os.path.join(args.pack, f"tmpl.{int(s):03d}.bin"))
        out_of_file += int((e > size).sum())
    first, count = tp.first.astype(np.int64), tp.count.astype(np.int64)
    Ln = c["L_native"].astype(np.int64)
    multi_ln = dup_draw = 0
    draw = c["draw"].astype(np.int64)
    for f, k in zip(first, count):
        multi_ln += len(np.unique(Ln[f:f + k])) != 1
        dup_draw += len(np.unique(draw[f:f + k])) != k
    in_range = (tm >= args.tm_min) & (tm <= args.tm_max)
    per_in = np.add.reduceat(in_range.astype(np.int64), first) if n else np.zeros(0)
    rep["index"] = {"templates": n, "chains": len(first), "per_chain": quantiles(count), "in_tm_range": int(in_range.sum()),
                    "per_chain_in_range": quantiles(per_in), "chains_with_none_in_range": int((per_in == 0).sum()),
                    "after_cap": int(np.minimum(per_in, args.cap).sum()), "chains_capped": int((per_in > args.cap).sum()),
                    "tm_native": quantiles(tm), "L_minus_L_native": quantiles(L - Ln), "bad_layout": bad_layout,
                    "bad_L_A_range": bad_range, "bad_tm": bad_tm, "overlapping_records": overlap,
                    "records_past_shard_end": out_of_file, "chains_with_several_L_native": int(multi_ln),
                    "chains_with_duplicate_draws": int(dup_draw)}
    for k in ("bad_layout", "bad_L_A_range", "bad_tm", "overlapping_records", "records_past_shard_end",
              "chains_with_several_L_native", "chains_with_duplicate_draws"):
        if rep["index"][k]:
            fails.append(f"index.{k}={rep['index'][k]}")

    z = np.load(args.native_pack + ".idx.npz", allow_pickle=True)
    nat = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    nat_fh = open(args.native_pack, "rb")
    rng = random.Random(args.seed)
    pool = [s for s in want if s in have and s in nat]
    rep["coverage"]["in_pack_but_not_in_native_pack"] = sum(1 for s in want if s in have and s not in nat)
    sample = rng.sample(pool, min(args.sample_chains, len(pool)))
    st = Counter()
    broken_hist, clash_hist, ident, ln_mismatch, steps_all = Counter(), Counter(), [], [], []
    tm_pairs = []
    us_budget = args.usalign_templates
    with tempfile.TemporaryDirectory() as td:
        for stem in sample:
            o, ln = nat[stem]
            nat_fh.seek(o)
            g = torch.load(io.BytesIO(nat_fh.read(ln)), map_location="cpu", weights_only=False)
            L_nat = int(g.coords.shape[0])
            n_seq = np.asarray(g.residue_type)
            nca = placeholder_ca(g.coords[:, CA].numpy().astype(np.float32), (g.coord_mask[:, CA] > 0.5).numpy())
            nat_break = np.linalg.norm(nca[1:] - nca[:-1], axis=1) > BROKEN_CA_CA
            pq = None
            for r in select_pack_rows(tp, stem, args.tm_min, args.tm_max, args.cap) or []:
                rec = tp._read(r)
                st["templates"] += 1
                am, co = rec["atom_mask"], rec["coords"]
                Lt = am.shape[0]
                if not np.isfinite(co).all():
                    st["nonfinite_coords"] += 1
                if int(am.sum()) != co.shape[0]:
                    st["mask_count_ne_A"] += 1
                bb = am[:, 0] & am[:, 1] & am[:, 2]
                st["residues"] += Lt
                st["residues_missing_backbone"] += int((~bb).sum())
                ri = rec["residue_index"]
                if not np.array_equal(ri, np.arange(1, Lt + 1)):
                    st["residue_index_not_1_to_L"] += 1
                oi = rec["orig_idx"]
                if ((oi < -1) | (oi >= rec["L_native"])).any():
                    st["orig_idx_out_of_range"] += 1
                pos = oi[oi >= 0]
                if (np.diff(pos) <= 0).any():
                    st["orig_idx_not_increasing"] += 1
                if rec["L_native"] != L_nat:  # = a template of a different native (provenance), not a length gate
                    ln_mismatch.append((stem, rec["L_native"], L_nat))
                    st["templates_L_native_mismatch"] += 1
                full = np.zeros((Lt, 37, 3), dtype=np.float32)
                full[am] = co
                ca = full[:, CA]
                has_ca = am[:, CA]
                step = np.linalg.norm(ca[1:] - ca[:-1], axis=1)[has_ca[1:] & has_ca[:-1]]
                steps_all.append(step)
                if rec["L_native"] == L_nat:  # the generator's rule needs the matching native's break flags
                    nb_ = n_broken_steps(ca, oi, nat_break)
                    broken_hist[min(nb_, 5)] += 1
                    if nb_ > 1:
                        st["templates_badly_broken"] += 1
                cac = ca[has_ca]
                d = np.linalg.norm(cac[:, None] - cac[None], axis=-1)
                iu = np.triu_indices(len(cac), k=2)
                clash_hist[min(int((d[iu] < CLASH_CA_CA).sum()), 5)] += 1
                m = oi >= 0
                if m.any() and len(n_seq) == L_nat:
                    ident.append(float((rec["aatype"][m] == n_seq[oi[m]]).mean()))
                if us_budget > 0:
                    if pq is None:
                        pq = os.path.join(td, "q.pdb")
                        write_ca_pdb(nca, np.ones(len(nca), dtype=bool), pq)  # with placeholder CAs, as the generator
                    pr = os.path.join(td, "t.pdb")
                    write_ca_pdb(ca, has_ca, pr)
                    tm_pairs.append((rec["tm_native"], usalign_tm(pr, pq, args.usalign)))
                    us_budget -= 1
    steps = np.concatenate(steps_all) if steps_all else np.zeros(0)
    dtm = np.array([b - a for a, b in tm_pairs])
    rep["sample"] = {"chains": len(sample), **dict(st), "broken_steps_per_template": dict(sorted(broken_hist.items())),
                     "ca_clash_pairs_per_template": dict(sorted(clash_hist.items())),
                     "ca_ca_step": quantiles(steps), "L_native_vs_training_native_mismatch": len(ln_mismatch),
                     "L_native_mismatch_examples": ln_mismatch[:10], "seq_identity_at_orig_idx": quantiles(ident),
                     "usalign_n": len(tm_pairs), "usalign_tm_minus_stored": quantiles(dtm),
                     "usalign_abs_diff_gt_0.05": int((np.abs(dtm) > 0.05).sum())}
    for k in ("nonfinite_coords", "mask_count_ne_A", "residue_index_not_1_to_L", "orig_idx_out_of_range",
              "orig_idx_not_increasing", "templates_badly_broken"):
        if st[k]:
            fails.append(f"sample.{k}={st[k]}")
    if st["residues_missing_backbone"]:
        fails.append(f"sample.residues_missing_backbone={st['residues_missing_backbone']}")
    rep["hard_fail"] = fails
    rep["verdict"] = "FAIL" if fails else "PASS (hard checks); review the reported distributions for sign-off"
    with open(os.path.join(args.out, "template_sanity.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    print("TEMPLATE_SANITY " + ("FAIL" if fails else "PASS"))


if __name__ == "__main__":
    main()
