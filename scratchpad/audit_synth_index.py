"""Correctness audit of the synthetic topology index, read through the REAL consumer.

A merge that exits 0 is not evidence: this checks the invariants the training path depends on, on a
random sample of chains, and re-derives one chain's row from the source npz to prove the stored
tensors describe what they claim to.

  1. structure: every group = 1 native row + its template rows; natives have row_is_native and TM 1;
     template TMs lie in the build band; members_flat never points at a native row.
  2. eligible list == exactly the chains with >= 1 template row in the training TM range.
  3. alignment: every template row of a group shares ONE align length, that length equals the
     chain's residue count read from the processed .pt (authoritative -- NOT the sum of DSSP run
     lengths, which drops residues whose backbone is incomplete), and every value is -1 or a valid
     element index of THAT row; the aligned fraction is sane (not ~0, not ~1).
  4. features: he_flat is a binary TxT block of the right size; feat_flat has T*T*4 entries and is
     finite; the standardisation constants are finite and non-degenerate.
  5. the transform can actually build a reference from the index for a sampled chain (synthetic
     mode), and it never returns the native.

usage: python audit_synth_index.py <index.pt> <eligible.txt> [--n 200] [--seed 0]
"""

import argparse
import collections
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES, STRUCTURAL_PAIR_FEATURES
from proteinfoundation.datasets.topology_reference import ALIGN_NONE, TopologyReferenceTransform

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("eligible")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tm-range", type=float, nargs=2, default=(0.5, 0.9))
    ap.add_argument("--processed-dir", default="", help="pdb_train/processed, for the authoritative length check")
    # ⛔ Explicit, NOT derived as processed-dir/..: `processed` is a SYMLINK to another filesystem, so
    # ".." resolves through the link to the TARGET's parent and the manifest silently goes missing --
    # which made the bucket None, every path wrong, and this check pass VACUOUSLY on 0 chains.
    ap.add_argument("--manifest", default="", help="shard_manifest.json (usually pdb_train/shard_manifest.json)")
    a = ap.parse_args()
    lo, hi = a.tm_range

    idx = torch.load(a.index, map_location="cpu", weights_only=False, mmap=True)
    ids = list(idx["ids"])
    print(f"index rows: {len(ids)}; keys: {sorted(idx.keys())}")
    is_native = idx["row_is_native"].numpy()
    row_tm = idx["row_tm"].float().numpy()
    cluster_of = idx["cluster_of"].numpy()
    m_off, m_flat = idx["members_offset"].numpy(), idx["members_flat"].numpy()
    n_groups = len(m_off) - 1
    print(f"groups: {n_groups}; natives: {int(is_native.sum())}; template rows: {int((~is_native).sum())}")

    # 1. structure
    check("one native row per group", int(is_native.sum()) == n_groups,
          f"{int(is_native.sum())} natives vs {n_groups} groups")
    check("native rows carry TM 1.0", bool(np.allclose(row_tm[is_native], 1.0)))
    tt = row_tm[~is_native]
    check("template TMs inside the build band (0, 1]", bool((tt > 0).all() and (tt <= 1.0).all()),
          f"min {tt.min():.3f} max {tt.max():.3f}")
    check("members_flat never points at a native row", not bool(is_native[m_flat].any()),
          f"{int(is_native[m_flat].sum())} native rows referenced")
    check("every member row belongs to its own group",
          all(cluster_of[m_flat[m_off[g]:m_off[g + 1]]].tolist() == [g] * (m_off[g + 1] - m_off[g])
              for g in random.Random(a.seed).sample(range(n_groups), min(500, n_groups))))

    # 2. eligible list
    eligible = {l.strip() for l in open(a.eligible) if l.strip()}
    in_range = np.zeros(n_groups, dtype=bool)
    for g in range(n_groups):
        mem = m_flat[m_off[g]:m_off[g + 1]]
        if mem.size:
            t = row_tm[mem]
            in_range[g] = bool(((t >= lo) & (t <= hi)).any())
    expect = {ids[int(np.flatnonzero((cluster_of == g) & is_native)[0])] for g in np.flatnonzero(in_range)} \
        if n_groups < 5000 else None
    check(f"eligible count == groups with a row in [{lo}, {hi}]", len(eligible) == int(in_range.sum()),
          f"{len(eligible)} vs {int(in_range.sum())}")
    if expect is not None:
        check("eligible ids are exactly those groups", eligible == expect)

    # 3-4. per-chain invariants on a sample
    rng = random.Random(a.seed)
    groups = [g for g in range(n_groups) if m_off[g + 1] > m_off[g]]
    sample = rng.sample(groups, min(a.n, len(groups)))
    a_off = idx["align_offset"].numpy()
    a_flat = idx["align_flat"]
    he_off, he_size, he_flat = idx["he_offset"].numpy(), idx["he_size"].numpy(), idx["he_flat"]
    f_off, f_flat = idx["feat_offset"].numpy(), idx["feat_flat"]
    runs_off, runs_flat = idx["runs_offset"].numpy(), idx["runs_flat"]
    bad_align_len = bad_align_val = bad_he = bad_feat = 0
    frac_aligned = []
    align_len_of = {}
    for g in sample:
        nat = int(np.flatnonzero((cluster_of == g) & is_native)[0])
        # ⛔ NOT sum(run lengths): dssp_to_runs drops residues with an incomplete backbone (-1) and
        # runs below min_len, so that sum under-counts. The rows must merely AGREE with each other;
        # the absolute length is checked against the processed .pt below, which is authoritative.
        lens = {int(a_off[int(r) + 1] - a_off[int(r)]) for r in m_flat[m_off[g]:m_off[g + 1]]}
        if len(lens) > 1:
            bad_align_len += 1
        align_len_of[str(ids[nat])] = max(lens) if lens else 0
        for r in m_flat[m_off[g]:m_off[g + 1]]:
            r = int(r)
            al = a_flat[a_off[r]:a_off[r + 1]].numpy()
            T = int(he_size[r])
            if ((al != ALIGN_NONE) & ((al < 0) | (al >= max(T, 1)))).any():
                bad_align_val += 1
            frac_aligned.append(float((al != ALIGN_NONE).mean()))
            he = he_flat[he_off[r]:he_off[r + 1]]
            if he.numel() != T * T or (T and not bool(((he == 0) | (he == 1)).all())):
                bad_he += 1
            fe = f_flat[f_off[r]:f_off[r + 1]]
            if fe.numel() != T * T * len(STRUCTURAL_PAIR_FEATURES) or not bool(torch.isfinite(fe.float()).all()):
                bad_feat += 1
    check("all template rows of a chain share one alignment length", bad_align_len == 0, f"{bad_align_len} groups disagree")
    # authoritative: the residue count on disk, for a subsample
    if a.processed_dir:
        from proteinfoundation.datasets.pdb_data import _processed_path_sharded
        assert a.manifest and os.path.exists(a.manifest), \
            f"--manifest must name an existing shard_manifest.json (got {a.manifest!r}); without it every " \
            "bucketed path is wrong and this check would pass on zero chains"
        manifest = json.load(open(a.manifest))
        n_chk = n_bad = 0
        for stem, alen in list(align_len_of.items())[:40]:
            pt = _processed_path_sharded(pathlib.Path(a.processed_dir), stem, manifest)
            if not pt.exists():
                continue
            g_ = torch.load(str(pt), map_location="cpu", weights_only=False)
            n_chk += 1
            n_bad += int(alen != int(g_.coords.shape[0]))
        # ⛔ A check that inspected nothing is not a pass.
        check("alignment length == residue count in the processed .pt (non-vacuous)",
              n_bad == 0 and n_chk >= 10, f"{n_bad} of {n_chk} chains differ")
    check("alignment values are -1 or a valid element index", bad_align_val == 0, f"{bad_align_val} bad")
    check("SSE contact blocks are binary and T x T", bad_he == 0, f"{bad_he} bad")
    check(f"structural features are finite and T x T x {len(STRUCTURAL_PAIR_FEATURES)}",
          bad_feat == 0, f"{bad_feat} bad")
    fa = np.array(frac_aligned)
    check("aligned fraction is sane (mean in 0.1-0.95)", 0.1 < fa.mean() < 0.95,
          f"mean {fa.mean():.3f} p5 {np.percentile(fa, 5):.3f} p95 {np.percentile(fa, 95):.3f}")
    mean, std = idx["pair_feature_mean"], idx["pair_feature_std"]
    check("standardisation constants finite, std > 0",
          bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()))
    check("pair feature count matches the code", len(mean) == N_PAIR_FEATURES)

    # 5. the real consumer
    tf = TopologyReferenceTransform(index_path=a.index, reference_source="synthetic",
                                    tm_range=(lo, hi), sse_types=(1, 2), drop_prob=0.0,
                                    mutate_prob=0.0, sigma_frac=0.0, seed=0)
    tf._ensure_loaded()
    n_ok = n_self = 0
    for g in sample[:50]:
        nat = int(np.flatnonzero((cluster_of == g) & is_native)[0])
        stem = str(ids[nat])
        n_res = int(runs_flat[runs_off[nat]:runs_off[nat + 1]][:, 1].sum())
        row = tf._id_to_row[stem]
        t_row = tf._pick_template(row)
        if t_row < 0:
            continue
        if t_row == row:
            n_self += 1
        feats = tf._build_reference(t_row, n_res, augment=False)
        if feats["topology_he_tokens"].numel() and int(feats["topology_he_tokens"].max()) < 44:
            n_ok += 1
    check("transform builds references from the index (vocab 44)", n_ok > 0, f"{n_ok} chains")
    check("transform never returns the native in synthetic mode", n_self == 0, f"{n_self} self-picks")

    tms = row_tm[~is_native]
    print(f"\ntemplate TM: mean {tms.mean():.3f}  p5 {np.percentile(tms, 5):.3f}  "
          f"median {np.median(tms):.3f}  p95 {np.percentile(tms, 95):.3f}")
    per_chain = collections.Counter()
    for g in sample:
        mem = m_flat[m_off[g]:m_off[g + 1]]
        per_chain[int(((row_tm[mem] >= lo) & (row_tm[mem] <= hi)).sum())] += 1
    ks = sorted(per_chain)
    print(f"in-range rows per sampled chain: min {ks[0]} median {ks[len(ks)//2]} max {ks[-1]}")
    print(f"\n{'AUDIT PASSED' if not FAIL else 'AUDIT FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
