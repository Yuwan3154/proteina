"""D36tc: a natives-only topology index in the CURRENT synthetic (8-feature) format, ConFind(F2C) contacts.

Every listed chain becomes one NATIVE row built by the training builder itself
(precompute_synthetic_topology_index._chain_job with --contact-source confind and no template npz), so runs,
contact/structural bytes and circuit/gap features come from the same code that built the cfft index. The
rows are merged by the builder's own cmd_merge, then the standardisation constants are REPLACED by the
cfft training index's (--stats-from): a natives-only merge has no template rows to compute them from, and
small eval sets must never be standardised by their own statistics (D36m, 0.31 sigma).

Fails loudly if any chain yields no native row or any skip other than the expected "no_npz".
"""
import argparse
import multiprocessing
import os
import tempfile
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor

import torch

import proteinfoundation.utils.precompute_synthetic_topology_index as psti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="flat dir of <stem>.pt, or a file of '<stem>\\t<path>' lines")
    ap.add_argument("--stems", required=True)
    ap.add_argument("--stats-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    stems = [ln.split()[0] for ln in open(args.stems) if ln.strip()]
    assert len(stems) == len(set(stems)), "duplicate stems"
    if os.path.isdir(args.processed_dir):
        paths = {s: os.path.join(args.processed_dir, f"{s}.pt") for s in stems}
    else:
        paths = dict(ln.rstrip("\n").split("\t") for ln in open(args.processed_dir) if ln.strip())
    missing = [s for s in stems if not os.path.exists(paths.get(s, ""))]
    assert not missing, f"{len(missing)} .pt missing: {missing[:10]}"

    psti._CONTACT_SOURCE = "confind"  # module global; fork context below makes workers inherit it
    jobs = [(s, paths[s], None, None, None, None, 1, 0.0, 1.0, False) for s in stems]
    chains, bad = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as ex:
        for stem, native, rows, skips, seq in ex.map(psti._chain_job, jobs):
            if native is None or rows or skips != [("templates", "no_npz")]:
                bad.append((stem, native is None, len(rows), skips))
            chains.append({"stem": stem, "native": native, "rows": rows, "seq_hash": psti._hash_sequence(seq, stem)})
    assert not bad, f"{len(bad)} chains failed: {bad[:10]}"
    print(f"built {len(chains)} native rows (min_len 1, contact {psti.CONTACT_DEF_CONFIND})", flush=True)

    with tempfile.TemporaryDirectory() as td:
        torch.save({"chains": chains, "n_parts": 1, "part": 0, "min_len": 1, "tm_build_range": (0.0, 1.0),
                    "contact_def": psti.CONTACT_DEF_CONFIND}, os.path.join(td, "part_0000.pt"))
        merged = os.path.join(td, "merged.pt")
        psti.cmd_merge(Namespace(parts_dir=td, n_parts=1, out=merged, eligible_out="", tm_range=(0.5, 0.9)))
        idx = torch.load(merged, map_location="cpu", weights_only=False)

    ref = torch.load(args.stats_from, map_location="cpu", weights_only=False, mmap=True)
    assert list(ref["pair_feature_names"]) == list(idx["pair_feature_names"]), \
        f"feature names differ: {ref['pair_feature_names']} vs {idx['pair_feature_names']}"
    assert ref["contact_def"] == idx["contact_def"], f"contact_def differs: {ref['contact_def']!r} vs {idx['contact_def']!r}"
    assert ref["min_len"] == idx["min_len"], f"min_len differs: {ref['min_len']} vs {idx['min_len']}"
    idx["pair_feature_mean"] = ref["pair_feature_mean"].clone().float()
    idx["pair_feature_std"] = ref["pair_feature_std"].clone().float()
    idx["stats_from"] = os.path.abspath(args.stats_from)
    assert list(idx["ids"]) == stems, "merged row order differs from the stem list"
    torch.save(idx, args.out)
    print(f"stats copied from {args.stats_from}:")
    for n, m, s in zip(idx["pair_feature_names"], idx["pair_feature_mean"].tolist(), idx["pair_feature_std"].tolist()):
        print(f"  {n:<24} mean={m:10.4f} std={s:10.4f}")
    print(f"wrote {args.out}: {len(idx['ids'])} rows")
    print("D36TC_INDEX_DONE")


if __name__ == "__main__":
    main()
