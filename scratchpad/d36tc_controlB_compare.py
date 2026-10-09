"""D36tc CONTROL B compare (CPU): my reruns vs the T7-NEWEST records. No tolerance was pre-registered: the verdict is
EXACT or DELTAS, and DELTAS is NOT a pass (it goes back to the orchestrator with the evidence).

--tri  STEM:MINE_DIR:RECORD_DIR   tri maps of STEM (all its samples): contact_prob bit-exact or max|dp|; contact_gt and
                                  ref_id must be equal (else structural FAIL).
--c2c  MINE_JSONL:RECORD_JSONL    every (stem, sample_index) row of the chunk: tm, rmsd_proper deltas; torch_seed equal.
"""
import argparse
import json
import os
import sys

import numpy as np


def load(path, stem=None):
    rs = [json.loads(l) for l in open(path) if l.strip()]
    return {(r["stem"], r["sample_index"]): r for r in rs if stem is None or r["stem"] == stem}


ap = argparse.ArgumentParser()
ap.add_argument("--tri", action="append", default=[])
ap.add_argument("--c2c", action="append", default=[])
args = ap.parse_args()

structural, exact_all, n_tri, n_c2c = [], True, 0, 0
for spec in args.tri:
    stem, mdir, rdir = spec.split(":")
    mt, rt = load(os.path.join(mdir, "samples.jsonl"), stem), load(os.path.join(rdir, "samples.jsonl"), stem)
    if sorted(mt) != sorted(rt) or not mt:
        structural.append(f"tri {stem}: samples mine {sorted(k[1] for k in mt)} record {sorted(k[1] for k in rt)}")
        continue
    for key in sorted(mt):
        a = np.load(os.path.join(mdir, mt[key]["file"]))
        b = np.load(os.path.join(rdir, rt[key]["file"]))
        same_shape = a["contact_prob"].shape == b["contact_prob"].shape
        exact = same_shape and bool(np.array_equal(a["contact_prob"], b["contact_prob"]))
        dp = float(np.abs(a["contact_prob"] - b["contact_prob"]).max()) if same_shape else float("inf")
        gt = bool(np.array_equal(a["contact_gt"], b["contact_gt"]))
        ref = mt[key]["ref_id"] == rt[key]["ref_id"]
        exact_all &= exact
        n_tri += 1
        print(f"  tri {stem} s{key[1]:02d}: contact_prob {'EXACT' if exact else f'max|dp| {dp:.3g}'}  "
              f"gt {'=' if gt else 'DIFF'}  ref_id {mt[key]['ref_id']}{'' if ref else ' != ' + str(rt[key]['ref_id'])}")
        if not (gt and ref):
            structural.append(f"tri {stem} s{key[1]:02d}: gt {gt} ref_id {ref}")
for spec in args.c2c:
    mpath, rpath = spec.split(":")
    mc, rc = load(mpath), load(rpath)
    if sorted(mc) != sorted(rc) or not mc:
        structural.append(f"c2c {mpath}: {len(mc)} rows vs record {len(rc)} (keys differ)")
        continue
    dtm = np.array([mc[k]["tm"] - rc[k]["tm"] for k in sorted(mc)])
    drm = np.array([mc[k]["rmsd_proper"] - rc[k]["rmsd_proper"] for k in sorted(mc)])
    seeds = sum(mc[k]["torch_seed"] != rc[k]["torch_seed"] for k in mc)
    n_c2c += len(mc)
    exact_all &= bool((dtm == 0).all() and (drm == 0).all())
    print(f"  c2c {os.path.basename(mpath)}: {len(mc)} rows; tm exact {int((dtm == 0).sum())}, max|dtm| {np.abs(dtm).max():.4g}; "
          f"rmsd exact {int((drm == 0).sum())}, max|drmsd| {np.abs(drm).max():.4g}; torch_seed mismatches {seeds}")
    if seeds:
        structural.append(f"c2c {mpath}: {seeds} torch_seed mismatches")
print(f"[controlB] tri samples compared {n_tri}, c2c rows compared {n_c2c}; structural problems {len(structural)}: {structural}")
verdict = "FAIL" if structural or not (n_tri or n_c2c) else ("EXACT" if exact_all else "DELTAS (not a pass)")
print(f"CONTROL_B {verdict}")
sys.exit(0 if verdict == "EXACT" else 1)
