"""D36tc CONTROL B compare: my rerun vs the T7-NEWEST records, per stem and tri sample.

tri : contact_prob (max |dp|, bit-exact?), contact_gt (exact), ref_id (exact) of <stem>_sKK.npz vs the record dump.
c2c : tm, rmsd_proper, torch_seed of each (stem, sample_index) row vs the record jsonl.
No tolerance is set (none was pre-registered): the verdict is EXACT or the deltas, reported in full.
"""
import argparse
import json
import os
import sys

import numpy as np


def rows(path, stem):
    return {r["sample_index"]: r for r in (json.loads(l) for l in open(path) if l.strip()) if r["stem"] == stem}


ap = argparse.ArgumentParser()
ap.add_argument("--mine", required=True)
ap.add_argument("--pair", action="append", required=True, help="stem:record_tri_dir:record_c2c_jsonl")
args = ap.parse_args()

problems, all_exact = [], True
for spec in args.pair:
    stem, rtri, rc2c = spec.split(":")
    mdir = os.path.join(args.mine, f"tri_{stem}")
    mt = rows(os.path.join(mdir, "samples.jsonl"), stem)
    rt = rows(os.path.join(rtri, "samples.jsonl"), stem)
    if sorted(mt) != list(range(8)) or sorted(rt) != list(range(8)):
        problems.append(f"{stem}: tri samples mine {sorted(mt)} record {sorted(rt)}")
        continue
    for k in range(8):
        a = np.load(os.path.join(mdir, mt[k]["file"]))
        b = np.load(os.path.join(rtri, rt[k]["file"]))
        dp = float(np.abs(a["contact_prob"] - b["contact_prob"]).max()) if a["contact_prob"].shape == b["contact_prob"].shape else float("inf")
        exact = bool(np.array_equal(a["contact_prob"], b["contact_prob"]))
        gt = bool(np.array_equal(a["contact_gt"], b["contact_gt"]))
        ref = mt[k]["ref_id"] == rt[k]["ref_id"]
        all_exact &= exact
        print(f"  tri {stem} s{k:02d}: contact_prob {'EXACT' if exact else f'max|dp| {dp:.3g}'}  gt {'=' if gt else 'DIFF'}  "
              f"ref_id {mt[k]['ref_id']} {'=' if ref else '!= ' + str(rt[k]['ref_id'])}")
        if not (gt and ref):
            problems.append(f"{stem} s{k:02d}: gt {gt} ref_id {ref}")
    mc = rows(os.path.join(args.mine, f"B_{stem}.jsonl"), stem)
    rc = rows(rc2c, stem)
    if sorted(mc) != list(range(8)) or sorted(rc) != list(range(8)):
        problems.append(f"{stem}: c2c rows mine {sorted(mc)} record {sorted(rc)}")
        continue
    for k in range(8):
        dtm = mc[k]["tm"] - rc[k]["tm"]
        drm = mc[k]["rmsd_proper"] - rc[k]["rmsd_proper"]
        seed = mc[k]["torch_seed"] == rc[k]["torch_seed"]
        all_exact &= dtm == 0 and drm == 0
        print(f"  c2c {stem} s{k:02d}: tm {mc[k]['tm']:.4f} vs {rc[k]['tm']:.4f} (d {dtm:+.4f})  "
              f"rmsd d {drm:+.3f}  seed {'=' if seed else 'DIFF'}")
        if not seed:
            problems.append(f"{stem} s{k:02d}: torch_seed {mc[k]['torch_seed']} vs {rc[k]['torch_seed']}")
print(f"[controlB] structural problems {len(problems)}: {problems}")
print("CONTROL_B " + ("EXACT" if all_exact and not problems else ("DELTAS (see rows)" if not problems else "FAIL")))
sys.exit(1 if problems else 0)
