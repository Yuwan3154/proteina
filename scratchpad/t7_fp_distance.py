"""T7 (b) (user 2026-09-30): how far apart in the NATIVE structure are each tri's false contacts?

For every sampled map (npz from the Stage-A dump dirs): binarise contact_prob at 0.5 (what the c2c consumes), take pairs
with sequence separation >= 6 in the upper triangle, split into TP / FP / FN against contact_gt, and measure each pair's
CB-CB distance in the native structure (CA where CB is absent). Writes one row per sample (FP/TP/FN counts and FP/FN
distance quantiles) as JSONL, joinable to the stageB rows by (stem, sample_index), and prints per-definition summaries.
No distance threshold is imposed: distributions are reported as quantiles.

Usage: python scratchpad/t7_fp_distance.py OUT.jsonl LABEL=dumpdir[,dumpdir...] [LABEL=...]
"""

import glob
import json
import pathlib
import sys

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.datasets.transforms import ContactMapTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

DATA = pathlib.Path("/orcd/pool/006/chenxiou/proteina/data/pdb_train")
SEP_MIN = 6
QS = (10, 25, 50, 75, 90)


def native_cb(stem, man):
    """CB exactly as the dataset's ContactMapTransform places it: the raw .pt is in PDB atom order (index 3 = O), so
    reorder to OpenFold (index 3 = CB) as pdb_data.py does, then fill missing CB with the pseudo-CB."""
    g = torch.load(_processed_path_sharded(DATA / "processed", stem, man), weights_only=False)
    x = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
    m = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].float()
    cb = ContactMapTransform._fill_missing_cb_pseudo(x[:, 3, :].clone(), x, m, m[:, 3] < 0.5)
    return cb.numpy(), (m[:, 1] >= 0.5).numpy()


def qd(d):
    return {f"q{q}": float(np.percentile(d, q)) for q in QS} if len(d) else {f"q{q}": None for q in QS}


def main():
    out = sys.argv[1]
    arms = {a.split("=", 1)[0]: a.split("=", 1)[1].split(",") for a in sys.argv[2:]}
    man = json.load(open(DATA / "shard_manifest.json"))
    cache, rows = {}, []
    for label, dirs in arms.items():
        files = sorted(f for d in dirs for f in glob.glob(f"{d}/*_s[0-9][0-9].npz"))
        for f in files:
            z = np.load(f)
            stem, k, L = str(z["stem"]), int(z["sample_index"]), int(z["L"])
            if stem not in cache:
                cache[stem] = native_cb(stem, man)
            cb, ok = cache[stem]
            assert len(cb) >= L, f"{stem}: native {len(cb)} < L {L}"
            cb, ok = cb[:L], ok[:L]
            D = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
            i, j = np.triu_indices(L, SEP_MIN)
            keep = ok[i] & ok[j]
            i, j = i[keep], j[keep]
            pred = z["contact_prob"][i, j] > 0.5
            gt = z["contact_gt"][i, j] > 0.5
            d = D[i, j]
            if label == "CB8":  # correctness gate: the CB-8 native map must be exactly CB <= 8 A on these coordinates
                assert np.array_equal(d <= 8.0, gt), f"{stem}: rebuilt CB<=8 map != dumped CB-8 native map"
            fp, tp, fn = pred & ~gt, pred & gt, ~pred & gt
            rows.append(dict(label=label, stem=stem, sample_index=k, L=L, n_pairs=int(len(d)), n_tp=int(tp.sum()),
                             n_fp=int(fp.sum()), n_fn=int(fn.sum()), fp_dist=qd(d[fp]), tp_dist=qd(d[tp]),
                             fn_dist=qd(d[fn]), gt_contact_dist=qd(d[gt])))
        print(f"[{label}] {len(files)} maps from {len(dirs)} dirs", flush=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for label in arms:
        rs = [r for r in rows if r["label"] == label]
        allfp = np.array([r["fp_dist"]["q50"] for r in rs if r["fp_dist"]["q50"] is not None])
        print(f"[{label}] samples {len(rs)}; FP per sample median {np.median([r['n_fp'] for r in rs]):.0f}; "
              f"per-sample median FP CB distance: median {np.median(allfp):.2f} A (q25 {np.percentile(allfp, 25):.2f}, "
              f"q75 {np.percentile(allfp, 75):.2f}); native-contact CB distance median "
              f"{np.median([r['gt_contact_dist']['q50'] for r in rs if r['gt_contact_dist']['q50'] is not None]):.2f} A")
    print(f"WROTE {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
