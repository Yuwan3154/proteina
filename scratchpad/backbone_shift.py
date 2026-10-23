"""Size of the backbone changes of the T8b re-process (all-stems run): per changed stem with backbone_changed, the old stored entry vs the
new .pt over matched residues (sidecar old_to_new index >= 0): max and median N/CA/C/O displacement, residues with any backbone shift, mask
changes, length change; flags template-list membership. Rows = stem. Summary printed for all and for template chains.
Usage: python scratchpad/backbone_shift.py --pack PACK --data-dir DATA --out-root OUT --templates template_chains.txt --out shift.tsv [--workers 32]
"""

import argparse
import glob
import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.utils.reprocess_seqres_chains import load_entry, open_pack

G = {}


def init(pack, out_root, data_dir):
    G["ent"], G["mm"], _ = open_pack(pack)
    G["out"] = pathlib.Path(out_root)
    G["man"] = json.load(open(pathlib.Path(data_dir) / "shard_manifest.json"))


def run(job):
    stem, idx = job
    old = load_entry(G["ent"], G["mm"], stem)
    new = torch.load(_processed_path_sharded(G["out"] / "processed", stem, G["man"]), weights_only=False)
    ok = idx >= 0
    oi, ni = np.flatnonzero(ok), idx[ok]
    oc, nc = old.coords[oi, :4].numpy(), new.coords[ni, :4].numpy()
    om, nm = old.coord_mask[oi, :4].numpy().astype(bool), new.coord_mask[ni, :4].numpy().astype(bool)
    both = om & nm
    d = np.where(both, np.linalg.norm(oc - nc, axis=-1), 0.0)
    return dict(stem=stem, L_old=len(old.residue_id), L_new=len(new.residue_id), n_unmatched_old=int((~ok).sum()), n_unmatched_new=len(new.residue_id) - len(set(ni.tolist())),
                max_shift=float(d.max()) if d.size else 0.0, median_shift_moved=float(np.median(d[d > 0])) if (d > 0).any() else 0.0,
                n_res_moved=int((d > 0).any(1).sum()), n_res_mask_changed=int((om != nm).any(1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args()
    st = pd.concat([pd.read_csv(f, sep="\t", dtype={"fields": str}) for f in sorted(glob.glob(os.path.join(a.out_root, "status", "*.tsv")))], ignore_index=True)
    stems = sorted(st[(st.status == "changed") & (st.backbone_changed.astype(str) == "True")].stem)
    assert stems, "no changed + backbone_changed stem (was the run made with --write-changed-only?)"
    want = set(stems)
    o2n = {}
    for f in glob.glob(os.path.join(a.out_root, "sidecars", "*.pt")):
        o2n.update({k: np.asarray(v["index"]) for k, v in torch.load(f, weights_only=False)["old_to_new"].items() if k in want})
    man = json.load(open(pathlib.Path(a.data_dir) / "shard_manifest.json"))
    assert want <= set(o2n), f"{len(want - set(o2n))} stems without an old_to_new sidecar entry"
    missing = [s for s in stems if not _processed_path_sharded(pathlib.Path(a.out_root) / "processed", s, man).is_file()]
    assert not missing, f"{len(missing)} stems without a new .pt, e.g. {missing[:3]}"
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.pack, a.out_root, a.data_dir)) as ex:
        res = pd.DataFrame(list(ex.map(run, [(s, o2n[s]) for s in stems], chunksize=16)))
    assert len(res) == len(stems), (len(res), len(stems))
    tmpl = set(l.strip() for l in open(a.templates) if l.strip())
    res["template"] = res.stem.isin(tmpl)
    res.to_csv(a.out, sep="\t", index=False)
    print("L_new = len(residue_id) of the new .pt; shift bins below are descriptive only")
    for name, sel in (("all", res), ("template chains", res[res.template])):
        if not len(sel):
            print(f"{name}: 0 backbone-changed stems")
            continue
        v = sel.max_shift
        print(f"{name}: {len(sel)} backbone-changed stems; max backbone shift q50 {v.quantile(.5):.3f} q90 {v.quantile(.9):.3f} q99 {v.quantile(.99):.3f} max {v.max():.3f} A; "
              f"length changed {int((sel.L_old != sel.L_new).sum())}; mask changed {int((sel.n_res_mask_changed > 0).sum())}; "
              f"max shift < 0.1 A {int((v < 0.1).sum())}, < 0.5 A {int((v < 0.5).sum())}, < 1 A {int((v < 1).sum())}")


if __name__ == "__main__":
    main()
