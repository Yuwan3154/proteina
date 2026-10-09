"""Native DSSP check for the proline-donor-mask rebuild (user 2026-10-09), on graphs read straight from the pack.

(1) control: compute_dssp_target(disk order, no donor mask) must reproduce the stored dssp_target EXACTLY on every residue
    with a complete N/CA/C/O backbone (it is how the synthetic-index builder recomputes the native under
    --proline-donor-mask). Stored labels on an INCOMPLETE backbone (e.g. CA-only chains such as 1a1q, whose stored
    labels came from the 1e-5 fill coordinates against precompute_dssp_targets' own -1 rule) are counted, not gated;
(2) the donor mask's effect (changed labels, and how many sit within 4 residues of a proline);
(3) writes the masked labels per stem, so two hosts / pydssp versions can be compared with --compare.

  python scripts/sse1d_check_native_dssp.py --pack $PACK_PATH --n 500 --out labels_<host>.npz
  python scripts/sse1d_check_native_dssp.py --compare labels_a.npz labels_b.npz
"""

import argparse
import io
import json
import mmap
from importlib.metadata import version

import numpy as np
import torch

from proteinfoundation.utils.dssp_utils import compute_dssp_target

PRO_IDX = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        x, y = np.load(a.compare[0]), np.load(a.compare[1])
        assert sorted(x.files) == sorted(y.files), "different stems"
        diff = sum(int((x[k] != y[k]).sum()) for k in x.files)
        print(json.dumps({"event": "compare", "n_chains": len(x.files), "n_res": sum(x[k].size for k in x.files),
                          "label_differences": diff}))
        return
    z = np.load(a.pack + ".idx.npz", allow_pickle=True)
    order = np.argsort(z["stems"].astype(str))[: a.n]  # same stems on every host
    fh = open(a.pack, "rb")  # keep the file object alive: mmap needs its descriptor
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    out, n_res, n_ctrl_diff, n_mask_diff, n_near, n_pro, n_bad_bb, chains_bad_bb = {}, 0, 0, 0, 0, 0, 0, set()
    for i in order:
        o, l = int(z["offsets"][i]), int(z["lengths"][i])
        g = torch.load(io.BytesIO(mm[o:o + l]), map_location="cpu", weights_only=False)
        L = int(g.coords.shape[0])
        args = (g.coords[None].float(), torch.ones(1, L, dtype=torch.bool))
        kw = dict(coord_mask=(g.coord_mask.float() > 0.5)[None], coord_layout="pdb")
        ctrl = compute_dssp_target(*args, **kw)[0]
        pro = g.residue_type == PRO_IDX
        masked = compute_dssp_target(*args, **kw, donor_mask=(~pro)[None])[0]
        n_res += L
        bb = (g.coord_mask[:, :4].float() > 0.5).all(1)
        n_ctrl_diff += int(((ctrl != g.dssp_target) & bb).sum())
        stale = (g.dssp_target >= 0) & ~bb
        n_bad_bb += int(stale.sum())
        if stale.any():
            chains_bad_bb.add(str(z["stems"][i]))
        d = masked != ctrl
        near = torch.zeros_like(pro)
        for sh in range(-4, 5):
            near |= torch.roll(pro, sh)
        n_mask_diff += int(d.sum()); n_near += int((d & near).sum()); n_pro += int(pro.sum())
        out[str(z["stems"][i])] = masked.numpy().astype(np.int8)
    rec = {"event": "native_dssp", "pydssp": version("pydssp"), "n_chains": len(out), "n_res": n_res, "n_pro": n_pro,
           "control_mismatches_vs_stored_complete_backbone": n_ctrl_diff,
           "stored_labels_on_incomplete_backbone": n_bad_bb, "chains_with_those": sorted(chains_bad_bb),
           "mask_changed": n_mask_diff, "mask_changed_within4_of_pro": n_near}
    print(json.dumps(rec))
    assert n_ctrl_diff == 0, rec
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
