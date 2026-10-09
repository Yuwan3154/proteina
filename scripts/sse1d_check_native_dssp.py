"""Native DSSP check for the proline-donor-mask rebuild (user 2026-10-09), on graphs read straight from the pack.

(1) control: compute_dssp_target(disk order, no donor mask) must reproduce the stored dssp_target EXACTLY (it is how the
    synthetic-index builder recomputes the native under --proline-donor-mask);
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
    mm = mmap.mmap(open(a.pack, "rb").fileno(), 0, access=mmap.ACCESS_READ)
    out, n_res, n_ctrl_diff, n_mask_diff, n_near, n_pro = {}, 0, 0, 0, 0, 0
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
        n_ctrl_diff += int((ctrl != g.dssp_target).sum())
        d = masked != ctrl
        near = torch.zeros_like(pro)
        for sh in range(-4, 5):
            near |= torch.roll(pro, sh)
        n_mask_diff += int(d.sum()); n_near += int((d & near).sum()); n_pro += int(pro.sum())
        out[str(z["stems"][i])] = masked.numpy().astype(np.int8)
    rec = {"event": "native_dssp", "pydssp": version("pydssp"), "n_chains": len(out), "n_res": n_res, "n_pro": n_pro,
           "control_mismatches_vs_stored": n_ctrl_diff, "mask_changed": n_mask_diff, "mask_changed_within4_of_pro": n_near}
    print(json.dumps(rec))
    assert n_ctrl_diff == 0, rec
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
