"""Diagnose regression stems whose flag-OFF re-processing differs from the stored pack entry (repair plan S3 failure, job 26561038).
Per stem: differing (residue, atom) count, max |d|, the residues involved with their atom names, raw mmCIF file mtime vs the pack entry, and per differing atom
every raw atom_site row of that residue/atom (altloc, occupancy, xyz) with which row the old and the new value equal.
Usage: python scratchpad/diag_seqres_regression.py --pack PACK --data-dir DATA --out-dir OUT stem [stem ...]
"""

import argparse
import datetime
import json
import os
import pathlib

import numpy as np
import torch
from biopandas.mmcif import PandasMmcif

from proteinfoundation.utils.constants import ATOM_NUMBERING
from proteinfoundation.utils.reprocess_seqres_chains import load_entry, open_pack, raw_path, reprocess_chain

ATOM_NAME = {v: k for k, v in ATOM_NUMBERING.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("stems", nargs="+")
    a = ap.parse_args()
    ent, mm, _ = open_pack(a.pack)
    data = pathlib.Path(a.data_dir)
    out = pathlib.Path(a.out_dir).resolve()
    assert out != data.resolve() and data.resolve() not in out.parents, "--out-dir must be outside --data-dir"
    man = json.load(open(data / "shard_manifest.json"))
    staging = pathlib.Path(a.out_dir) / "staging" / "processed"
    print(f"pack {a.pack} mtime {datetime.datetime.fromtimestamp(os.path.getmtime(a.pack)):%Y-%m-%d %H:%M}")
    for stem in a.stems:
        pdb, chain = stem.split("_", 1)
        old = load_entry(ent, mm, stem)
        status, g = reprocess_chain(stem, data / "raw", staging, man, False, old)
        path = raw_path(data / "raw", pdb, man)
        if g is None or path is None:
            print(f"\n== {stem}: {status}; raw {path}: nothing to compare")
            continue
        print(f"\n== {stem}: {status}; raw {path} mtime {datetime.datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}; L old {len(old.residue_id)} new {len(g.residue_id)}")
        assert list(old.residue_id) == list(g.residue_id), f"{stem}: residue_id lists differ"
        mo, mn = old.coord_mask.bool(), g.coord_mask.bool()
        d = (old.coords - g.coords).abs().amax(-1)
        bad = (mo != mn) | ((mo & mn) & (d > 0))
        r, k = torch.nonzero(bad, as_tuple=True)
        off = ~(mo | mn)
        no, nn = int(torch.isnan(old.coords).any(-1).sum()), int(torch.isnan(g.coords).any(-1).sum())
        nb = int((torch.isnan(old.coords) & torch.isnan(g.coords)).any(-1)[off].sum())
        print(f"  unresolved slots {int(off.sum())} (NaN in both {nb}); NaN slots overall old {no} new {nn}; differing unresolved slots {int((off & ~(d == 0)).sum())} "
              f"(old values {torch.unique(old.coords[off]).tolist()[:4]}, new {torch.unique(g.coords[off]).tolist()[:4]}); dtype {old.coords.dtype}/{g.coords.dtype}")
        print(f"  differing (residue, atom): {len(r)} of {int((mo | mn).sum())}; mask-only {int((mo != mn).sum())}; max |d| {float(d[mo & mn].max()) if bool((mo & mn).any()) else float('nan'):.3f} A")
        atoms = PandasMmcif().read_mmcif(str(path)).df["ATOM"]
        atoms = atoms[(atoms["label_asym_id"].astype(str) == chain) & (atoms["pdbx_PDB_model_num"].astype(int) == 1)]   # processing reads get_model(1)
        for i, j in list(zip(r.tolist(), k.tolist()))[:12]:
            rid, name = old.residue_id[i], ATOM_NAME[j]
            _, resn, num, _ = (rid.split(":") + [""])[:4]
            rows = atoms[(atoms["auth_seq_id"].astype(str) == num) & (atoms["auth_atom_id"] == name) & (atoms["auth_comp_id"] == resn)]   # processing's atom_name / residue_name columns
            o, n = old.coords[i, j].numpy(), g.coords[i, j].numpy()
            print(f"  {rid} {name}: old {np.round(o, 3).tolist()} (mask {bool(mo[i, j])}) new {np.round(n, 3).tolist()} (mask {bool(mn[i, j])}); raw rows {len(rows)}")
            for _, row in rows.iterrows():
                xyz = np.array([row["Cartn_x"], row["Cartn_y"], row["Cartn_z"]], float)
                tag = "=old" if np.allclose(xyz, o, atol=1e-3) else ""
                tag += "=new" if np.allclose(xyz, n, atol=1e-3) else ""
                print(f"      alt {row['label_alt_id']} occ {row['occupancy']} ins {row['pdbx_PDB_ins_code']} comp {row['auth_comp_id']} model {row['pdbx_PDB_model_num']} xyz {xyz.tolist()} {tag}")


if __name__ == "__main__":
    main()
