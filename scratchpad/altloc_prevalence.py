"""How the stored processed entries resolved altloc atoms (protein_df_to_tensor scatters every altloc row into one slot; duplicate-index
index_put is undefined). Rows = altloc atom = (residue, atom name) of model 1 with > 1 atom_site row, for the regression's arm-A chains (status ok,
no insertion codes, so residue_id 'chain:resname:num:' maps 1:1 to (auth_seq_id, auth_comp_id)).
Per atom: stored value equals one row exactly ('pure', which altloc), else 'chimera' (per-component mix); distance from the stored value to the nearest
row; spread between rows; the AF2 choice (Bio.PDB: highest occupancy, ties -> first row) and whether the stored value equals it.
Usage: python scratchpad/altloc_prevalence.py --pack PACK --data-dir DATA --regression REG.json --out OUT.tsv [--workers 32]
"""

import argparse
import collections
import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from biopandas.mmcif import PandasMmcif

from proteinfoundation.utils.constants import ATOM_NUMBERING
from proteinfoundation.utils.reprocess_seqres_chains import load_entry, open_pack, raw_path

BACKBONE = {"N", "CA", "C", "O"}
G = {}


def init(pack, data_dir):
    G["ent"], G["mm"], _ = open_pack(pack)
    G["raw"] = pathlib.Path(data_dir) / "raw"
    G["man"] = json.load(open(pathlib.Path(data_dir) / "shard_manifest.json"))


def run(stem):
    pdb, chain = stem.split("_", 1)
    old = load_entry(G["ent"], G["mm"], stem)
    index = {}
    for i, rid in enumerate(old.residue_id):
        _, resn, num, ins = rid.split(":")
        assert ins == "" and (num, resn) not in index, (stem, rid)
        index[(num, resn)] = i
    a = PandasMmcif().read_mmcif(str(raw_path(G["raw"], pdb, G["man"]))).df["ATOM"]
    a = a[(a["label_asym_id"].astype(str) == chain) & (a["pdbx_PDB_model_num"].astype(int) == 1) & a["auth_atom_id"].isin(list(ATOM_NUMBERING))]
    a = a.assign(occ_num=pd.to_numeric(a["occupancy"], errors="coerce"))
    rows, n_skip, n_ctrl, n_ctrl_bad = [], 0, 0, 0
    for (num, resn, name), grp in a.groupby([a["auth_seq_id"].astype(str), "auth_comp_id", "auth_atom_id"], sort=False):
        if (num, resn) not in index:
            n_skip += 1
            continue
        if len(grp) < 2:   # control: a single-row atom must equal its stored value exactly (residue rows aligned)
            n_ctrl += 1
            v = old.coords[index[(num, resn)], ATOM_NUMBERING[name]].numpy().astype(float)
            n_ctrl_bad += int(np.abs(grp[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(float)[0] - v).max() >= 1e-3)
            continue
        xyz = grp[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(float)
        occ = grp["occ_num"].to_numpy(float)
        blank = grp["label_alt_id"].map(lambda v: v is None or v != v or str(v) in (".", "?", "None", "")).any()
        stored = old.coords[index[(num, resn)], ATOM_NUMBERING[name]].numpy().astype(float)
        dist = np.linalg.norm(xyz - stored, axis=1)
        pure = np.flatnonzero(np.abs(xyz - stored).max(1) < 1e-3)
        af2 = None if blank or np.isnan(occ).any() else int(np.flatnonzero(occ == occ.max())[0])   # Bio.PDB: strictly greater occupancy replaces, ties keep the first; blank ids / bad occupancy: not modelled
        rows.append(dict(stem=stem, residue=f"{resn}{num}", atom=name, backbone=name in BACKBONE, n_rows=len(grp),
                         alts=",".join(map(str, grp["label_alt_id"])), occ=",".join(f"{o:g}" for o in occ),
                         kind="pure" if len(pure) else "chimera", pure_alt=str(grp["label_alt_id"].iloc[pure[0]]) if len(pure) else "",
                         pure_is_last=bool(len(pure) and pure[-1] == len(grp) - 1), err_nearest=float(dist.min()),
                         spread=float(np.linalg.norm(xyz[:, None] - xyz[None], axis=-1).max()), af2_alt=("blank" if blank else "unparseable") if af2 is None else str(grp["label_alt_id"].iloc[af2]),
                         stored_is_af2=af2 is not None and bool(np.abs(xyz[af2] - stored).max() < 1e-3), err_vs_af2=np.nan if af2 is None else float(dist[af2])))
    return stem, rows, n_skip, n_ctrl, n_ctrl_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--regression", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args()
    stems = sorted({s for v in json.load(open(a.regression))["strata"].values() for s in v})
    assert len(stems) == 2000, len(stems)
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.pack, a.data_dir)) as ex:
        res = list(ex.map(run, stems, chunksize=8))
    assert len(res) == len(stems)
    df = pd.DataFrame([r for _, rs, *_ in res for r in rs])
    skip, ctrl, ctrl_bad = (sum(r[k] for r in res) for k in (2, 3, 4))
    print(f"groups skipped (residue not in the stored entry) {skip}; control single-row atoms {ctrl}, != stored {ctrl_bad}")
    mixed = df[df.kind == "pure"].groupby(["stem", "residue"]).pure_alt.nunique()
    print(f"residues whose pure altloc atoms come from different altlocs: {int((mixed > 1).sum())} of {len(mixed)}")
    df.to_csv(a.out, sep="\t", index=False)
    per = df.groupby("stem").agg(n=("kind", "size"), chim=("kind", lambda k: int((k == "chimera").sum())))
    print(f"chains {len(stems)}: with altloc atoms {len(per)}, with >= 1 chimera atom {int((per.chim > 0).sum())}, with >= 1 atom != AF2 choice "
          f"{df[~df.stored_is_af2 & df.err_vs_af2.notna()].stem.nunique()}")
    print(f"altloc atoms {len(df)} (backbone {int(df.backbone.sum())}): pure {int((df.kind == 'pure').sum())} (of them last row {int(df.pure_is_last.sum())}), "
          f"chimera {int((df.kind == 'chimera').sum())} (backbone {int(((df.kind == 'chimera') & df.backbone).sum())}); stored == AF2 choice {int(df.stored_is_af2.sum())}; AF2 not modelled {dict(collections.Counter(df.af2_alt[df.af2_alt.isin(['blank', 'unparseable'])]))}")
    for col, sel in (("spread", df.kind.notna()), ("err_nearest", df.kind == "chimera"), ("err_vs_af2", ~df.stored_is_af2 & df.err_vs_af2.notna())):
        v = df.loc[sel, col]
        print(f"{col} over {len(v)} atoms: " + ", ".join(f"q{int(q * 100)} {v.quantile(q):.2f}" for q in (0.5, 0.9, 0.99, 1.0)) + " A" if len(v) else f"{col}: none")
    print("altloc row counts", dict(collections.Counter(df.n_rows)))


if __name__ == "__main__":
    main()
