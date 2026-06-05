#!/usr/bin/env python3
"""Build the combined PDB+AFDB CSV + merged chain_to_cat for 2b (Phase 2b step 2).

PDB: df_pdb maxl256 CSV filtered to deposition_date <= cutoff (purge-test), id+pdb
lowercased to match .pt stems / lowercased pdb_chain_to_cat keys. AFDB: d_FS.csv as-is
(case-sensitive). Output combined CSV (pdb,id,sequence) + merged chain_to_cat.pkl.
"""
import argparse
import pickle

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-csv", required=True)
    ap.add_argument("--afdb-csv", required=True)
    ap.add_argument("--pdb-pkl", required=True)
    ap.add_argument("--afdb-pkl", required=True)
    ap.add_argument("--deposition-cutoff", default="2019-08-28")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-pkl", required=True)
    args = ap.parse_args()

    pdb = pd.read_csv(args.pdb_csv)
    n_pdb_all = len(pdb)
    dep = pdb["deposition_date"].astype(str).str[:10]
    pdb = pdb[dep <= args.deposition_cutoff]
    n_pdb_cut = len(pdb)
    pdb_out = pdb[["pdb", "id", "sequence"]].copy()
    pdb_out["pdb"] = pdb_out["pdb"].astype(str).str.lower()
    pdb_out["id"] = pdb_out["id"].astype(str).str.lower()

    afdb = pd.read_csv(args.afdb_csv)
    afdb_out = afdb[["pdb", "id", "sequence"]].copy()

    combined = pd.concat([pdb_out, afdb_out], ignore_index=True)
    n_dup = int(combined["id"].duplicated().sum())
    combined.to_csv(args.out_csv, index=False)

    with open(args.pdb_pkl, "rb") as fh:
        c_pdb = pickle.load(fh)
    with open(args.afdb_pkl, "rb") as fh:
        c_afdb = pickle.load(fh)
    overlap = set(c_pdb) & set(c_afdb)
    assert not overlap, f"chain_to_cat key collision PDB vs AFDB: {list(overlap)[:10]}"
    combined_pkl = {**c_pdb, **c_afdb}
    with open(args.out_pkl, "wb") as fh:
        pickle.dump(combined_pkl, fh)

    print(f"[csv] PDB rows {n_pdb_all} -> deposition<={args.deposition_cutoff}: {n_pdb_cut}")
    print(f"[csv] AFDB rows {len(afdb_out)}; combined rows {len(combined)}; dup ids {n_dup}")
    print(f"[csv] sample PDB id: {pdb_out['id'].iloc[0]}  sample AFDB id: {afdb_out['id'].iloc[0]}")
    print(f"[pkl] pdb_keys {len(c_pdb)} afdb_keys {len(c_afdb)} overlap {len(overlap)} -> combined {len(combined_pkl)}")
    print("[done] combined CSV + chain_to_cat written")


if __name__ == "__main__":
    main()
