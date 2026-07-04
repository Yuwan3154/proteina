"""Audit the TED->CATH join for AFDB chains in the combined PDB+AFDB dataset.

Checks whether the large no-CAT rate among AFDB chains reflects genuine TED v4
coverage gaps (many AFDB structures have no confident TED domain call, and some
v6 accessions postdate TED's v4 UniProt snapshot) versus a join bug (a chain has
a real TED CATH row that the v4->v6 id-matching in build_ted_cath_cache_from_index.py
fails to find).

Usage:
  python scripts/analysis/audit_ted_cath_join_afdb.py --n-sample 500
"""
import argparse
import gzip
import os
import pickle
import random
import re
import time

import pandas as pd

_MIN_COLUMNS_FOR_CATH = 14
_CATH_CODE_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def open_tsv(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="/orcd/pool/006/chenxiou/proteina/data")
    ap.add_argument("--tsv-path", default=None, help="Override the TED TSV path (e.g. a faster-disk copy)")
    ap.add_argument("--n-sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    combined_csv = f"{args.data_path}/pdb_dFS_combined/pdb_dFS_combined.csv"
    chain_to_cat_pkl = f"{args.data_path}/pdb_dFS_combined/combined_chain_to_cat.pkl"
    tsv_path = args.tsv_path or f"{args.data_path}/d_FS/ted_365m.domain_summary.cath.globularity.taxid.tsv"

    print(f"Loading {combined_csv} ...", flush=True)
    df = pd.read_csv(combined_csv, usecols=["id"])
    afdb_ids = set(df["id"][df["id"].str.startswith("AF-")])
    print(f"  {len(df)} total combined chains, {len(afdb_ids)} AFDB chains", flush=True)

    print(f"Loading {chain_to_cat_pkl} ...", flush=True)
    with open(chain_to_cat_pkl, "rb") as f:
        chain_to_cat = pickle.load(f)
    afdb_labeled = {c for c in afdb_ids if c in chain_to_cat}
    afdb_unlabeled = afdb_ids - afdb_labeled
    print(f"\n--- AFDB chain-level TED/CATH label summary (combined dataset population) ---", flush=True)
    print(f"  AFDB chains total:     {len(afdb_ids)}", flush=True)
    print(f"  AFDB chains labeled:   {len(afdb_labeled)} ({len(afdb_labeled)/len(afdb_ids):.2%})", flush=True)
    print(f"  AFDB chains unlabeled: {len(afdb_unlabeled)} ({len(afdb_unlabeled)/len(afdb_ids):.2%})", flush=True)

    random.seed(args.seed)
    sample = random.sample(sorted(afdb_unlabeled), min(args.n_sample, len(afdb_unlabeled)))
    # v6 id -> v4 id (same substitution as build_ted_cath_cache_from_index.py)
    v4_targets = {}
    for v6_id in sample:
        v4_id = v6_id.replace("model_v6", "model_v4") if "model_v6" in v6_id else v6_id
        v4_targets[v4_id] = v6_id

    # ALL v4-derived AFDB ids (not just the unlabeled sample) -- lets the same single pass over
    # the 128GB TSV also compute the GLOBAL fraction of the actual dataset population with a
    # real TED CATH match, not just spot-check the 500-chain sample.
    all_afdb_v4_to_v6 = {}
    for v6_id in afdb_ids:
        v4_id = v6_id.replace("model_v6", "model_v4") if "model_v6" in v6_id else v6_id
        all_afdb_v4_to_v6[v4_id] = v6_id

    print(f"\nScanning {tsv_path} ({os.path.getsize(tsv_path)/1e9:.1f} GB) for "
          f"{len(sample)} sampled ids + full {len(all_afdb_v4_to_v6)}-id population ...", flush=True)

    found_with_cath = []
    found_no_cath = []
    global_matched_with_cath = 0
    n_lines = 0
    t_start = time.time()
    t_last_report = t_start
    with open_tsv(tsv_path) as fh:
        for line in fh:
            n_lines += 1
            now = time.time()
            if now - t_last_report >= 300:  # time-based, not line-count-based -- guarantees visibility
                pos = fh.tell()
                elapsed = now - t_start
                rate_mb_s = pos / 1e6 / elapsed if elapsed > 0 else 0
                print(f"  ...{elapsed/60:.1f} min elapsed, {pos/1e9:.2f} GB / "
                      f"{os.path.getsize(tsv_path)/1e9:.1f} GB read ({rate_mb_s:.1f} MB/s), "
                      f"{n_lines/1e6:.1f}M lines, {len(found_with_cath)} true-misses, "
                      f"{global_matched_with_cath} global matches so far", flush=True)
                t_last_report = now
            parts = line.rstrip("\n").split("\t")
            if len(parts) < _MIN_COLUMNS_FOR_CATH:
                continue
            full_sample_id = parts[0]
            sample_id = "_".join(full_sample_id.split("_")[:-1])
            cath_field = parts[13]
            has_cath = cath_field != "-" and _CATH_CODE_RE.match(cath_field.split(",")[0])
            if has_cath and sample_id in all_afdb_v4_to_v6:
                global_matched_with_cath += 1
            if sample_id not in v4_targets:
                continue
            v6_id = v4_targets[sample_id]
            if has_cath:
                found_with_cath.append((v6_id, sample_id, cath_field))
            else:
                found_no_cath.append((v6_id, sample_id, cath_field))

    n_afdb_total = len(all_afdb_v4_to_v6)
    print(f"\n--- GLOBAL: fraction of the {n_afdb_total} AFDB chains actually in the combined "
          f"dataset with a real TED CATH match ---", flush=True)
    print(f"  Matched: {global_matched_with_cath} ({global_matched_with_cath/n_afdb_total:.2%})", flush=True)
    print(f"  Not matched (absent from TSV or '-' CATH field): "
          f"{n_afdb_total - global_matched_with_cath} "
          f"({(n_afdb_total - global_matched_with_cath)/n_afdb_total:.2%})", flush=True)

    n_absent = len(sample) - len(found_with_cath) - len(found_no_cath)
    print(f"\n--- Diagnostic: of {len(sample)} sampled 'no-CAT' AFDB chains ---", flush=True)
    print(f"  Absent from TED v4 TSV entirely (genuine v4/v6 gap, expected): {n_absent} "
          f"({n_absent/len(sample):.2%})", flush=True)
    print(f"  Present in TSV but CATH field is '-' (TED itself has no confident call): "
          f"{len(found_no_cath)} ({len(found_no_cath)/len(sample):.2%})", flush=True)
    print(f"  *** Present in TSV WITH a real CATH code -- JOIN BUG, should have matched ***: "
          f"{len(found_with_cath)} ({len(found_with_cath)/len(sample):.2%})", flush=True)
    if found_with_cath:
        print("\n  Examples of missed matches:", flush=True)
        for v6_id, sample_id, cath in found_with_cath[:10]:
            print(f"    {v6_id}  (v4 form: {sample_id})  CATH={cath}", flush=True)


if __name__ == "__main__":
    main()
