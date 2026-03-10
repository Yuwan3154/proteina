#!/usr/bin/env python3
"""
Build TED CATH subset cache from an AFDB index file.

Streams the TED domain summary TSV and keeps only entries whose sample IDs
appear in the index. Writes a single cache file (msgpack if available, else pickle)
and prints summary statistics on the fraction of index entries with TED CATH assignment.

Usage:
  python script_utils/build_ted_cath_cache_from_index.py \\
    --index $DATA_PATH/d_FS/d_FS_index.txt \\
    --tsv $DATA_PATH/d_FS/ted_365m.domain_summary.cath.globularity.taxid.tsv \\
    --output $DATA_PATH/d_FS/afdb_to_cath_ted.pkl

Or with defaults (DATA_PATH/d_FS/):
  python script_utils/build_ted_cath_cache_from_index.py
"""

import argparse
import gzip
import os
import pickle
import re
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import msgpack
except ImportError:
    msgpack = None

_MIN_COLUMNS_FOR_CATH = 14
_CATH_CODE_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def load_index_ids(path: str) -> tuple[set, dict]:
    """Load sample IDs from index file (one per line).
    Returns (index_ids, tsv_to_index) mapping TSV-style IDs to index IDs for lookup.
    TED TSV uses model_v4; index may use model_v6. We map v4->v6 when building cache.
    """
    index_ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                index_ids.add(line)
    # TED TSV has model_v4; if index has v6, we need to match TSV rows (v4) to index (v6)
    tsv_to_index = {}
    for idx_id in index_ids:
        tsv_to_index[idx_id] = idx_id
        if "model_v6" in idx_id:
            v4_id = idx_id.replace("model_v6", "model_v4")
            tsv_to_index[v4_id] = idx_id
    return index_ids, tsv_to_index


def open_tsv(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def build_cache(tsv_path: str, tsv_to_index: dict) -> dict:
    """Stream TSV and collect CATH codes. Keys in output use index IDs (e.g. model_v6)."""
    sample_to_cath = {}
    matched = 0
    with open_tsv(tsv_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < _MIN_COLUMNS_FOR_CATH:
                continue
            full_sample_id = parts[0]
            cath_codes_str = parts[13]
            if cath_codes_str == "-":
                continue
            # TED format: "AF-XXX-F1-model_v4_TED01" -> sample_id "AF-XXX-F1-model_v4"
            sample_id = "_".join(full_sample_id.split("_")[:-1])
            if sample_id not in tsv_to_index:
                continue
            index_id = tsv_to_index[sample_id]
            cath_codes = cath_codes_str.split(",")
            if index_id not in sample_to_cath:
                sample_to_cath[index_id] = []
                matched += 1
                if matched % 50000 == 0:
                    print(f"  Collected {matched} matching entries…")
            sample_to_cath[index_id].extend(cath_codes)
    return sample_to_cath


def save_cache(data: dict, output_base: str):
    """Save to {output_base}.subset.msgpack or .subset (matches TEDLabelTransform pkl_path)."""
    subset_msgpack = output_base + ".subset.msgpack"
    subset_pickle = output_base + ".subset"
    if msgpack is not None:
        with open(subset_msgpack, "wb") as f:
            f.write(msgpack.packb(data, use_bin_type=True))
        if os.path.exists(subset_pickle):
            os.remove(subset_pickle)
        print(f"Saved {subset_msgpack} (msgpack)")
        return subset_msgpack
    else:
        with open(subset_pickle, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {subset_pickle} (pickle)")
        return subset_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--index",
        default=os.environ.get("DATA_PATH", ".") + "/d_FS/d_FS_index.txt",
        help="Path to AFDB index file (one ID per line)",
    )
    parser.add_argument(
        "--tsv",
        default=os.environ.get("DATA_PATH", ".") + "/d_FS/ted_365m.domain_summary.cath.globularity.taxid.tsv",
        help="Path to TED domain summary TSV (or .gz)",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("DATA_PATH", ".") + "/d_FS/afdb_to_cath_ted.pkl",
        help="Base path for output cache (TEDLabelTransform expects .subset or .subset.msgpack)",
    )
    args = parser.parse_args()

    # Normalize index path (handle d_FS_index_v6.txt if provided)
    index_path = Path(args.index)
    if not index_path.exists():
        alt = index_path.parent / "d_FS_index_v6.txt"
        if alt.exists():
            index_path = alt
            print(f"Using index: {index_path}")

    print(f"Loading index from {index_path}…")
    index_ids, tsv_to_index = load_index_ids(str(index_path))
    print(f"  {len(index_ids)} IDs in index")

    print(f"Streaming TED TSV from {args.tsv}…")
    sample_to_cath = build_cache(args.tsv, tsv_to_index)

    n_with_cath = len(sample_to_cath)
    n_total = len(index_ids)
    fraction = n_with_cath / n_total if n_total else 0.0

    print(f"\n--- Summary ---")
    print(f"  Total in index:        {n_total}")
    print(f"  With TED CATH:         {n_with_cath}")
    print(f"  Fraction with CATH:    {fraction:.2%}")

    out_path = save_cache(sample_to_cath, args.output)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Cache size:            {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
