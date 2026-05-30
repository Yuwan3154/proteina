"""Precompute afdb_chain_to_cat: maps each AFDB stem (case-sensitive, e.g.
"AF-A0A000-F1-model_v6") to the sorted unique list of CAT (3-level CATH:
Class.Architecture.Topology) codes covering all of its TED domains. Multi-domain
chains keep all their CATs. Output is a pickle consumed by CATBalancedSampler.

AFDB analogue of precompute_chain_to_cat.py (which sources PDB labels from
CATHLabelTransform). Here labels come from the TED CATH subset cache built by
build_ted_cath_cache_from_index.py (stem -> [raw 4-level CATH codes]).

Filter ordering (per design): restrict to processed (length-filtered) stems FIRST,
then intersect with the AFDB v6 index LAST. The v6 membership runs on the reduced
labeled+processed set; the expensive 128GB TSV scan already happened when the TED
cache was built.

Keys are NOT lowercased: AFDB accessions are case-sensitive and must match the
processed .pt stems exactly (unlike PDB, which lowercases).

Run on Engaging (DATA_PATH set), CPU-only:
    conda activate cue_openfold
    python script_utils/precompute_afdb_chain_to_cat.py \
        --processed-root $DATA_PATH/d_FS/processed
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import msgpack

from proteinfoundation.utils.ff_utils.pdb_utils import extract_cath_code_by_level

_CATH_CODE_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def load_ted_cache(cache_base):
    """Load the TED CATH subset cache (stem -> [raw 4-level CATH codes]).

    cache_base is TEDLabelTransform's pkl_path; the on-disk cache is
    <cache_base>.subset.msgpack (preferred) or <cache_base>.subset (pickle).
    """
    msgpack_path = f"{cache_base}.subset.msgpack"
    pickle_path = f"{cache_base}.subset"
    if os.path.exists(msgpack_path):
        with open(msgpack_path, "rb") as f:
            return msgpack.unpackb(f.read(), strict_map_key=False)
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"No TED cache at {msgpack_path} or {pickle_path}. Build it first with "
        f"build_ted_cath_cache_from_index.py --index $DATA_PATH/d_FS/d_FS_index_v6.txt"
    )


def processed_stems(processed_root):
    """Set of .pt stems under processed_root (AFDB is sharded: <bucket>/<stem>.pt)."""
    root = Path(processed_root)
    if not root.exists():
        raise FileNotFoundError(f"processed-root does not exist: {processed_root}")
    return {p.stem for p in root.rglob("*.pt")}


def load_index_ids(index_path):
    ids = set()
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def build_afdb_chain_to_cat(sample_to_cath, keep_stems):
    """sample_to_cath: stem -> [raw CATH codes]. Returns stem -> sorted unique CAT
    list, restricted to keep_stems. Skips entries that are not valid CATH (X.X.X.X)."""
    chain_to_cat = {}
    for stem, cath_codes in sample_to_cath.items():
        if stem not in keep_stems:
            continue
        cats = set()
        for code in cath_codes:
            if not _CATH_CODE_RE.match(code):
                continue
            cats.add(extract_cath_code_by_level(code, "T"))
        if cats:
            chain_to_cat[stem] = sorted(cats)
    return chain_to_cat


def main():
    parser = argparse.ArgumentParser(
        description="Precompute afdb_chain_to_cat (AFDB stem -> sorted list[CAT code])."
    )
    parser.add_argument("--cache-base", default=None,
                        help="TED cache pkl_path base (default: $DATA_PATH/d_FS/afdb_to_cath_ted.pkl)")
    parser.add_argument("--processed-root", default=None,
                        help="Processed .pt root to restrict to (default: $DATA_PATH/d_FS/processed)")
    parser.add_argument("--v6-index", default=None,
                        help="AFDB v6 index for the final membership filter (default: $DATA_PATH/d_FS/d_FS_index_v6.txt)")
    parser.add_argument("--out", default=None,
                        help="Output pickle (default: $DATA_PATH/d_FS/afdb_chain_to_cat.pkl)")
    parser.add_argument("--no-v6-filter", action="store_true",
                        help="Skip the final v6 intersection (when the cache is already v6-keyed).")
    args = parser.parse_args()

    data_path = os.environ.get("DATA_PATH")
    if data_path is None and None in (args.cache_base, args.processed_root, args.v6_index, args.out):
        parser.error("DATA_PATH not set; pass --cache-base, --processed-root, --v6-index, --out explicitly.")
    cache_base = args.cache_base or os.path.join(data_path, "d_FS", "afdb_to_cath_ted.pkl")
    processed_root = args.processed_root or os.path.join(data_path, "d_FS", "processed")
    v6_index = args.v6_index or os.path.join(data_path, "d_FS", "d_FS_index_v6.txt")
    out_path = args.out or os.path.join(data_path, "d_FS", "afdb_chain_to_cat.pkl")

    sample_to_cath = load_ted_cache(cache_base)
    print(f"Loaded TED cache: {len(sample_to_cath)} labeled stems.")

    # Cheap-first: restrict to processed (length-filtered) stems.
    proc = processed_stems(processed_root)
    print(f"Processed stems under {processed_root}: {len(proc)}.")
    keep = proc

    # Expensive-last: AFDB v6 membership.
    if not args.no_v6_filter:
        v6 = load_index_ids(v6_index)
        keep = keep & v6
        print(f"v6 index ids: {len(v6)}; processed-and-v6 stems: {len(keep)}.")

    chain_to_cat = build_afdb_chain_to_cat(sample_to_cath, keep)

    all_cats = set()
    multi = 0
    for cats in chain_to_cat.values():
        all_cats.update(cats)
        if len(cats) > 1:
            multi += 1
    labeled_frac = len(chain_to_cat) / len(keep) if keep else 0.0
    print("\n=== afdb_chain_to_cat ===")
    print(f"  labeled stems (in pickle):   {len(chain_to_cat)}")
    print(f"  candidate stems:             {len(keep)}")
    print(f"  labeled fraction:            {labeled_frac:.2%}")
    print(f"  distinct CAT codes:          {len(all_cats)}")
    print(f"  multi-CAT (multi-domain):    {multi}")

    assert len(chain_to_cat) > 0, (
        "afdb_chain_to_cat is empty -- likely a stem-key mismatch between the TED "
        "cache (v6 index ids) and processed .pt stems. Verify processed stem format/casing."
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(chain_to_cat, f)
    print(f"Wrote pickle -> {out_path}")


if __name__ == "__main__":
    main()
