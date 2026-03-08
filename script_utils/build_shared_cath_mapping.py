#!/usr/bin/env python
"""Build or extend a shared CATH/TED label mapping for mixed PDB + AFDB training.

Usage:
    # Bootstrap from existing PDB mapping (copies to shared location):
    python script_utils/build_shared_cath_mapping.py \
        --pdb_mapping data/pdb_raw/cath_label_mapping.pt \
        --output data/cath_shared/cath_label_mapping.pt

    # Extend with TED CATH codes from the domain summary TSV:
    python script_utils/build_shared_cath_mapping.py \
        --pdb_mapping data/pdb_raw/cath_label_mapping.pt \
        --ted_tsv data/d_FS/ted_365m.domain_summary.cath.globularity.taxid.tsv \
        --output data/cath_shared/cath_label_mapping.pt

Existing indices are preserved so that pretrained checkpoints remain compatible.
New C/A/T labels found in the TED source are appended at the end.
"""

import argparse
import gzip
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch


CATH_CODE_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def extract_level(cath_code: str, level: str) -> str:
    mapping = {"H": 0, "T": 1, "A": 2, "C": 3}
    return cath_code.rsplit(".", mapping[level])[0]


def load_existing_mapping(path: str) -> dict:
    if not os.path.exists(path):
        return {"C": OrderedDict(), "A": OrderedDict(), "T": OrderedDict()}
    data = torch.load(path, weights_only=False)
    return {k: OrderedDict(data[k]) for k in ("C", "A", "T")}


def collect_ted_cath_codes(tsv_path: str):
    """Yield CATH code strings from a TED domain summary TSV."""
    opener = gzip.open if str(tsv_path).endswith(".gz") else open
    with opener(tsv_path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 14:
                continue
            field = parts[13]
            if field == "-":
                continue
            for code in field.split(","):
                code = code.strip()
                if CATH_CODE_RE.match(code):
                    yield code


def extend_mapping(mapping: dict, cath_codes):
    """Add new C/A/T entries from cath_codes, preserving existing indices."""
    added = {"C": 0, "A": 0, "T": 0}
    for code in cath_codes:
        for level in ("C", "A", "T"):
            key = extract_level(code, level)
            if key not in mapping[level]:
                mapping[level][key] = len(mapping[level])
                added[level] += 1
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb_mapping", type=str, required=True, help="Path to existing PDB cath_label_mapping.pt")
    parser.add_argument("--ted_tsv", type=str, default=None, help="Path to TED domain summary TSV (optional)")
    parser.add_argument("--output", type=str, required=True, help="Output path for shared cath_label_mapping.pt")
    args = parser.parse_args()

    mapping = load_existing_mapping(args.pdb_mapping)
    print(f"Loaded PDB mapping: C={len(mapping['C'])}, A={len(mapping['A'])}, T={len(mapping['T'])}")

    if args.ted_tsv:
        print(f"Scanning TED codes from {args.ted_tsv} ...")
        codes = collect_ted_cath_codes(args.ted_tsv)
        added = extend_mapping(mapping, codes)
        print(f"New entries added: C={added['C']}, A={added['A']}, T={added['T']}")

    print(f"Final mapping: C={len(mapping['C'])}, A={len(mapping['A'])}, T={len(mapping['T'])}")

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_data = {k: dict(v) for k, v in mapping.items()}
    torch.save(save_data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
