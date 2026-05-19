#!/usr/bin/env python3
"""
Get stoichiometry information for PDB entries from RCSB PDB.

Consolidates the original get_stoichiometry.py + get_stoichiometry_enhanced.py:
  - Robust REST → GraphQL fallback for assembly symmetry
  - Obsolete-PDB detection with automatic replacement
  - Batch CLI with resume-from-existing-output
  - --test flag for the 3-case obsolete-PDB harness

Usage:
  # Full batch run (defaults match ~/data/bad_afdb/ layout)
  python get_stoichiometry.py

  # Smaller run
  python get_stoichiometry.py \\
    --input_csv /path/to/small.csv \\
    --cache_dir /tmp/pdb_info \\
    --output_csv /tmp/stoich.csv

  # Run the 3-case obsolete-PDB harness
  python get_stoichiometry.py --test
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


DEFAULT_BASE_DIR = Path.home() / "data" / "bad_afdb"
DEFAULT_INPUT_CSV = DEFAULT_BASE_DIR / "pdb_70_cluster_reps_aligned_confidence_aggregate.csv"
DEFAULT_CACHE_DIR = DEFAULT_BASE_DIR / "pdb_info"
DEFAULT_OUTPUT_CSV = DEFAULT_BASE_DIR / "stoichiometry_results_full.csv"


def check_obsolete_pdb(pdb_id: str, cache_dir: Path):
    """Return the replacement PDB ID if pdb_id is obsolete, else None.

    Scrapes the RCSB obsolete-structure page; result is cached per PDB.
    """
    cache_file = cache_dir / f"{pdb_id}_obsolete.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f).get("replacement")

    url = f"https://www.rcsb.org/structure/removed/{pdb_id}"
    response = requests.get(url, timeout=10)

    replacement = None
    if response.status_code == 200:
        match = re.search(r'href="/structure/([0-9][A-Z0-9]{3})"', response.text)
        if match:
            replacement = match.group(1).upper()

    with open(cache_file, "w") as f:
        json.dump({"pdb_id": pdb_id, "replacement": replacement}, f)

    time.sleep(0.1)
    return replacement


def get_pdb_info(pdb_id: str, cache_dir: Path):
    """Fetch RCSB entry record (REST), with retries and on-disk caching."""
    cache_file = cache_dir / f"{pdb_id}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                with open(cache_file, "w") as f:
                    json.dump(data, f, indent=2)
                time.sleep(0.1)
                return data
            if response.status_code == 404:
                return None
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  Error fetching {pdb_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def get_assembly_info(pdb_id: str, cache_dir: Path):
    """Fetch primary biological assembly (REST), with retries and caching."""
    cache_file = cache_dir / f"{pdb_id}_assembly.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/1"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                with open(cache_file, "w") as f:
                    json.dump(data, f, indent=2)
                time.sleep(0.1)
                return data
            if response.status_code == 404:
                return None
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  Error fetching assembly for {pdb_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def get_assembly_graphql(pdb_id: str, cache_dir: Path):
    """GraphQL fallback for cases where the REST assembly record lacks symmetry."""
    cache_file = cache_dir / f"{pdb_id}_graphql.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
            if data.get("success"):
                stoich_list = data.get("stoichiometry", [])
                return ("".join(stoich_list), data.get("oligomeric_state"), stoich_list)
            return (None, None, None)

    url = "https://data.rcsb.org/graphql"
    query = """
    query {
      entry(entry_id: "%s") {
        assemblies {
          rcsb_id
          rcsb_struct_symmetry {
            kind
            stoichiometry
            oligomeric_state
          }
        }
      }
    }
    """ % pdb_id

    try:
        response = requests.post(url, json={"query": query}, timeout=10)
        if response.status_code == 200:
            result = response.json()
            entry = (result.get("data") or {}).get("entry") or {}
            assemblies = entry.get("assemblies") or []
            if assemblies:
                symmetry = assemblies[0].get("rcsb_struct_symmetry") or []
                if symmetry:
                    sym = symmetry[0]
                    stoich_list = sym.get("stoichiometry", [])
                    state = sym.get("oligomeric_state")
                    with open(cache_file, "w") as f:
                        json.dump(
                            {
                                "success": True,
                                "stoichiometry": stoich_list,
                                "oligomeric_state": state,
                            },
                            f,
                        )
                    time.sleep(0.1)
                    return ("".join(stoich_list), state, stoich_list)
    except Exception as e:
        print(f"  GraphQL error for {pdb_id}: {e}")

    with open(cache_file, "w") as f:
        json.dump({"success": False}, f)
    return (None, None, None)


def determine_stoichiometry(pdb_id: str, cache_dir: Path):
    """REST first, GraphQL fallback. Returns (stoich_str, oligomeric_state, stoich_list)."""
    assembly_info = get_assembly_info(pdb_id, cache_dir)

    if assembly_info and "rcsb_struct_symmetry" in assembly_info:
        symmetry = assembly_info["rcsb_struct_symmetry"]
        if isinstance(symmetry, list) and len(symmetry) > 0:
            stoich_list = symmetry[0].get("stoichiometry", [])
            state = symmetry[0].get("oligomeric_state", "Unknown")
            if stoich_list:
                return ("".join(stoich_list), state, stoich_list)

    # REST oligomeric_details fallback (catches obvious "monomer" cases)
    if assembly_info and "pdbx_struct_assembly" in assembly_info:
        details = (assembly_info["pdbx_struct_assembly"].get("oligomeric_details") or "").lower()
        if "monomer" in details:
            return ("A1", "Monomer", ["A1"])

    # GraphQL fallback
    return get_assembly_graphql(pdb_id, cache_dir)


def process_pdb(pdb_id: str, cache_dir: Path):
    """Resolve obsolete PDB → replacement, then fetch stoichiometry.

    Returns (final_pdb, stoich_str, oligomeric_state, stoich_list, is_obsolete, replacement).
    """
    replacement = check_obsolete_pdb(pdb_id, cache_dir)

    if replacement:
        print(f"  ⚠️  Obsolete - Using replacement: {replacement}")
        pdb_to_use = replacement
        is_obsolete = True
    else:
        pdb_to_use = pdb_id
        is_obsolete = False

    if get_pdb_info(pdb_to_use, cache_dir):
        print("  ✓ Retrieved PDB info")

    stoich_str, oligomeric_state, stoich_list = determine_stoichiometry(pdb_to_use, cache_dir)

    if stoich_str:
        print(f"  ✓ Stoichiometry: {stoich_str}")
        print(f"    Oligomeric state: {oligomeric_state}")
    else:
        print("  ✗ Could not determine stoichiometry")

    return (pdb_to_use, stoich_str, oligomeric_state, stoich_list, is_obsolete, replacement)


def run_test_harness(cache_dir: Path) -> None:
    """3-case obsolete-PDB harness (was get_stoichiometry_enhanced.py main())."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        ("8JOH", "A", "Obsolete - should find 9UOZ with A1"),
        ("8T0Q", "A", "Obsolete - should find 9Q1A"),
        ("9A1S", "A", "Current - should get A1"),
    ]

    print("=" * 80)
    print("TESTING STOICHIOMETRY EXTRACTION (obsolete-PDB harness)")
    print("=" * 80)

    for pdb_id, _, description in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Testing {pdb_id}: {description}")
        print("=" * 80)
        final_pdb, stoich_str, state, stoich_list, is_obsolete, replacement = process_pdb(
            pdb_id, cache_dir
        )
        print("\nResult:")
        print(f"  Original PDB: {pdb_id}")
        print(f"  Final PDB: {final_pdb}")
        print(f"  Is obsolete: {is_obsolete}")
        if replacement:
            print(f"  Replacement: {replacement}")
        print(f"  Stoichiometry: {stoich_str}")
        print(f"  Oligomeric state: {state}")
        print(f"  Full composition: {stoich_list}")


def run_batch(
    input_csv: Path,
    cache_dir: Path,
    output_csv: Path,
    id_column: str,
    save_interval: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if id_column not in df.columns:
        raise SystemExit(
            f"ERROR: --input_csv {input_csv} has no column '{id_column}'. "
            f"Found columns: {df.columns.tolist()}"
        )

    existing_results = {}
    if output_csv.exists():
        print(f"Found existing output file: {output_csv}")
        existing_df = pd.read_csv(output_csv)
        existing_results = {row[id_column]: row.to_dict() for _, row in existing_df.iterrows()}
        print(f"Loaded {len(existing_results)} existing results")

    print(f"Processing {len(df)} entries...")
    print("=" * 60)

    results = []
    processed_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        pdb_chain = row[id_column]

        if pdb_chain in existing_results:
            print(f"\n[{idx + 1}/{len(df)}] Skipping {pdb_chain} (already processed)")
            results.append(existing_results[pdb_chain])
            skipped_count += 1
            continue

        parts = str(pdb_chain).split("_")
        pdb_id = parts[0]

        print(f"\n[{idx + 1}/{len(df)}] Processing {pdb_chain} (PDB: {pdb_id})")

        final_pdb, stoich_str, state, stoich_list, is_obsolete, replacement = process_pdb(
            pdb_id, cache_dir
        )

        results.append(
            {
                id_column: pdb_chain,
                "stoichiometry": stoich_str,
                "oligomeric_state": state,
                "full_composition": str(stoich_list) if stoich_list else None,
                "final_pdb": final_pdb,
                "is_obsolete": is_obsolete,
                "replacement": replacement,
            }
        )
        processed_count += 1

        if processed_count % save_interval == 0:
            print(f"\n💾 Saving progress ({len(results)} total entries)...")
            pd.DataFrame(results).to_csv(output_csv, index=False)
            print(f"  ✓ Saved to {output_csv}")

    print("\n" + "=" * 60)
    print("FINAL SAVE")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    simple_output = output_csv.with_name(output_csv.stem + "_simple" + output_csv.suffix)
    results_df[[id_column, "stoichiometry"]].to_csv(simple_output, index=False)

    print(f"✅ Full results saved to: {output_csv}")
    print(f"✅ Simple results saved to: {simple_output}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total entries: {len(df)}")
    print(f"Newly processed: {processed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Total in output: {len(results_df)}")
    print("=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch stoichiometry information for PDB entries from RCSB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--id_column", default="pdb",
                   help="Column in --input_csv that contains PDB_chain IDs.")
    p.add_argument("--save_interval", type=int, default=100,
                   help="Save partial results every N processed entries.")
    p.add_argument("--test", action="store_true",
                   help="Run the 3-case obsolete-PDB harness instead of batch processing.")
    return p


def main():
    args = build_arg_parser().parse_args()
    cache_dir = Path(args.cache_dir).expanduser()

    if args.test:
        run_test_harness(cache_dir)
        return

    run_batch(
        input_csv=Path(args.input_csv).expanduser(),
        cache_dir=cache_dir,
        output_csv=Path(args.output_csv).expanduser(),
        id_column=args.id_column,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
