"""Cross-check flagged worst-offender chains against REAL RCSB PDB deposit metadata.

Companion to audit_chain_quality_vs_resolution.py. Takes that script's
`worst_offenders_for_rcsb_check.csv` (pdb, chain, resolution, n_chains_in_entry,
frac_resolved, sibling_max_frac_resolved) and, for each row, queries RCSB's public
Data API (data.rcsb.org, no auth needed) to independently verify:
  1. entry-level resolution (does OUR value match RCSB's, i.e. did we read/join it
     correctly) -- GET /rest/v1/core/entry/{pdb_id}
  2. per-chain-INSTANCE unobserved-residue coverage, RCSB's own authoritative
     computation of how much of that specific chain is missing density -- GET
     /rest/v1/core/polymer_entity_instance/{pdb_id}/{asym_id}
     (rcsb_polymer_instance_feature_summary, type UNOBSERVED_RESIDUE_XYZ)

CAVEAT (see memory rule on CIF label_asym_id vs auth_asym_id, and the schema
investigation this script's sibling analysis is based on): the `chain` column in
our metadata CSV comes from pdb_seqres.txt.gz, which historically uses AUTHOR
chain IDs (auth_asym_id) -- but the RCSB polymer_entity_instance endpoint expects
the ASYM ID (label_asym_id) in its URL path. These coincide for simple,
single-entity depositions but can diverge for large multi-chain assemblies --
confirmed empirically during testing (a 22-chain entry where the direct lookup
returned a real instance, but its implied total length was wildly inconsistent
with ours). This script does NOT attempt to auto-resolve that (would need
querying every instance in the entry and picking a length-based best match, out
of scope here) -- it flags the risk instead: `length_plausibility` backs out the
RCSB instance's implied total residue count from (n_unobserved / coverage) and
compares it against our own `length_deposited`; a large mismatch means the
direct lookup very likely hit the WRONG chain and the "confirmation" (or
contradiction) it reports should NOT be trusted without manual RCSB lookup at
https://www.rcsb.org/structure/{pdb_id}.

This is designed to run purely LOCALLY (stdlib urllib only, no extra deps, no
cluster/GPU needed) -- it just needs outbound internet access to data.rcsb.org.

Usage:
    python scripts/analysis/rcsb_crosscheck_worst_offenders.py \
        --input analysis_out/chain_quality_vs_resolution/worst_offenders_for_rcsb_check.csv \
        --out analysis_out/chain_quality_vs_resolution/rcsb_crosscheck.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE = "https://data.rcsb.org/rest/v1/core"
UA = {"User-Agent": "proteina-data-quality-audit/1.0 (research use)"}


def _get_json(url: str, retries: int = 3, backoff: float = 1.5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)
    return None


def fetch_entry_resolution(pdb_id: str):
    data = _get_json(f"{BASE}/entry/{pdb_id.lower()}")
    if data is None:
        return None
    # resolution_combined is a list (multi-method entries can have >1 value)
    res_list = (data.get("rcsb_entry_info") or {}).get("resolution_combined")
    return res_list[0] if res_list else None


def fetch_instance_unobserved(pdb_id: str, asym_id: str):
    data = _get_json(f"{BASE}/polymer_entity_instance/{pdb_id.lower()}/{asym_id}")
    if data is None:
        return None
    summary = data.get("rcsb_polymer_instance_feature_summary") or []
    unobs = next((s for s in summary if s.get("type") == "UNOBSERVED_RESIDUE_XYZ"), None)
    n_total_est = None
    poly_comp = data.get("rcsb_polymer_entity_instance_container_identifiers") or {}
    return {
        "n_unobserved": unobs.get("count") if unobs else 0,
        "coverage_unobserved": unobs.get("coverage") if unobs else 0.0,
        "auth_asym_id": poly_comp.get("auth_asym_id"),
        "asym_id": poly_comp.get("asym_id"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--sleep", type=float, default=0.34,
                     help="Seconds between RCSB requests (be a polite API citizen).")
    args = ap.parse_args()

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[rcsb_crosscheck] {len(rows)} worst-offender chains to check")

    out_rows = []
    for i, row in enumerate(rows):
        pdb_id, chain = row["pdb"], row["chain"]
        print(f"[{i+1}/{len(rows)}] {pdb_id}_{chain} ...", file=sys.stderr)

        rcsb_resolution = fetch_entry_resolution(pdb_id)
        time.sleep(args.sleep)

        inst = fetch_instance_unobserved(pdb_id, chain)
        chain_id_matched_via = "direct" if inst is not None else "NOT_FOUND"
        time.sleep(args.sleep)

        # Sanity-check the direct chain-ID lookup actually hit the SAME chain: our
        # `chain` column comes from pdb_seqres (historically auth_asym_id), but this
        # RCSB endpoint expects asym_id (label_asym_id) -- these coincide for simple
        # depositions but frequently diverge for large multi-chain assemblies (see
        # memory rule on CIF label vs auth chain IDs). Back out the RCSB instance's
        # implied total residue count from (n_unobserved / coverage) and compare
        # against our own length_deposited -- if they're wildly different, "direct"
        # was very likely the WRONG chain, not confirmation of a real discrepancy.
        length_plausibility = None
        if inst is not None and row.get("length_deposited"):
            expected_len = float(row["length_deposited"])
            coverage = inst["coverage_unobserved"]
            if coverage and coverage > 0:
                implied_total = inst["n_unobserved"] / coverage
                ratio = implied_total / expected_len if expected_len > 0 else None
                if ratio is not None and (ratio < 0.5 or ratio > 2.0):
                    length_plausibility = f"MISMATCH_SUSPECTED (implied_total={implied_total:.0f} vs expected={expected_len:.0f})"
                else:
                    length_plausibility = "OK"
            else:
                # coverage==0 (fully observed per RCSB) -- can't back out a total
                # length this way; not inherently suspicious, just uninformative.
                length_plausibility = "UNVERIFIABLE (coverage=0)"

        out_rows.append({
            **row,
            "rcsb_resolution": rcsb_resolution,
            "resolution_matches_ours": (
                abs(rcsb_resolution - float(row["resolution"])) < 0.05
                if rcsb_resolution is not None and row.get("resolution") else None
            ),
            "length_plausibility": length_plausibility,
            "rcsb_n_unobserved_residues": inst["n_unobserved"] if inst else None,
            "rcsb_unobserved_coverage": inst["coverage_unobserved"] if inst else None,
            "rcsb_auth_asym_id": inst["auth_asym_id"] if inst else None,
            "rcsb_asym_id": inst["asym_id"] if inst else None,
            "chain_id_matched_via": chain_id_matched_via,
        })

    out_fields = list(out_rows[0].keys()) if out_rows else []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    n_confirmed_good_res = sum(
        1 for r in out_rows
        if r["rcsb_resolution"] is not None and r["rcsb_resolution"] <= 3.0
    )
    n_confirmed_high_unobserved = sum(
        1 for r in out_rows
        if r["rcsb_n_unobserved_residues"] is not None and r["rcsb_unobserved_coverage"]
        and r["rcsb_unobserved_coverage"] > 0.2
    )
    n_not_found = sum(1 for r in out_rows if r["chain_id_matched_via"] == "NOT_FOUND")
    n_mismatch_suspected = sum(
        1 for r in out_rows if r["length_plausibility"] and "MISMATCH_SUSPECTED" in r["length_plausibility"]
    )
    print(f"\n[rcsb_crosscheck] {len(out_rows)} checked -> {args.out}")
    print(f"  RCSB confirms good entry resolution (<=3.0 A): {n_confirmed_good_res}")
    print(f"  RCSB confirms high unobserved-residue coverage (>20%) for this SPECIFIC chain: "
          f"{n_confirmed_high_unobserved}")
    if n_not_found:
        print(f"  WARNING: {n_not_found} chain IDs not found via direct lookup -- likely a "
              f"label_asym_id vs auth_asym_id mismatch; needs manual RCSB lookup "
              f"(https://www.rcsb.org/structure/{{pdb_id}}).")
    if n_mismatch_suspected:
        print(f"  WARNING: {n_mismatch_suspected} rows have length_plausibility=MISMATCH_SUSPECTED "
              f"-- the direct chain-ID lookup likely hit the WRONG chain (auth_asym_id vs "
              f"label_asym_id divergence, common in large assemblies); do NOT trust their "
              f"rcsb_n_unobserved_residues/coverage without a manual RCSB lookup.")


if __name__ == "__main__":
    main()
