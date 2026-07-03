"""Cross-check per-chain structural completeness against deposit-level resolution.

Motivation: PDBDataSelector filters on `resolution`, but resolution is a property
of the WHOLE deposited entry (see PDBManager._parse_resolution -> a dict keyed by
4-char PDB ID only, broadcast via df.pdb.map(...) to every chain of that entry --
proteinfoundation/graphein_utils/graphein_utils.py). A multi-chain deposit with
good overall resolution can still contain an individual chain that is largely
unresolved (flexible loop, partially-disordered subunit) -- the resolution filter
has NO way to catch that per-chain, since it only ever keeps or drops an entry as
a whole.

This script quantifies exactly that gap:
  1. Reuses `_audit_one` from audit_broken_chains.py for per-chain CA-CA break
     stats (internal gaps only -- does not need to change that script's logic).
  2. Adds a SEPARATE, complementary metric this doesn't cover: total missing
     fraction INCLUDING N/C-terminal missing residues, via
     `frac_resolved = n_residues_in_pt / length_in_metadata_csv`
     (`length` in the sidecar CSV comes from pdb_seqres -- the full deposited/
     construct sequence -- while n_residues in the .pt is resolved-only).
  3. Joins both against the sidecar metadata CSV (resolution, n_chains, pdb,
     chain, deposition_date, experiment_type).
  4. Flags "worst offenders": chains with low frac_resolved despite the entry
     passing a good-resolution cutoff, AND (for multi-chain entries) compares
     against sibling chains of the SAME entry to see whether they are much more
     complete -- the direct signature of the user's hypothesis.

Outputs (under --out_dir):
    - chain_quality_report.md         -- headline numbers
    - per_chain_quality.csv           -- one row per chain, all metrics + metadata
    - worst_offenders.csv             -- flagged cases, sorted by severity
    - worst_offenders_for_rcsb_check.csv -- (pdb, chain, resolution, n_chains,
      frac_resolved, length_deposited, n_residues_resolved, sibling info) for the
      top-K, to feed into the separate local RCSB cross-check script (needs
      length_deposited for its own auth/label chain-ID mismatch sanity check).

Usage:
    python scripts/analysis/audit_chain_quality_vs_resolution.py \
        --processed_dir data/pdb_train/processed \
        --metadata_csv data/pdb_train/df_pdb_f1_....csv \
        --cutoff 4.0 --sample_size 10000 --resolution_good_cutoff 3.0 \
        --frac_resolved_bad_cutoff 0.7 --top_k 30 --workers 32 \
        --out_dir analysis_out/chain_quality_vs_resolution
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

_THIS = pathlib.Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from audit_broken_chains import _audit_one  # noqa: E402


def _audit_one_or_none(args_tuple):
    """Picklable top-level wrapper for ProcessPoolExecutor -- swallows per-file
    load failures the same way the single-process loop does, instead of killing
    the whole pool over one corrupt .pt."""
    p, cutoff = args_tuple
    try:
        summary, _breaks = _audit_one(p, cutoff)
    except Exception as e:
        return (p.stem, None, repr(e))
    return (p.stem, summary, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", type=pathlib.Path, required=True)
    ap.add_argument("--metadata_csv", type=pathlib.Path, required=True,
                     help="Sidecar CSV written by PDBLightningDataModule.prepare_data() "
                          "(columns: id, pdb, chain, length, n_chains, resolution, "
                          "deposition_date, experiment_type, ...)")
    ap.add_argument("--cutoff", type=float, default=4.0, help="CA-CA break cutoff (Angstrom)")
    ap.add_argument("--sample_size", type=int, default=20000,
                     help="Random subset of .pt files to scan; 0 = scan all (slow).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution_good_cutoff", type=float, default=3.0,
                     help="Entries at or better than this resolution are 'good overall quality'.")
    ap.add_argument("--frac_resolved_bad_cutoff", type=float, default=0.7,
                     help="Chains with frac_resolved below this are flagged as 'largely unresolved'.")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1,
                     help="Parallel worker processes for the .pt scan (I/O-bound; "
                          "match to allocated CPUs for a full-corpus run).")
    ap.add_argument("--out_dir", type=pathlib.Path,
                     default=pathlib.Path("analysis_out/chain_quality_vs_resolution"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    missing_cols = {"id", "pdb", "chain", "length", "n_chains", "resolution"} - set(meta.columns)
    if missing_cols:
        raise ValueError(f"--metadata_csv is missing expected columns: {missing_cols}")
    meta = meta.set_index("id")

    pt_files = sorted(args.processed_dir.glob("*.pt"))
    print(f"[chain_quality] {len(pt_files)} .pt files in {args.processed_dir}, "
          f"{len(meta)} rows in metadata CSV")
    if args.sample_size and args.sample_size < len(pt_files):
        rng = random.Random(args.seed)
        pt_files = rng.sample(pt_files, args.sample_size)
        print(f"[chain_quality] subsampled to {len(pt_files)}")

    # Only load/scan files we can actually join against metadata -- skip the rest
    # up front rather than wasting worker time on them.
    scan_files = [p for p in pt_files if p.stem in meta.index]
    n_no_metadata = len(pt_files) - len(scan_files)

    summaries = {}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            work = [(p, args.cutoff) for p in scan_files]
            for chain_id, summary, err in tqdm(
                pool.map(_audit_one_or_none, work, chunksize=32), total=len(work), desc="scanning"
            ):
                if err is not None:
                    print(f"[chain_quality] {chain_id}: load failed {err}", file=sys.stderr)
                    continue
                if summary is not None:
                    summaries[chain_id] = summary
    else:
        for p in tqdm(scan_files, desc="scanning"):
            chain_id, summary, err = _audit_one_or_none((p, args.cutoff))
            if err is not None:
                print(f"[chain_quality] {chain_id}: load failed {err}", file=sys.stderr)
                continue
            if summary is not None:
                summaries[chain_id] = summary

    rows = []
    for chain_id, summary in summaries.items():
        m = meta.loc[chain_id]
        length = m["length"]
        n_residues = summary["n_residues"]
        frac_resolved = (n_residues / length) if length and length > 0 else float("nan")
        rows.append({
            "id": chain_id,
            "pdb": m["pdb"],
            "chain": m["chain"],
            "n_chains_in_entry": m["n_chains"],
            "resolution": m["resolution"],
            "deposition_date": m.get("deposition_date"),
            "experiment_type": m.get("experiment_type"),
            "length_deposited": length,
            "n_residues_resolved": n_residues,
            "frac_resolved": frac_resolved,
            "n_breaks": summary["n_breaks"],
            "n_suspicious": summary["n_suspicious"],
            "n_extreme": summary["n_extreme"],
            "max_gap": summary["max_gap"],
        })

    df = pd.DataFrame(rows)
    if n_no_metadata:
        print(f"[chain_quality] WARNING: {n_no_metadata} .pt files had no matching metadata row (skipped)")
    if df.empty:
        print("[chain_quality] no chains scanned; abort")
        return
    df.to_csv(args.out_dir / "per_chain_quality.csv", index=False)

    # For every multi-chain entry, compute the SIBLING max frac_resolved -- the
    # direct evidence for "this chain is uniquely bad, its siblings are fine".
    sibling_max = df.groupby("pdb")["frac_resolved"].transform("max")
    sibling_n = df.groupby("pdb")["frac_resolved"].transform("count")
    df["sibling_max_frac_resolved"] = sibling_max
    df["n_siblings_scanned"] = sibling_n

    worst = df[
        (df["frac_resolved"] < args.frac_resolved_bad_cutoff)
        & (df["resolution"] <= args.resolution_good_cutoff)
    ].copy()
    worst["sibling_gap"] = worst["sibling_max_frac_resolved"] - worst["frac_resolved"]
    worst = worst.sort_values(["sibling_gap", "frac_resolved"], ascending=[False, True])
    worst.to_csv(args.out_dir / "worst_offenders.csv", index=False)

    top_k = worst.head(args.top_k)
    top_k[["pdb", "chain", "resolution", "n_chains_in_entry", "frac_resolved",
           "length_deposited", "n_residues_resolved",
           "sibling_max_frac_resolved", "n_siblings_scanned"]].to_csv(
        args.out_dir / "worst_offenders_for_rcsb_check.csv", index=False
    )

    n = len(df)
    n_flagged = len(worst)
    n_flagged_multichain_with_better_sibling = int(
        ((worst["n_chains_in_entry"] > 1) & (worst["sibling_gap"] > 0.2)).sum()
    )
    lines = ["# Chain quality vs. deposit resolution audit\n"]
    lines.append(f"- processed_dir: `{args.processed_dir}`")
    lines.append(f"- metadata_csv: `{args.metadata_csv}`")
    lines.append(f"- chains scanned: {n}  (skipped, no metadata match: {n_no_metadata})")
    lines.append(f"- 'good resolution' cutoff: <= {args.resolution_good_cutoff} A")
    lines.append(f"- 'largely unresolved' cutoff: frac_resolved < {args.frac_resolved_bad_cutoff}\n")
    lines.append("## Headline\n")
    lines.append(f"- chains flagged (good overall resolution, but this chain largely unresolved): "
                 f"**{n_flagged}** / {n}  ({n_flagged / n:.2%})")
    lines.append(f"- of those, chains in a multi-chain entry where a SIBLING chain is >20pp more "
                 f"complete (direct 'one bad chain in an otherwise fine complex' signature): "
                 f"**{n_flagged_multichain_with_better_sibling}**")
    lines.append("")
    lines.append("## frac_resolved distribution (all scanned chains)\n")
    for q in [1, 5, 10, 25, 50, 75, 90]:
        lines.append(f"- p{q}: {np.nanpercentile(df['frac_resolved'], q):.3f}")
    lines.append("")
    lines.append(f"## Top {min(args.top_k, len(worst))} worst offenders (by sibling_gap, then frac_resolved)\n")
    lines.append("See worst_offenders.csv (full) and worst_offenders_for_rcsb_check.csv "
                 "(compact, feed into the RCSB cross-check script) for detail.\n")
    for _, r in top_k.iterrows():
        lines.append(
            f"- {r['pdb']}_{r['chain']}: frac_resolved={r['frac_resolved']:.2f}, "
            f"resolution={r['resolution']:.2f} A, n_chains_in_entry={int(r['n_chains_in_entry'])}, "
            f"sibling_max={r['sibling_max_frac_resolved']:.2f} (n_siblings_scanned={int(r['n_siblings_scanned'])})"
        )
    lines.append("")
    (args.out_dir / "chain_quality_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[chain_quality] all outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
