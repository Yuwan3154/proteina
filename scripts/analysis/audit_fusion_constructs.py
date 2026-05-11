"""Flag suspected fusion-construct entries in a Proteina master CSV.

Many crystallography entries in the PDB are biologically irrelevant fusions —
the target protein is fused to a soluble crystallization scaffold (MBP, SUMO,
thioredoxin, GST, fluorescent protein, antibody Fab, etc.). The fusion partner
typically dominates the residue count, so the model trains on geometry of the
*scaffold* rather than the target.

This script scans the ``name`` column of the master CSV (built from the PDB
SEQRES FASTA via ``" ".join(params[3:])`` at graphein_utils.py:2289 — verified
not truncated) and bins each chain by which scaffold/fusion category, if any,
its description matches.

Outputs (under ``--out_dir``):
    - audit_fusion_constructs_report.md   — headline counts + per-category top names
    - flagged_chains.csv                   — full list with categories + length
    - per_category_top_names.csv           — top-K names per category for spot-check

Usage:
    python scripts/analysis/audit_fusion_constructs.py \\
        --df_csv data/pdb_train/df_pdb_f1_minl50_maxl384_<...>.csv \\
        --out_dir analysis_out/fusion_constructs
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import Counter
from typing import Dict, List

import pandas as pd

# Patterns. Each pattern is a compiled regex applied (case-insensitive) against
# the full ``name`` string. The categorization is best-effort — false positives
# are likely (e.g. a real maltose-binding protein not used as a fusion tag), so
# the output is meant for inspection rather than automated filtering.
_PATTERNS: Dict[str, List[re.Pattern]] = {
    "mbp": [
        re.compile(r"maltose[\s-]?binding", re.IGNORECASE),
        re.compile(r"\bMBP\b"),  # uppercase MBP only, lowercase matches too many false positives
    ],
    "gfp_family": [
        re.compile(r"\b(GFP|sfGFP|EGFP|YFP|CFP|BFP|mCherry|mTurquoise|mNeonGreen|tdTomato)\b"),
        re.compile(r"green\s+fluorescent", re.IGNORECASE),
        re.compile(r"yellow\s+fluorescent", re.IGNORECASE),
        re.compile(r"red\s+fluorescent", re.IGNORECASE),
        re.compile(r"cyan\s+fluorescent", re.IGNORECASE),
        re.compile(r"fluorescent\s+protein", re.IGNORECASE),
    ],
    "gst": [
        re.compile(r"glutathione[\s-]?S?[\s-]?transferase", re.IGNORECASE),
        re.compile(r"\bGST\b"),
    ],
    "sumo": [
        re.compile(r"\bSUMO\b"),
        re.compile(r"small\s+ubiquitin[\s-]?(?:related|like)\s+modifier", re.IGNORECASE),
    ],
    "thioredoxin": [
        re.compile(r"thioredoxin", re.IGNORECASE),
        re.compile(r"\bTrxA\b"),
    ],
    "halo_snap_clip": [
        re.compile(r"\bHaloTag\b", re.IGNORECASE),
        re.compile(r"\bSNAP[\s-]?tag\b", re.IGNORECASE),
        re.compile(r"\bCLIP[\s-]?tag\b", re.IGNORECASE),
    ],
    "antibody": [
        re.compile(r"\b(Fab|Fv|scFv)\b"),
        re.compile(r"(?:heavy|light)\s+chain", re.IGNORECASE),
        re.compile(r"\bIgG\b"),
        re.compile(r"variable\s+region", re.IGNORECASE),
        re.compile(r"immunoglobulin", re.IGNORECASE),
    ],
    "ubiquitin": [
        re.compile(r"\bubiquitin\b", re.IGNORECASE),  # not always a fusion but often is
    ],
    "explicit_chimera": [
        re.compile(r"chimera", re.IGNORECASE),
        re.compile(r"chimeric", re.IGNORECASE),
        re.compile(r"\bfusion\s+protein\b", re.IGNORECASE),
        re.compile(r"\bfusion\s+construct\b", re.IGNORECASE),
    ],
    "multi_segment_marker": [
        # PDB headers list multi-protein entries with these markers.
        re.compile(r"\[(?:Contains|Includes):", re.IGNORECASE),
        # Comma-separated names: e.g. "Maltose-binding...,Telomerase-associated..."
        # Restrict to entries where the comma is between two title-case words to
        # avoid lists like "EC 2.4.1.40" or "Tetrahymena thermophila, Escherichia coli".
        re.compile(r"[a-z],[A-Z]"),
    ],
}


def _name_truncation_check(names: pd.Series) -> Dict[str, object]:
    """Sanity-check that the name column survived all CSV/PDB-header round trips.

    Truncation symptoms we look for:
      - any name ending in '...' (pandas/CSV truncation marker)
      - any name exactly at a suspicious round line width (e.g. all 80, 132)
      - mean length < 5 chars (no signal at all)
    """
    nz = names.fillna("").astype(str)
    lens = nz.str.len()
    stats = {
        "n_rows": int(len(nz)),
        "n_nonempty": int((lens > 0).sum()),
        "min_len": int(lens.min()),
        "max_len": int(lens.max()),
        "mean_len": float(lens.mean()),
        "median_len": float(lens.median()),
        "ends_with_ellipsis": int(nz.str.endswith("...").sum()),
        "exact_80_chars": int((lens == 80).sum()),
        "exact_132_chars": int((lens == 132).sum()),
    }
    return stats


def _classify(name: str) -> List[str]:
    """Return list of category names matching the description (may be empty)."""
    if not isinstance(name, str) or not name:
        return []
    cats = []
    for cat, patterns in _PATTERNS.items():
        if any(p.search(name) for p in patterns):
            cats.append(cat)
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_csv", type=pathlib.Path, required=True,
                    help="Master CSV from PDBDataSelector (df_pdb_<...>.csv)")
    ap.add_argument("--out_dir", type=pathlib.Path,
                    default=pathlib.Path("analysis_out/fusion_constructs"))
    ap.add_argument("--top_k", type=int, default=20,
                    help="Top-K most-common names to surface per category")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fusion_audit] loading {args.df_csv}")
    df = pd.read_csv(args.df_csv)
    if "name" not in df.columns:
        print("ERROR: no 'name' column in CSV", file=sys.stderr)
        sys.exit(2)

    # Truncation sanity check — print + persist for the user.
    trunc_stats = _name_truncation_check(df["name"])
    print("[fusion_audit] name-column truncation check:")
    for k, v in trunc_stats.items():
        print(f"  {k}: {v}")
    if trunc_stats["ends_with_ellipsis"] > 0:
        print(
            f"  WARNING: {trunc_stats['ends_with_ellipsis']} names end with '...' "
            "— possible truncation upstream.",
            file=sys.stderr,
        )

    # Classify.
    df["fusion_categories"] = df["name"].apply(_classify)
    df["is_fusion_candidate"] = df["fusion_categories"].apply(lambda c: bool(c))
    df["fusion_categories_str"] = df["fusion_categories"].apply(lambda c: "|".join(c))

    flagged = df[df["is_fusion_candidate"]]
    n_total = len(df)
    n_flagged = len(flagged)
    print(f"[fusion_audit] flagged {n_flagged}/{n_total} ({n_flagged/n_total:.1%}) chains")

    # Per-category counts and top names.
    cat_counts = Counter()
    cat_names: Dict[str, Counter] = {cat: Counter() for cat in _PATTERNS}
    for _, row in flagged.iterrows():
        for cat in row["fusion_categories"]:
            cat_counts[cat] += 1
            cat_names[cat][row["name"]] += 1

    cat_summary_rows = []
    cat_top_rows = []
    for cat in sorted(_PATTERNS.keys()):
        n = cat_counts.get(cat, 0)
        cat_summary_rows.append({"category": cat, "n_chains": n, "frac_of_total": n / n_total})
        for name, count in cat_names[cat].most_common(args.top_k):
            cat_top_rows.append({"category": cat, "n": count, "name": name})

    cat_summary_df = pd.DataFrame(cat_summary_rows).sort_values("n_chains", ascending=False)
    cat_top_df = pd.DataFrame(cat_top_rows)

    # Write outputs.
    flagged_out = flagged[["id", "pdb", "chain", "length", "name", "fusion_categories_str"]].copy()
    flagged_out.to_csv(args.out_dir / "flagged_chains.csv", index=False)
    cat_summary_df.to_csv(args.out_dir / "category_summary.csv", index=False)
    cat_top_df.to_csv(args.out_dir / "per_category_top_names.csv", index=False)

    # Markdown report.
    lines = ["# Fusion-construct audit\n"]
    lines.append(f"- master CSV: `{args.df_csv}`")
    lines.append(f"- chains scanned: **{n_total}**")
    lines.append(f"- chains flagged as fusion candidates: **{n_flagged} ({n_flagged/n_total:.1%})**")
    lines.append("")
    lines.append("## Name-column truncation sanity check\n")
    for k, v in trunc_stats.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    if trunc_stats["ends_with_ellipsis"] > 0 or trunc_stats["max_len"] in (80, 132):
        lines.append(
            "**WARNING**: suspicious truncation pattern detected — the audit may be missing "
            "fusion partners listed past the truncation point. Inspect the source.\n"
        )
    else:
        lines.append("**OK**: no truncation symptoms detected (no '...' suffixes, max_len exceeds 80/132 line widths).\n")

    lines.append("## Category counts\n")
    lines.append("| category | n_chains | frac_of_total |")
    lines.append("|---|---|---|")
    for _, row in cat_summary_df.iterrows():
        lines.append(f"| {row['category']} | {row['n_chains']} | {row['frac_of_total']:.2%} |")
    lines.append("")

    lines.append("## Top-{} most-common descriptions per category\n".format(args.top_k))
    for cat in cat_summary_df["category"].tolist():
        sub = cat_top_df[cat_top_df["category"] == cat].sort_values("n", ascending=False)
        if sub.empty:
            continue
        lines.append(f"### {cat} (n_chains={int(cat_counts.get(cat, 0))})\n")
        for _, row in sub.iterrows():
            lines.append(f"- {row['n']}× `{row['name'][:200]}{'…' if len(row['name'])>200 else ''}`")
        lines.append("")

    out_md = args.out_dir / "audit_fusion_constructs_report.md"
    out_md.write_text("\n".join(lines))
    print(f"\n[fusion_audit] wrote outputs under {args.out_dir}")
    print(f"  - {out_md}")
    print(f"  - {args.out_dir / 'flagged_chains.csv'}")
    print(f"  - {args.out_dir / 'category_summary.csv'}")
    print(f"  - {args.out_dir / 'per_category_top_names.csv'}")


if __name__ == "__main__":
    main()
