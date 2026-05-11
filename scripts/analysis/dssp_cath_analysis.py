#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
DSSP × CATH distribution analysis for Proteina training data.

For each protein chain that has a pre-computed dssp_target (from
precompute_dssp_targets.py), computes L/H/E residue fractions.  Then
aggregates by CATH topology (CAT = Class.Architecture.Topology, x.x.x) and
CATH class (C = first digit) and produces:

  dssp_cath_summary.csv        — per-chain: n_valid, n_loop/helix/strand, l/h/e_frac
  dssp_cath_per_topology.csv   — per-CAT-topology: mean/median/std of L/H/E
  dssp_cath_3d.html            — interactive Plotly 3D scatter (one point per topology)
  dssp_cath_c_level_bars.pdf   — C-level stacked bar sanity check

Usage
-----
# Full run:
python scripts/analysis/dssp_cath_analysis.py \\
    --processed-dir data/pdb_train/processed \\
    --sifts data/cathdata/pdb_chain_cath_uniprot.tsv.gz \\
    --cath-codes data/cathdata/cath-b-newest-all.gz \\
    --workers 32 \\
    --output-dir results/dssp_cath/

# Quick preview on 5 000 chains (< 5 min):
python scripts/analysis/dssp_cath_analysis.py ... --sample 5000

# Re-generate plots from a cached summary CSV (no .pt loading needed):
python scripts/analysis/dssp_cath_analysis.py \\
    --summary-csv results/dssp_cath/dssp_cath_summary.csv \\
    --sifts ...  --cath-codes ...  --output-dir results/dssp_cath/
"""

import argparse
import gzip
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# CATH metadata helpers
# ---------------------------------------------------------------------------

def load_sifts(sifts_gz: Path) -> pd.DataFrame:
    """Parse pdb_chain_cath_uniprot.tsv.gz → DataFrame(chain_id, CATH_ID)."""
    df = pd.read_csv(
        sifts_gz,
        sep="\t",
        comment="#",
        usecols=["PDB", "CHAIN", "CATH_ID"],
        dtype=str,
        low_memory=False,
    )
    df = df.dropna(subset=["PDB", "CHAIN", "CATH_ID"])
    # chain_id format matches .pt file stems: lowercase pdb + "_" + chain
    df["chain_id"] = df["PDB"].str.lower() + "_" + df["CHAIN"]
    return df[["chain_id", "CATH_ID"]]


def load_cath_codes(cath_b_gz: Path) -> Dict[str, str]:
    """Parse cath-b-newest-all.gz → {CATH_ID: 'C.A.T.H' code string}.

    File format (space-separated):
        CATH_ID  version  code  residue_range
        101mA00  v4_3_0   1.10.490.10  0-153:A
    """
    codes: Dict[str, str] = {}
    with gzip.open(cath_b_gz, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                codes[parts[0]] = parts[2]
    return codes


def build_chain_cath_table(
    sifts: pd.DataFrame,
    cath_codes: Dict[str, str],
) -> pd.DataFrame:
    """Join SIFTS with CATH codes; parse C level and CAT (topology) level."""
    df = sifts.copy()
    df["cath_code"] = df["CATH_ID"].map(cath_codes)
    df = df.dropna(subset=["cath_code"])
    parts = df["cath_code"].str.split(".", expand=True)
    df["c_level"] = parts[0].astype(int)
    df["cat_level"] = parts[0] + "." + parts[1] + "." + parts[2]
    return df[["chain_id", "CATH_ID", "cath_code", "c_level", "cat_level"]]


# ---------------------------------------------------------------------------
# Per-file worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _extract_stats(path: Path) -> Dict:
    """Load one .pt file and return DSSP residue counts."""
    try:
        g = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"chain_id": path.stem, "status": "failed", "error": str(exc)}

    dssp = getattr(g, "dssp_target", None)
    if dssp is None:
        return {"chain_id": path.stem, "status": "no_dssp"}

    valid_mask = dssp >= 0
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return {
            "chain_id": path.stem, "status": "all_invalid",
            "n_valid": 0, "n_loop": 0, "n_helix": 0, "n_strand": 0,
        }

    v = dssp[valid_mask]
    return {
        "chain_id": path.stem,
        "status": "ok",
        "n_valid": n_valid,
        "n_loop":   int((v == 0).sum()),
        "n_helix":  int((v == 1).sum()),
        "n_strand": int((v == 2).sum()),
    }


# ---------------------------------------------------------------------------
# Parallel loader (same sliding-window pattern as precompute_confind_maps.py)
# ---------------------------------------------------------------------------

def _load_stats_parallel(paths: List[Path], workers: int) -> List[Dict]:
    results: List[Dict] = []
    total = len(paths)
    completed = 0
    start = last_log = time.perf_counter()
    paths_iter = iter(paths)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        future_to_path: Dict = {}

        for _ in range(workers):
            try:
                p = next(paths_iter)
            except StopIteration:
                break
            f = executor.submit(_extract_stats, p)
            futures.add(f)
            future_to_path[f] = p

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    results.append(fut.result())
                except Exception as exc:
                    p = future_to_path.get(fut)
                    results.append({
                        "chain_id": p.stem if p else "?",
                        "status": "failed",
                        "error": str(exc),
                    })
                completed += 1
                now = time.perf_counter()
                if completed % 1000 == 0 or (now - last_log) >= 30:
                    elapsed = now - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (total - completed) / rate if rate > 0 else 0.0
                    print(
                        f"  {completed:>7,}/{total:,}  "
                        f"rate={rate:.1f}/s  ETA={eta/60:.1f}m",
                        flush=True,
                    )
                    last_log = now

                try:
                    p = next(paths_iter)
                    f = executor.submit(_extract_stats, p)
                    futures.add(f)
                    future_to_path[f] = p
                except StopIteration:
                    pass

    return results


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

C_NAMES = {
    1: "Mainly Alpha",
    2: "Mainly Beta",
    3: "Alpha + Beta",
    4: "Few Sec. Struct.",
}
C_COLORS_HEX = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c", 4: "#7f7f7f"}


def _plot_3d_scatter(cat_agg: pd.DataFrame, out_path: Path) -> None:
    import numpy as np
    import plotly.graph_objects as go

    traces = []
    for c_val, grp in cat_agg.groupby("c_level"):
        color = C_COLORS_HEX.get(int(c_val), "#bcbd22")
        name = C_NAMES.get(int(c_val), f"C={c_val}")
        max_n = grp["n_chains"].max()
        sizes = 3 + 9 * (np.log1p(grp["n_chains"]) / np.log1p(max(max_n, 1)))
        customdata = grp[
            ["cat_level", "n_chains", "std_L", "std_H", "std_E",
             "median_L", "median_H", "median_E"]
        ].values
        traces.append(
            go.Scatter3d(
                x=grp["mean_L"],
                y=grp["mean_H"],
                z=grp["mean_E"],
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=sizes,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "n_chains=%{customdata[1]}<br>"
                    "mean L=%{x:.3f} ± %{customdata[2]:.3f}  (med %{customdata[5]:.3f})<br>"
                    "mean H=%{y:.3f} ± %{customdata[3]:.3f}  (med %{customdata[6]:.3f})<br>"
                    "mean E=%{z:.3f} ± %{customdata[4]:.3f}  (med %{customdata[7]:.3f})"
                    "<extra></extra>"
                ),
            )
        )

    # Sparse simplex reference: points on L+H+E=1 plane
    grid = np.linspace(0, 1, 8)
    sx, sy, sz = [], [], []
    for lv in grid:
        for hv in grid:
            ev = 1.0 - lv - hv
            if 0.0 <= ev <= 1.0:
                sx.append(lv); sy.append(hv); sz.append(ev)
    traces.append(
        go.Scatter3d(
            x=sx, y=sy, z=sz,
            mode="markers",
            marker=dict(color="lightgray", size=1.5, opacity=0.15),
            name="Simplex (L+H+E=1)",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="DSSP composition per CATH CAT topology<br>"
                 "<sub>Each point = one topology; size ∝ log(n_chains); "
                 "hover for stats</sub>",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(title="mean L (loop)", range=[0, 1]),
            yaxis=dict(title="mean H (helix)", range=[0, 1]),
            zaxis=dict(title="mean E (strand)", range=[0, 1]),
        ),
        legend=dict(title="CATH C-level"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.write_html(str(out_path))
    print(f"Saved interactive 3D plot → {out_path}", flush=True)


def _plot_c_level_bars(cat_agg: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Equal weight per CAT topology: average the per-topology means
    c_means = (
        cat_agg
        .groupby("c_level")[["mean_L", "mean_H", "mean_E", "std_L", "std_H", "std_E"]]
        .mean()
        .reindex([1, 2, 3, 4])
        .dropna()
    )
    n_topo_per_c = cat_agg.groupby("c_level")["cat_level"].nunique().reindex([1, 2, 3, 4])
    n_chains_per_c = cat_agg.groupby("c_level")["n_chains"].sum().reindex([1, 2, 3, 4])

    labels = [
        f"C={c}\n{C_NAMES.get(c, '')}\n"
        f"({int(n_topo_per_c.get(c, 0))} topologies,\n"
        f"n={int(n_chains_per_c.get(c, 0)):,} chains)"
        for c in c_means.index
    ]

    bar_w = 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0.0] * len(c_means)

    layer_specs = [
        ("mean_L", "#aec7e8", "Loop (L)"),
        ("mean_H", "#d62728", "Helix (H)"),
        ("mean_E", "#1f77b4", "Strand (E)"),
    ]
    for col, color, label in layer_specs:
        vals = c_means[col].values
        ax.bar(labels, vals, bottom=bottoms, color=color, label=label, width=bar_w)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.04:
                ax.text(
                    i, b + v / 2, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean fraction\n(equal weight per CAT topology)", fontsize=10)
    ax.set_title(
        "DSSP composition by CATH C-level class\n"
        "(per-CAT-topology mean L/H/E, then averaged across topologies per C-level)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved C-level bar chart → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_analysis(
    processed_dir: Optional[Path],
    sifts_gz: Path,
    cath_b_gz: Path,
    workers: int,
    output_dir: Path,
    summary_csv: Path,
    rebuild: bool,
    sample: Optional[int],
    seed: int,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load CATH metadata ------------------------------------------------
    print("Loading CATH metadata ...", flush=True)
    t0 = time.perf_counter()
    sifts = load_sifts(sifts_gz)
    cath_codes = load_cath_codes(cath_b_gz)
    cath_table = build_chain_cath_table(sifts, cath_codes)
    print(
        f"  SIFTS domain entries: {len(sifts):,}  "
        f"(unique chains: {sifts['chain_id'].nunique():,})  "
        f"[{time.perf_counter()-t0:.1f}s]",
        flush=True,
    )
    print(f"  CATH codes loaded: {len(cath_codes):,}", flush=True)
    print(
        f"  Matched (chain×domain) pairs: {len(cath_table):,}  "
        f"({cath_table['chain_id'].nunique():,} unique chains)",
        flush=True,
    )
    print(
        f"  Unique CAT topologies: {cath_table['cat_level'].nunique():,}  "
        f"C-levels present: {sorted(cath_table['c_level'].unique())}",
        flush=True,
    )

    cath_chain_ids = set(cath_table["chain_id"].unique())

    # ---- Build or load per-chain summary -----------------------------------
    if not rebuild and summary_csv.exists():
        print(f"Loading cached per-chain summary from {summary_csv} ...", flush=True)
        chain_stats = pd.read_csv(summary_csv, dtype={"chain_id": str})
        print(f"  {len(chain_stats):,} chains loaded from cache.", flush=True)
    else:
        if processed_dir is None:
            raise ValueError(
                "--processed-dir is required when the summary CSV does not exist. "
                "Point to the dataset's processed/ directory."
            )

        # Discover .pt files that have CATH annotations
        print(f"Scanning {processed_dir} for .pt files ...", flush=True)
        all_files = {p.stem: p for p in processed_dir.glob("*.pt")}
        cath_files = {s: p for s, p in all_files.items() if s in cath_chain_ids}
        print(
            f"  {len(cath_files):,} chains have both a .pt file and a CATH annotation "
            f"(out of {len(all_files):,} total .pt files)",
            flush=True,
        )

        if sample is not None and sample < len(cath_files):
            sampled = random.sample(list(cath_files.keys()), sample)
            cath_files = {s: cath_files[s] for s in sampled}
            print(
                f"  --sample {sample}: using {len(cath_files):,} randomly selected chains.",
                flush=True,
            )

        paths = list(cath_files.values())
        print(
            f"Loading {len(paths):,} .pt files with {workers} workers ...",
            flush=True,
        )
        t1 = time.perf_counter()
        raw = _load_stats_parallel(paths, workers)
        print(f"  Done in {time.perf_counter()-t1:.1f}s", flush=True)

        n_ok = sum(1 for r in raw if r.get("status") == "ok")
        n_fail = sum(1 for r in raw if r.get("status") == "failed")
        n_nods = sum(1 for r in raw if r.get("status") == "no_dssp")
        print(
            f"  ok={n_ok:,}  failed={n_fail:,}  no_dssp={n_nods:,}  "
            f"all_invalid={len(raw)-n_ok-n_fail-n_nods:,}",
            flush=True,
        )

        ok_rows = [r for r in raw if r.get("status") == "ok"]
        chain_stats = pd.DataFrame(ok_rows)
        chain_stats["l_frac"] = chain_stats["n_loop"]   / chain_stats["n_valid"]
        chain_stats["h_frac"] = chain_stats["n_helix"]  / chain_stats["n_valid"]
        chain_stats["e_frac"] = chain_stats["n_strand"] / chain_stats["n_valid"]
        chain_stats.to_csv(summary_csv, index=False)
        print(f"Saved per-chain summary → {summary_csv}", flush=True)

    # ---- Join with CATH annotations ----------------------------------------
    merged = chain_stats.merge(cath_table, on="chain_id", how="inner")
    print(f"Joined rows (chain × CATH domain): {len(merged):,}", flush=True)

    if merged.empty:
        print("WARNING: no rows after join — check that chain_id format matches.", flush=True)
        return

    # ---- Per-CAT-topology aggregation ----------------------------------------
    print("Aggregating per CAT topology ...", flush=True)
    cat_agg = (
        merged
        .groupby("cat_level")
        .agg(
            c_level=("c_level", "first"),
            n_chains=("chain_id", "nunique"),
            mean_L=("l_frac", "mean"),    median_L=("l_frac", "median"),    std_L=("l_frac", "std"),
            mean_H=("h_frac", "mean"),    median_H=("h_frac", "median"),    std_H=("h_frac", "std"),
            mean_E=("e_frac", "mean"),    median_E=("e_frac", "median"),    std_E=("e_frac", "std"),
        )
        .reset_index()
    )
    topo_csv = output_dir / "dssp_cath_per_topology.csv"
    cat_agg.to_csv(topo_csv, index=False)
    print(f"  {len(cat_agg):,} unique CAT topologies → {topo_csv}", flush=True)

    # ---- Print per-C-level summary -----------------------------------------
    print("\nC-level summary (equal weight per CAT topology):", flush=True)
    c_summary = (
        cat_agg
        .groupby("c_level")
        .agg(
            n_topologies=("cat_level", "count"),
            n_chains=("n_chains", "sum"),
            mean_L=("mean_L", "mean"),
            mean_H=("mean_H", "mean"),
            mean_E=("mean_E", "mean"),
        )
        .reindex([1, 2, 3, 4])
        .dropna()
    )
    for c, row in c_summary.iterrows():
        name = C_NAMES.get(int(c), "?")
        print(
            f"  C={c} ({name:18s}): "
            f"{int(row.n_topologies):4d} topologies  "
            f"{int(row.n_chains):6,} chains  "
            f"L={row.mean_L:.3f}  H={row.mean_H:.3f}  E={row.mean_E:.3f}",
            flush=True,
        )

    # ---- Visualizations ----------------------------------------------------
    print("\nGenerating plots ...", flush=True)
    _plot_3d_scatter(cat_agg, output_dir / "dssp_cath_3d.html")
    _plot_c_level_bars(cat_agg, output_dir / "dssp_cath_c_level_bars.pdf")

    print(f"\nAll outputs written to {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze DSSP composition within CATH structural classes."
    )
    parser.add_argument(
        "--processed-dir", default=None,
        help="Path to dataset processed/ directory containing .pt files.",
    )
    parser.add_argument(
        "--sifts",
        default="data/cathdata/pdb_chain_cath_uniprot.tsv.gz",
        help="SIFTS pdb_chain_cath_uniprot TSV (gzipped).",
    )
    parser.add_argument(
        "--cath-codes",
        default="data/cathdata/cath-b-newest-all.gz",
        help="CATH-B domain code mapping (gzipped).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for .pt loading (default: all CPUs).",
    )
    parser.add_argument(
        "--output-dir", default="results/dssp_cath",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--summary-csv", default=None,
        help="Per-chain summary CSV. If it exists and --rebuild is not set, "
             ".pt loading is skipped and plots are regenerated from the cache.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force re-read .pt files even if summary CSV already exists.",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N CATH-annotated chains (for a quick preview run).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workers = args.workers or int(
        os.getenv("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1
    )
    output_dir = Path(args.output_dir)
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else output_dir / "dssp_cath_summary.csv"
    )

    run_analysis(
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        sifts_gz=Path(args.sifts),
        cath_b_gz=Path(args.cath_codes),
        workers=workers,
        output_dir=output_dir,
        summary_csv=summary_csv,
        rebuild=args.rebuild,
        sample=args.sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
