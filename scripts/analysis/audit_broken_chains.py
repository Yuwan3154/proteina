"""Broken-chain audit for the processed Proteina PDB dataset.

Walks ``processed/*.pt``, computes consecutive-CA distances, and reports how
many chains have CA-CA gaps larger than a configurable threshold (default
4.0 Å — same as ChainBreakPerResidueTransform). Filtering is NOT applied; the
goal is to quantify the problem and surface the worst offenders so the user
can eyeball them in PyMOL.

Outputs (under ``--out_dir``):
    - audit_broken_chains_report.md  — headline numbers
    - per_chain_breaks.csv           — one row per chain (id, length, n_breaks, max_gap)
    - hist_largest_gap.png           — distribution of largest gap per chain
    - hist_n_breaks.png              — distribution of break count per chain
    - examples/<id>.pdb              — top-K worst chains for visual inspection

Usage:
    python scripts/analysis/audit_broken_chains.py \
        --processed_dir data/pdb_train/processed \
        --cutoff 4.0 --sample_size 10000 --top_k_examples 20 \
        --out_dir analysis_out/broken_chains
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_THIS = pathlib.Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

# CA is index 1 in both the on-disk PDB ordering and the openfold ordering used at
# runtime — `PDB_TO_OPENFOLD_INDEX_TENSOR[1] == 1`. So we can read directly without
# the runtime reorder.
_CA_IDX_IN_PT = 1


def _audit_one(pt_path: pathlib.Path, cutoff: float):
    """Compute per-chain break statistics including residue_pdb_idx correlation.

    Returns the per-chain summary plus a list of per-break tuples:
    ``(spatial_gap_A, idx_gap, n_missing, gap_per_missing_A, kind)`` where ``kind``
    classifies the break:
      - "expected": idx_gap >= 2 (missing residue gap) and gap_per_missing <= 5 Å
      - "extreme":  idx_gap >= 2 but gap_per_missing > 8 Å (likely longer unresolved
        loop with end-points pulled apart, or fragment-level concat)
      - "suspicious": idx_gap == 1 (residues are *consecutive* in the PDB but their
        CAs are >cutoff Å apart) — this is the only category that signals
        actually-corrupted data, not normal unmodeled regions.
    """
    g = torch.load(pt_path, weights_only=False, map_location="cpu")
    if not hasattr(g, "coords") or g.coords is None:
        return None, []
    ca_coords = g.coords[:, _CA_IDX_IN_PT, :]
    ca_mask = g.coord_mask[:, _CA_IDX_IN_PT].bool()
    if ca_mask.sum().item() < 2:
        return None, []
    valid_ca = ca_coords[ca_mask]
    diffs = (valid_ca[1:] - valid_ca[:-1]).norm(dim=-1)
    n_breaks = int((diffs > cutoff).sum().item())
    max_gap = float(diffs.max().item()) if diffs.numel() else 0.0
    n_residues = int(ca_mask.sum().item())

    # Classify each break by index-gap correlation. residue_pdb_idx is stored as
    # a tensor of original PDB residue numbers; if it's missing for old caches,
    # gracefully skip the classification.
    per_break: List = []
    if hasattr(g, "residue_pdb_idx") and g.residue_pdb_idx is not None:
        pdb_idx = g.residue_pdb_idx
        if pdb_idx.dim() > 1:
            pdb_idx = pdb_idx.flatten()
        valid_pdb_idx = pdb_idx[ca_mask].long()
        for i, gap in enumerate(diffs.tolist()):
            if gap <= cutoff:
                continue
            idx_gap = int((valid_pdb_idx[i + 1] - valid_pdb_idx[i]).item())
            n_missing = max(idx_gap - 1, 0)
            gap_per_missing = gap / max(idx_gap, 1)
            if idx_gap == 1:
                kind = "suspicious"
            elif idx_gap >= 2 and gap_per_missing > 8.0:
                kind = "extreme"
            else:
                kind = "expected"
            per_break.append((float(gap), idx_gap, n_missing, float(gap_per_missing), kind))

    summary = {
        "id": pt_path.stem,
        "n_residues": n_residues,
        "n_breaks": n_breaks,
        "max_gap": max_gap,
        "median_gap": float(diffs.median().item()) if diffs.numel() else 0.0,
        "n_suspicious": sum(1 for b in per_break if b[4] == "suspicious"),
        "n_extreme": sum(1 for b in per_break if b[4] == "extreme"),
        "n_expected": sum(1 for b in per_break if b[4] == "expected"),
    }
    return summary, per_break


def _write_example_pdb(pt_path: pathlib.Path, out_pdb: pathlib.Path):
    """Best-effort PDB dump for visual inspection.

    Defers to write_prot_to_pdb if available; otherwise writes a minimal
    CA-only PDB so PyMOL can still open it.
    """
    g = torch.load(pt_path, weights_only=False, map_location="cpu")
    coords = g.coords  # [n, 37, 3] in PDB ordering
    mask = g.coord_mask  # [n, 37]
    aatype = getattr(g, "residue_type", None)

    try:
        from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
        from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

        coords_of = coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].numpy()
        mask_of = mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].float().numpy()
        aatype_np = aatype.long().numpy() if aatype is not None else np.zeros(coords.shape[0], dtype=np.int64)
        write_prot_to_pdb(
            coords_of, str(out_pdb), aatype=aatype_np, atom37_mask=mask_of,
            overwrite=True, no_indexing=True,
        )
        return
    except Exception as e:
        print(f"[example_pdb] write_prot_to_pdb failed for {pt_path.stem}: {e!r}; falling back to CA-only", file=sys.stderr)

    ca = coords[:, _CA_IDX_IN_PT, :].numpy()
    cam = mask[:, _CA_IDX_IN_PT].bool().numpy()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w") as f:
        atom_id = 1
        for i, (xyz, ok) in enumerate(zip(ca, cam)):
            if not ok:
                continue
            f.write(
                f"ATOM  {atom_id:5d}  CA  ALA A{(i+1):4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
            )
            atom_id += 1
        f.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=pathlib.Path, required=True)
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--sample_size", type=int, default=20000,
                    help="Random subset of .pt files to scan; set 0 to scan all (slow).")
    ap.add_argument("--top_k_examples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=pathlib.Path,
                    default=pathlib.Path("analysis_out/broken_chains"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(args.processed_dir.glob("*.pt"))
    print(f"[broken_chains] {len(pt_files)} .pt files in {args.processed_dir}")
    if args.sample_size and args.sample_size < len(pt_files):
        rng = random.Random(args.seed)
        pt_files = rng.sample(pt_files, args.sample_size)
        print(f"[broken_chains] subsampled to {len(pt_files)}")

    rows = []
    all_breaks: List = []
    for p in tqdm(pt_files, desc="scanning"):
        try:
            r, breaks = _audit_one(p, args.cutoff)
        except Exception as e:
            print(f"[broken_chains] {p.stem}: load failed {e!r}", file=sys.stderr)
            continue
        if r is not None:
            rows.append(r)
            for b in breaks:
                all_breaks.append({
                    "id": p.stem,
                    "spatial_gap": b[0],
                    "idx_gap": b[1],
                    "n_missing": b[2],
                    "gap_per_missing": b[3],
                    "kind": b[4],
                })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_chain_breaks.csv", index=False)

    breaks_df = pd.DataFrame(all_breaks)
    if not breaks_df.empty:
        breaks_df.to_csv(args.out_dir / "per_break_records.csv", index=False)
        suspicious = breaks_df[breaks_df["kind"] == "suspicious"]
        suspicious.sort_values("spatial_gap", ascending=False).to_csv(
            args.out_dir / "suspicious_breaks.csv", index=False
        )

    n = len(df)
    if n == 0:
        print("[broken_chains] no chains scanned; abort"); return

    fracs = {
        ">=1 break": float((df["n_breaks"] >= 1).mean()),
        ">=2 breaks": float((df["n_breaks"] >= 2).mean()),
        ">=5 breaks": float((df["n_breaks"] >= 5).mean()),
    }
    largest_gap_quantiles = {
        f"p{q}": float(np.percentile(df["max_gap"], q)) for q in [50, 90, 95, 99, 99.9]
    }

    intact = df[df["n_breaks"] == 0]
    broken = df[df["n_breaks"] >= 1]
    len_summary = {
        "intact_count": int(len(intact)),
        "intact_mean_len": float(intact["n_residues"].mean()) if len(intact) else float("nan"),
        "broken_count": int(len(broken)),
        "broken_mean_len": float(broken["n_residues"].mean()) if len(broken) else float("nan"),
    }

    # Write top-K worst as PDB.
    worst = df.sort_values(["n_breaks", "max_gap"], ascending=[False, False]).head(args.top_k_examples)
    written = []
    for _, row in worst.iterrows():
        pt_path = args.processed_dir / f"{row['id']}.pt"
        out_pdb = examples_dir / f"{row['id']}.pdb"
        try:
            _write_example_pdb(pt_path, out_pdb)
            written.append(out_pdb.name)
        except Exception as e:
            print(f"[broken_chains] failed example for {row['id']}: {e!r}", file=sys.stderr)

    # Plots.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["max_gap"].clip(upper=50), bins=80, color="firebrick", edgecolor="black", alpha=0.85)
    ax.axvline(args.cutoff, color="black", linestyle="--", label=f"cutoff={args.cutoff}")
    ax.set_xlabel("Largest CA-CA gap per chain (Å, clipped at 50)")
    ax.set_ylabel("Number of chains")
    ax.set_title("Largest CA-CA gap per chain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "hist_largest_gap.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = list(range(0, max(int(df["n_breaks"].max()) + 2, 6)))
    ax.hist(df["n_breaks"], bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel(f"Number of breaks per chain (cutoff={args.cutoff} Å)")
    ax.set_ylabel("Number of chains")
    ax.set_title("Number of breaks per chain")
    fig.tight_layout()
    fig.savefig(args.out_dir / "hist_n_breaks.png", dpi=130)
    plt.close(fig)

    if not breaks_df.empty:
        # Scatter: spatial gap vs index gap. Most "broken" chains should land on
        # a roughly linear ridge (≈3.5–4 Å per missing residue). Points off that
        # ridge are the genuinely suspicious ones.
        fig, ax = plt.subplots(figsize=(7, 5))
        kind_colors = {"expected": "#3878c5", "extreme": "#c47b3b", "suspicious": "#d4322c"}
        for kind, color in kind_colors.items():
            sub = breaks_df[breaks_df["kind"] == kind]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["idx_gap"].clip(upper=20), sub["spatial_gap"].clip(upper=50),
                s=8, alpha=0.45, color=color,
                label=f"{kind} (n={len(sub)})",
            )
        # Reference line: 3.8 Å per residue (typical CA-CA bond)
        x = np.linspace(1, 20, 50)
        ax.plot(x, 3.8 * x, color="gray", linestyle="--", lw=1, label="3.8 Å × idx_gap (expected)")
        ax.set_xlabel("residue_pdb_idx gap (clipped at 20)")
        ax.set_ylabel("spatial CA-CA gap (Å, clipped at 50)")
        ax.set_title("Spatial gap vs PDB-index gap per break")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / "scatter_idx_gap_vs_spatial_gap.png", dpi=130)
        plt.close(fig)

    # Break classification summary.
    classification = {}
    if not breaks_df.empty:
        classification = {
            "expected (idx_gap>=2 & gap_per_missing<=8 Å)": int((breaks_df["kind"] == "expected").sum()),
            "extreme (idx_gap>=2 & gap_per_missing>8 Å)": int((breaks_df["kind"] == "extreme").sum()),
            "suspicious (idx_gap==1 & spatial>cutoff — corruption signal)": int((breaks_df["kind"] == "suspicious").sum()),
        }

    # Markdown report.
    lines = ["# Broken-chain audit\n"]
    lines.append(f"- processed_dir: `{args.processed_dir}`")
    lines.append(f"- cutoff: {args.cutoff} Å")
    lines.append(f"- chains scanned: {n} (sampled from {args.sample_size or 'all'})\n")
    lines.append("## Headline\n")
    for k, v in fracs.items():
        lines.append(f"- chains with {k}: **{v:.1%}**  ({int(v * n)} / {n})")
    lines.append("")
    lines.append("## Largest-gap percentiles (Å)\n")
    for k, v in largest_gap_quantiles.items():
        lines.append(f"- {k}: {v:.2f}")
    lines.append("")
    lines.append("## Length comparison\n")
    for k, v in len_summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    if classification:
        lines.append("## Break classification by residue_pdb_idx gap\n")
        total_breaks = sum(classification.values())
        for k, v in classification.items():
            pct = (v / total_breaks * 100.0) if total_breaks else 0.0
            lines.append(f"- {k}: {v} ({pct:.1f}%)")
        lines.append("")
        lines.append(
            "**Interpretation**: 'expected' breaks correspond to unmodeled loops "
            "(missing residues in the original PDB) — the spatial gap matches the "
            "number of missing residues at ~3.5-4 Å each. The model already sees "
            "this via `residue_pdb_idx` (consumed by SequenceSeparationPairFeat) and "
            "`chain_breaks_per_residue` (consumed by ChainBreakPerResidueSeqFeat), so "
            "these are NOT data corruption. 'suspicious' breaks (idx_gap==1) are "
            "the only category that represents actually-corrupted data: two PDB-"
            "consecutive residues whose CAs are >cutoff apart. Inspect "
            "suspicious_breaks.csv for these."
        )
        lines.append("")
    lines.append("## Worst examples (written as PDB to ./examples/)\n")
    for fname in written:
        lines.append(f"- {fname}")
    lines.append("")
    (args.out_dir / "audit_broken_chains_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[broken_chains] all outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
