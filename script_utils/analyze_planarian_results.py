#!/usr/bin/env python3
"""Merge the planarian ligand-search runs and report the pTM-cutoff hit rates.

Reads every per-(CATH code x batch) prediction_summary.csv, joins the
no-best-hit-subset flag from the run manifest, and reports the fraction of
sequences whose best AF2Rank pTM clears the cutoff -- full set vs the
no-best-hit subset, per CATH code.

Usage:
    python script_utils/analyze_planarian_results.py \
        --run_dir ~/planarian_lig --out ~/planarian_lig/planarian_results.csv
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="dir holding batch_manifest.csv and out/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ptm_cutoff", type=float, default=0.7)
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    manifest = pd.read_csv(os.path.join(run_dir, "planarian_extracellular.csv"))
    batches = pd.read_csv(os.path.join(run_dir, "batch_manifest.csv"))

    frames, missing = [], []
    for _, row in batches.iterrows():
        path = os.path.join(run_dir, "out", f"{row.tag}_b{row.batch}", "prediction_summary.csv")
        if not os.path.exists(path):
            missing.append(f"{row.tag}_b{row.batch}")
            continue
        df = pd.read_csv(path)
        df["cath_code"] = row.cath_code
        df["batch"] = row.batch
        frames.append(df)

    if missing:
        print(f"NOT YET COLLECTED ({len(missing)}): {', '.join(missing)}")
    if not frames:
        raise SystemExit("no prediction_summary.csv found")

    res = pd.concat(frames, ignore_index=True)
    res = res.merge(
        manifest[["id", "length", "in_no_besthit_subset"]],
        left_on="protein_id", right_on="id", how="left",
    ).drop(columns=["id"])
    res.to_csv(args.out, index=False)
    print(f"wrote {args.out}  rows={len(res)}")

    print(f"\n=== best_ptm >= {args.ptm_cutoff} ===")
    rows = []
    for code, g in res.groupby("cath_code"):
        scored = g[g.best_ptm.notna()]
        sub = scored[scored.in_no_besthit_subset]
        rows.append({
            "cath_code": code,
            "n_scored": len(scored),
            "n_pass": int((scored.best_ptm >= args.ptm_cutoff).sum()),
            "frac_pass": round(float((scored.best_ptm >= args.ptm_cutoff).mean()), 4) if len(scored) else float("nan"),
            "n_subset": len(sub),
            "n_pass_subset": int((sub.best_ptm >= args.ptm_cutoff).sum()),
            "frac_pass_subset": round(float((sub.best_ptm >= args.ptm_cutoff).mean()), 4) if len(sub) else float("nan"),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n=== top 20 candidates per CATH code ===")
    cols = ["protein_id", "length", "in_no_besthit_subset", "best_ptm", "best_plddt", "best_energy"]
    for code, g in res.groupby("cath_code"):
        print(f"\n-- {code} --")
        print(g.nlargest(20, "best_ptm")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
