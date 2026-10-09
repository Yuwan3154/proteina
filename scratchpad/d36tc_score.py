"""D36tc final scoring (CPU). Completeness from FILES first, then the per-target table.

Arms (labels <prefix>_<group>_<arm>): groups A (30 targets <=384 with a hit), B (6XRY_A, 7F7N_A), C (8AUC_B, 8X4J_A);
arms self + tmpl (AF2Rank-selected) and native (native-map c2c ceiling, c2c TMs only, D10 definition).
  c2c TM   : score_arm.py's call -- USalign <native cif> -chain1 <ch> <c2c pdb> -TMscore 5 -outfmt 2, TM1 (native-
             normalised) -- on all 8 c2c structures -> median (sorted(v)[len(v)//2]) and best.
  selection: model_1_ptm af2rank csv (step-5 rewritten); composite recomputed = ptm * plddt * tm_template_pred and
             required to equal the stored column; pick = argmax (a tie at the top is reported, never broken silently).
  headline : tm_ref_pred of the pick (AF2's refold vs the native); beside it the pick's own c2c TM (step-5
             tm_ref_template, cross-checked against the independent USalign call).
Counts >= 0.7 / >= 0.9 / < 0.5 are printed before medians.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

GROUPS = {"A": ("self", "tmpl", "native"), "B": ("self", "native"), "C": ("self", "tmpl", "native")}


def tm_vs_native(usalign, native, chain, pred):
    r = subprocess.run([usalign, native, "-chain1", chain, pred, "-TMscore", "5", "-outfmt", "2"],
                       capture_output=True, text=True, check=True)
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) >= 7 and not ln.startswith("#"):
            return float(p[2])
    raise RuntimeError(f"no USalign row for {pred}")


def med(v):
    return sorted(v)[len(v) // 2] if v else float("nan")


def counts(v):
    return f"n={len(v)} >=0.7:{sum(x >= 0.7 for x in v)} >=0.9:{sum(x >= 0.9 for x in v)} <0.5:{sum(x < 0.5 for x in v)} median={med(v):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="d36tc")
    ap.add_argument("--groups-dir", default="/orcd/pool/006/chenxiou/d36tc/inputs")
    ap.add_argument("--c2c-pdb", default="/orcd/pool/006/chenxiou/d36tc/c2c_pdb")
    ap.add_argument("--af2rank", default="/orcd/pool/006/chenxiou/d36tc/af2rank")
    ap.add_argument("--natives", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    expected, found, problems, out = 0, 0, [], []
    for g, arms in GROUPS.items():
        stems = [l.strip() for l in open(os.path.join(args.groups_dir, f"group{g}.txt")) if l.strip()]
        for arm in arms:
            label = f"{args.prefix}_{g}_{arm}"
            for st in stems:
                entry, ch = st.rsplit("_", 1)
                native = os.path.join(args.natives, f"{entry}.cif")
                pdbs = sorted(glob.glob(os.path.join(args.c2c_pdb, label, "*", st, "*_gen.pdb")))
                expected += 8
                found += len(pdbs)
                rec = {"pid": st, "group": g, "arm": arm, "n_c2c": len(pdbs)}
                if len(pdbs) != 8:
                    problems.append(f"{label}/{st}: {len(pdbs)} of 8 c2c PDBs")
                    out.append(rec)
                    continue
                tms = {os.path.basename(p): tm_vs_native(args.usalign, native, ch, p) for p in pdbs}
                rec.update(c2c_median8=med(list(tms.values())), c2c_best8=max(tms.values()))
                if arm != "native":
                    for model, sub in (("m1", "af2rank_analysis"), ("m2", "af2rank_analysis_model_2_ptm")):
                        n_pred = len(glob.glob(os.path.join(args.af2rank, label, st, sub, "predicted_structures", "*.pdb")))
                        expected += 8
                        found += n_pred
                        if n_pred != 8:
                            problems.append(f"{label}/{st} {model}: {n_pred} of 8 AF2 predictions")
                    csvp = os.path.join(args.af2rank, label, st, "af2rank_analysis", f"af2rank_scores_{st}.csv")
                    if not os.path.exists(csvp):
                        problems.append(f"{label}/{st}: no m1 csv")
                        out.append(rec)
                        continue
                    df = pd.read_csv(csvp)
                    assert len(df) == 8 and sorted(df["structure_file"]) == sorted(tms), f"{csvp}: rows != c2c files"
                    comp = df["ptm"] * df["plddt"] * df["tm_template_pred"]
                    if not (comp == df["composite"]).all():
                        problems.append(f"{label}/{st}: stored composite != ptm*plddt*tm_template_pred")
                    top = comp.max()
                    n_top = len(np.unique(df.loc[comp == top, "structure_file"]))
                    if n_top > 1:
                        problems.append(f"{label}/{st}: {n_top}-way tie at the top composite")
                    i = int(comp.idxmax())
                    sel = df.loc[i, "structure_file"]
                    d_tm = float(df.loc[i, "tm_ref_template"]) - tms[sel]
                    if d_tm != 0:
                        problems.append(f"{label}/{st}: step-5 tm_ref_template {df.loc[i, 'tm_ref_template']} vs USalign {tms[sel]}")
                    rec.update(sel_file=sel, sel_composite=float(top), sel_ptm=float(df.loc[i, "ptm"]),
                               headline_tm_ref_pred=float(df.loc[i, "tm_ref_pred"]), sel_c2c_tm=tms[sel])
                out.append(rec)
    pd.DataFrame(out).to_csv(args.out, index=False)
    print(f"[score] completeness from files: {found} of {expected} expected (c2c PDBs + m1/m2 AF2 predictions)")
    for p in problems:
        print(f"  PROBLEM {p}")
    df = pd.DataFrame(out)
    for arm in ("self", "tmpl", "native"):
        d = df[df["arm"] == arm]
        print(f"\n== arm {arm} ({len(d)} targets)")
        cols = ["headline_tm_ref_pred", "sel_c2c_tm", "c2c_median8", "c2c_best8"] if arm != "native" else ["c2c_median8", "c2c_best8"]
        for c in cols:
            if c in d:
                print(f"  {c:<22} {counts([float(x) for x in d[c].dropna()])}")
    print(f"\nwrote {args.out}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
