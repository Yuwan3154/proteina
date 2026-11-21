"""D36 W4 control planning and gate.
  plan  --what ctrl_pids   longest and shortest target in the manifest (D36ab native-template control + shakedown)
  plan  --what ctrl_stems  longest and shortest target with a published d36tc_A_tmpl pool for BOTH models (control i)
  plan  --what chunk --chunk k   production targets of chunk k (targets sorted by L, alternating ranks; 2 chunks)
  check  gate: control (i) |dpTM| <= 0.0005 (Control C's stated precision) and no 0.5/0.7 bin flip of pTM,
         tm_ref_pred, tm_template_pred; control (ii) no 0.5/0.7 bin flip of TM-to-native or pTM vs the D36ab factorial.
         Exit 1 on any failure (production depends afterok on this).
"""
import argparse
import csv
import glob
import os
import sys

import pandas as pd

PUB = "/orcd/pool/006/chenxiou/d36tc/af2rank"
SUB = {"model_1_ptm": "af2rank_analysis", "model_2_ptm": "af2rank_analysis_model_2_ptm"}
FACT = {"none": ("scores_gt_fulltmpl.csv", "d36r_gttmpl_ptm.csv"),
        "seqx": ("scores_gt_fulltmpl_masked.csv", "d36r_gttmpl_masked_ptm.csv"),
        "sc": ("scores_gt_fulltmpl_scmasked.csv", "d36ad_gttmpl_scmasked_ptm.csv"),
        "both": ("scores_gt_fulltmpl_bothmasked.csv", "d36ad_gttmpl_bothmasked_ptm.csv")}
TMCOL = {"model_1_ptm": "top1_tm", "model_2_ptm": "top1_m2_tm"}
PTMCOL = {"model_1_ptm": "ptm_m1", "model_2_ptm": "ptm_m2"}
BARS = (0.5, 0.7)


def lengths(manifest):
    return sorted({(int(r["L"]), r["pid"]) for r in csv.DictReader(open(manifest))})


def flip(a, b):
    return any((a >= t) != (b >= t) for t in BARS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "check"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--what", choices=["ctrl_pids", "ctrl_stems", "chunk"])
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--root")
    ap.add_argument("--factorial-dir")
    args = ap.parse_args()
    L = lengths(args.manifest)
    if args.cmd == "plan":
        if args.what == "ctrl_pids":
            print(f"{L[-1][1]},{L[0][1]}")
        elif args.what == "ctrl_stems":
            ok = [(n, p) for n, p in L if all(os.path.exists(f"{PUB}/d36tc_A_tmpl/{p}/{SUB[m]}/af2rank_scores_{p}.csv")
                                             for m in SUB)]
            print(f"{ok[-1][1]},{ok[0][1]}")
        else:
            assert args.chunk in (0, 1)
            print(",".join(p for i, (_, p) in enumerate(L) if i % 2 == args.chunk))
        return 0

    fails = []
    for f in sorted(glob.glob(os.path.join(args.root, "control", "prod", "control_prod_*.csv"))):
        df = pd.read_csv(f)
        dptm = (df["ptm_new"] - df["ptm_pub"]).abs().max()
        print(f"[gate i] {os.path.basename(f)}: n {len(df)}; max |d| " + ", ".join(
            f"{c} {(df[c + '_new'] - df[c + '_pub']).abs().max():.6f}"
            for c in ("ptm", "plddt", "tm_template_pred", "tm_ref_pred", "composite")))
        if dptm > 0.0005:
            fails.append(f"{f}: max |dpTM| {dptm:.6f} > 0.0005")
        for c in ("ptm", "tm_ref_pred", "tm_template_pred"):
            nf = sum(flip(a, b) for a, b in zip(df[c + "_new"], df[c + "_pub"]))
            print(f"[gate i]   {c}: {nf} bin flips (0.5/0.7)")
            if nf:
                fails.append(f"{f}: {c} {nf} bin flips")
    n_i = len(glob.glob(os.path.join(args.root, "control", "prod", "control_prod_*.csv")))
    if n_i != 2:
        fails.append(f"control (i): {n_i} comparison files, want 2")

    n_ii = 0
    for cell, (tmf, ptmf) in FACT.items():
        tm = {r["protein_id"]: r for r in csv.DictReader(open(os.path.join(args.factorial_dir, tmf)))}
        pt = {r["protein_id"]: r for r in csv.DictReader(open(os.path.join(args.factorial_dir, ptmf)))}
        for f in sorted(glob.glob(os.path.join(args.root, "control", "native", cell, "*", "*", "af2rank_scores_*.csv"))):
            model = f.split(os.sep)[-3]
            for _, r in pd.read_csv(f).iterrows():
                p = r["protein_id"]
                ref_tm, ref_ptm = float(tm[p][TMCOL[model]]), float(pt[p][PTMCOL[model]])
                ftm, fptm = flip(r["tm_ref_pred"], ref_tm), flip(r["ptm"], ref_ptm)
                n_ii += 1
                print(f"[gate ii] {p} {cell:5s} {model}: TM {r['tm_ref_pred']:.4f} vs D36ab {ref_tm:.4f}"
                      f"{' FLIP' if ftm else ''} | pTM {r['ptm']:.4f} vs {ref_ptm:.4f}{' FLIP' if fptm else ''}")
                if ftm or fptm:
                    fails.append(f"control (ii) {p} {cell} {model}: bin flip")
    if n_ii != 16:
        fails.append(f"control (ii): {n_ii} comparisons, want 16 (2 targets x 4 cells x 2 models)")
    for x in fails:
        print(f"[gate] FAIL {x}")
    print("D36W4_GATE_" + ("PASS" if not fails else "FAIL"), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
