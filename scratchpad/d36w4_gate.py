"""D36 W4 control planning and gate.
  plan  --what ctrl_pids   longest and shortest target in the manifest (D36ab native-template control + shakedown)
  plan  --what ctrl_stems  longest and shortest target with a published d36tc_A_tmpl pool for BOTH models (control i)
  plan  --what chunk --chunk k   production targets of chunk k (targets sorted by L, alternating ranks; 2 chunks)
  check  gate: control (i) |dpTM| <= 0.0005 (Control C's stated precision) and no 0.5/0.7 bin flip of pTM,
         tm_ref_pred, tm_template_pred; control (ii) no 0.5/0.7 bin flip of TM-to-native or pTM vs the D36ab factorial.
         Exit 1 on any failure (production depends afterok on this). Control (i) rows must carry kernels == cueq,
         control (ii) rows kernels == --kernels (the run-wide setting).
  kernel_report  control (iii): the control (i) pools re-scored with --kernels vs the published (cuEq) values; writes
         <ctrl-dir>/control_iii.csv (per structure pTM / tm_ref_pred deltas, 0.5/0.7 crossings). No pass/fail; exit 1
         only if its inputs are missing.
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
    ap.add_argument("cmd", choices=["plan", "check", "kernel_report"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--what", choices=["ctrl_pids", "ctrl_stems", "chunk"])
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--ctrl-dir", help="control root of this run: prod/ (i), native/ (ii), prod_<kernels>/ (iii)")
    ap.add_argument("--kernels", choices=["cueq", "plain"], help="run-wide kernel setting of control (ii) / (iii)")
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
    assert args.ctrl_dir and args.kernels, "--ctrl-dir and --kernels are required"

    if args.cmd == "kernel_report":
        files = sorted(glob.glob(os.path.join(args.ctrl_dir, f"prod_{args.kernels}", "control_prod_*.csv")))
        if len(files) != 2:
            print(f"[iii] {len(files)} comparison files, want 2")
            return 1
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        assert (df["kernels"] == args.kernels).all(), f"control (iii) rows not all kernels == {args.kernels}"
        out = df[["label", "stem", "model", "structure_file", "kernels"]].copy()
        for c in ("ptm", "tm_ref_pred"):
            out[f"{c}_cueq_pub"], out[f"{c}_{args.kernels}"] = df[c + "_pub"], df[c + "_new"]
            out[f"d_{c}"] = df[c + "_new"] - df[c + "_pub"]
            for t in BARS:
                out[f"{c}_cross_{t}"] = (df[c + "_new"] >= t) != (df[c + "_pub"] >= t)
        out_csv = os.path.join(args.ctrl_dir, "control_iii.csv")
        out.to_csv(out_csv, index=False)
        for c in ("ptm", "tm_ref_pred"):
            for m, g in out.groupby("model"):
                d = g[f"d_{c}"]
                print(f"[iii] {m} {c}: n {len(g)}; {args.kernels} - cuEq mean {d.mean():+.4f} median {d.median():+.4f} "
                      f"min {d.min():+.4f} max {d.max():+.4f}; up {(d > 0).sum()} down {(d < 0).sum()}; crossings "
                      + ", ".join(f"{t}: {int(g[f'{c}_cross_{t}'].sum())}" for t in BARS))
        print(f"[iii] wrote {out_csv} ({len(out)} rows)")
        print("D36W4_KERNEL_REPORT_DONE", flush=True)
        return 0

    fails = []
    for f in sorted(glob.glob(os.path.join(args.ctrl_dir, "prod", "control_prod_*.csv"))):
        df = pd.read_csv(f)
        if not (df["kernels"] == "cueq").all():
            fails.append(f"{f}: control (i) must use cueq kernels")
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
    n_i = len(glob.glob(os.path.join(args.ctrl_dir, "prod", "control_prod_*.csv")))
    if n_i != 2:
        fails.append(f"control (i): {n_i} comparison files, want 2")

    n_ii = 0
    for cell, (tmf, ptmf) in FACT.items():
        tm = {r["protein_id"]: r for r in csv.DictReader(open(os.path.join(args.factorial_dir, tmf)))}
        pt = {r["protein_id"]: r for r in csv.DictReader(open(os.path.join(args.factorial_dir, ptmf)))}
        for f in sorted(glob.glob(os.path.join(args.ctrl_dir, "native", cell, "*", "*", "af2rank_scores_*.csv"))):
            model = f.split(os.sep)[-3]
            for _, r in pd.read_csv(f).iterrows():
                p = r["protein_id"]
                if r["kernels"] != args.kernels:
                    fails.append(f"control (ii) {p} {cell} {model}: kernels {r['kernels']} != {args.kernels}")
                ref_tm, ref_ptm = float(tm[p][TMCOL[model]]), float(pt[p][PTMCOL[model]])
                ftm, fptm = flip(r["tm_ref_pred"], ref_tm), flip(r["ptm"], ref_ptm)
                n_ii += 1
                print(f"[gate ii] {p} {cell:5s} {model} {r['kernels']}: TM {r['tm_ref_pred']:.4f} vs D36ab {ref_tm:.4f}"
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
