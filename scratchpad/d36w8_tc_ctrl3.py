"""D36 W8 (A) control: the plain-kernel d36tc re-score must reproduce control (iii) (W4b job 29950167, plain kernels,
same scorer call, d36tc_A_tmpl 8TVL_A + 6UCK_A x 2 models). Compares the step-5 CSVs decoy by decoy on
ptm, plddt, tm_template_pred, tm_ref_pred, composite; exit 1 on FAIL (control (i) of the same code path
reproduced the published cuEq values to 0.000000 at its 6 printed decimals). PASS_EXACT = bitwise equal;
PASS_6DP = max |diff| < 5e-7, i.e. equal at the 6 decimals control (i) was established at; anything else FAIL.
NaN must sit at the same positions on both sides.
"""
import argparse
import os
import sys

import pandas as pd

SUB = {"model_1_ptm": "af2rank_analysis", "model_2_ptm": "af2rank_analysis_model_2_ptm"}
COLS = ["ptm", "plddt", "tm_template_pred", "tm_ref_pred", "composite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="W8 af2rank out root (<label>/<stem>/<sub>/)")
    ap.add_argument("--ref", required=True, help="control (iii) out root (<label>/<stem>/<sub>/)")
    ap.add_argument("--label", default="d36tc_A_tmpl")
    ap.add_argument("--stems", default="8TVL_A,6UCK_A")
    args = ap.parse_args()
    n, worst = 0, {c: 0.0 for c in COLS}
    for stem in args.stems.split(","):
        for model, sub in SUB.items():
            rel = os.path.join(args.label, stem, sub, f"af2rank_scores_{stem}.csv")
            new = pd.read_csv(os.path.join(args.new, rel)).set_index("structure_file")
            ref = pd.read_csv(os.path.join(args.ref, rel)).set_index("structure_file")
            assert sorted(new.index) == sorted(ref.index) and len(new) == 8, f"{rel}: decoy sets differ"
            assert set(new["kernels"]) == {"plain"} and set(ref["kernels"]) == {"plain"}, f"{rel}: kernels column"
            for c in COLS:
                a, b = new[c], ref.loc[new.index, c]
                assert (a.isna() == b.isna()).all(), f"{rel} {c}: NaN positions differ"
                worst[c] = max(worst[c], float((a - b).abs().max(skipna=True)))
            n += len(new)
    for c in COLS:
        print(f"[ctrl3] {c}: max |W8 - control (iii)| = {worst[c]!r} over {n} decoys")
    exact = n == 32 and all(v == 0.0 for v in worst.values())
    dp6 = n == 32 and all(v < 5e-7 for v in worst.values())
    verdict = "PASS_EXACT" if exact else ("PASS_6DP" if dp6 else "FAIL")
    print(f"D36W8_CTRL3_{verdict} ({n} of 32 decoys)")
    return 0 if dp6 else 1


if __name__ == "__main__":
    sys.exit(main())
