"""D36 W4 control (i): the DEFAULT scorer path of this branch (no template_seq_mask argument, mask_sidechains=True =
the production 'both' recipe) re-scores already-published d36tc af2rank pools with the d36tc_af2rank.py call, then
step 5, and writes per-decoy |delta| vs the published d36tc CSV (pTM, pLDDT, tm_template_pred, tm_ref_pred, composite).
"""
import argparse
import glob
import os
import shutil
import sys

import pandas as pd

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank, save_af2rank_scores
from proteinfoundation.prediction_pipeline.proteina_analysis import enrich_af2rank_output_dir

SUB = {"model_1_ptm": "af2rank_analysis", "model_2_ptm": "af2rank_analysis_model_2_ptm"}
COLS = ["ptm", "plddt", "tm_template_pred", "tm_ref_pred", "composite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--model", choices=sorted(SUB), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--c2c-pdb", default="/orcd/pool/006/chenxiou/d36tc/c2c_pdb")
    ap.add_argument("--published", default="/orcd/pool/006/chenxiou/d36tc/af2rank")
    ap.add_argument("--natives", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    args = ap.parse_args()
    usalign = shutil.which("USalign")
    assert usalign, "USalign not on PATH"
    scorer, cmp_rows = None, []
    for stem in args.stems.split(","):
        entry, chain = stem.rsplit("_", 1)
        native = os.path.join(args.natives, f"{entry}.cif")
        pdbs = sorted(glob.glob(os.path.join(args.c2c_pdb, args.label, "*", stem, "*_gen.pdb")))
        assert len(pdbs) == 8, (stem, len(pdbs))
        if scorer is None:
            scorer = OpenFoldAF2Rank(native, chain=chain, model_name=args.model, recycles=6,
                                     use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=True,
                                     use_cuequivariance_multiplicative_update=True, mask_sidechains=True,
                                     usalign_path=usalign)
        else:
            scorer.reset_reference(native, chain)
        scorer.set_segments(None)
        out_dir = os.path.join(args.out, args.label, stem, SUB[args.model])
        os.makedirs(os.path.join(out_dir, "predicted_structures"), exist_ok=True)
        rows = []
        for p in pdbs:
            s = scorer.score_structure(p, decoy_chain="A", recycles=6,
                                       output_pdb=os.path.join(out_dir, "predicted_structures", os.path.basename(p)))
            assert "error" not in s, f"{stem} {os.path.basename(p)}: {s}"
            s.update({"protein_id": stem, "structure_file": os.path.basename(p), "structure_path": p})
            rows.append(s)
        save_af2rank_scores(rows, out_dir, stem)
        enrich_af2rank_output_dir(stem, os.path.dirname(pdbs[0]), out_dir, native, chain, usalign_path=usalign)
        new = pd.read_csv(os.path.join(out_dir, f"af2rank_scores_{stem}.csv")).set_index("structure_file")
        pub = pd.read_csv(os.path.join(args.published, args.label, stem, SUB[args.model],
                                       f"af2rank_scores_{stem}.csv")).set_index("structure_file")
        assert sorted(new.index) == sorted(pub.index), stem
        for f in new.index:
            row = {"label": args.label, "stem": stem, "model": args.model, "structure_file": f}
            for c in COLS:
                row[f"{c}_new"], row[f"{c}_pub"] = float(new.at[f, c]), float(pub.at[f, c])
            cmp_rows.append(row)
    out_csv = os.path.join(args.out, f"control_prod_{args.label}_{args.model}.csv")
    pd.DataFrame(cmp_rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(cmp_rows)
    for c in COLS:
        print(f"[controlP] {args.model} {c}: max |new - published| {(df[c + '_new'] - df[c + '_pub']).abs().max():.6f}")
    print("D36W4_CONTROL_PROD_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
