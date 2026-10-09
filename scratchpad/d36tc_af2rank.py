"""D36tc AF2Rank on the c2c outputs -- the published 'ours' scoring path, whole chain (user D4).

Per (label, stem) the 8 c2c `*_gen.pdb` (all-atom atom14, chain A) are scored with ONE model per process
(model_1_ptm -> af2rank_analysis/, model_2_ptm -> af2rank_analysis_model_2_ptm/, the production dir names):
OpenFoldAF2Rank(native cif, chain, recycles 6, mask_sidechains, cuEq attention + mult-update ON, DeepSpeed OFF),
segments None, `score_structure(pdb, "A", output_pdb=...)` (the production call, run_af2rank_prediction.py:322), then
step 5 = proteina_analysis.enrich_af2rank_output_dir (file-based, sequence-paired USalign -TMscore 5; rewrites
tm_ref_template / tm_ref_pred / tm_template_pred and composite = ptm * plddt * tm_template_pred).
Reference = $S/d36u/natives/<ENTRY>.cif (the D36 scoring natives). Resumable per (label, stem, model).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SUB), required=True)
    ap.add_argument("--labels", required=True, help="comma list of c2c labels")
    ap.add_argument("--c2c-pdb", default="/orcd/pool/006/chenxiou/d36tc/c2c_pdb")
    ap.add_argument("--out", default="/orcd/pool/006/chenxiou/d36tc/af2rank")
    ap.add_argument("--natives", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    usalign = shutil.which("USalign")
    assert usalign, "USalign not on PATH"

    jobs = []
    for label in args.labels.split(","):
        dirs = sorted(glob.glob(os.path.join(args.c2c_pdb, label, "*", "*")))
        assert dirs, f"no c2c dirs for {label}"
        for d in dirs:
            stem = os.path.basename(d)
            pdbs = sorted(glob.glob(os.path.join(d, "*_gen.pdb")))
            assert len(pdbs) == args.n, f"{label}/{stem}: {len(pdbs)} gen PDBs, want {args.n}"
            entry, chain = stem.rsplit("_", 1)
            native = os.path.join(args.natives, f"{entry}.cif")
            assert os.path.exists(native), native
            jobs.append((label, stem, chain, native, pdbs))
    print(f"[af2rank] {args.model}: {len(jobs)} (label, stem) pools, {sum(len(j[4]) for j in jobs)} decoys", flush=True)

    scorer = None
    for label, stem, chain, native, pdbs in jobs:
        out_dir = os.path.join(args.out, label, stem, SUB[args.model])
        csv_path = os.path.join(out_dir, f"af2rank_scores_{stem}.csv")
        names = [os.path.basename(p) for p in pdbs]
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            preds = [os.path.join(out_dir, "predicted_structures", n) for n in names]
            if sorted(df["structure_file"]) == names and all(os.path.exists(p) for p in preds) \
                    and "gdt_template_pred" in df.columns:
                print(f"[af2rank] skip {label}/{stem} (complete)", flush=True)
                continue
        if scorer is None:
            scorer = OpenFoldAF2Rank(native, chain=chain, model_name=args.model, recycles=6,
                                     use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=True,
                                     use_cuequivariance_multiplicative_update=True, mask_sidechains=True,
                                     usalign_path=usalign)
        else:
            scorer.reset_reference(native, chain)
        scorer.set_segments(None)
        os.makedirs(os.path.join(out_dir, "predicted_structures"), exist_ok=True)
        rows = []
        for p in pdbs:
            s = scorer.score_structure(p, decoy_chain="A", recycles=6,
                                       output_pdb=os.path.join(out_dir, "predicted_structures", os.path.basename(p)))
            assert "error" not in s, f"{label}/{stem} {os.path.basename(p)}: {s}"
            s.update({"protein_id": stem, "structure_file": os.path.basename(p), "structure_path": p})
            rows.append(s)
        save_af2rank_scores(rows, out_dir, stem)
        enrich_af2rank_output_dir(stem, os.path.dirname(pdbs[0]), out_dir, native, chain, usalign_path=usalign)
        df = pd.read_csv(csv_path)
        assert len(df) == args.n and df[["ptm", "plddt", "tm_template_pred", "tm_ref_pred"]].notna().all().all(), \
            f"{label}/{stem}: incomplete step-5 csv"
        print(f"[af2rank] {label}/{stem} {args.model}: {len(df)} scored; best composite "
              f"{df['composite'].max():.4f}", flush=True)
    print("D36TC_AF2RANK_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
