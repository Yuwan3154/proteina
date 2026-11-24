"""D36tc AF2Rank on the c2c outputs -- the published 'ours' scoring path, whole chain (user D4).

Per (label, stem) the 8 c2c `*_gen.pdb` (all-atom atom14, chain A) are scored with ONE model per process
(model_1_ptm -> af2rank_analysis/, model_2_ptm -> af2rank_analysis_model_2_ptm/, the production dir names):
OpenFoldAF2Rank(native cif, chain, recycles 6, mask_sidechains, cuEq attention + mult-update ON, DeepSpeed OFF),
segments None, `score_structure(pdb, "A", output_pdb=...)` (the production call, run_af2rank_prediction.py:322), then
step 5 = proteina_analysis.enrich_af2rank_output_dir (file-based, sequence-paired USalign -TMscore 5; rewrites
tm_ref_template / tm_ref_pred / tm_template_pred and composite = ptm * plddt * tm_template_pred).
Reference = $S/d36u/natives/<ENTRY>.cif (the D36 scoring natives). Resumable per (label, stem, model).
--kernels plain (W8 re-score) turns both cuEq kernels off; it needs its own --out (never the production dir) and
refuses to resume from a CSV written with another kernel set. --stems restricts the pools to those stems.
--score-chains JSON (D36tc-v3; default none = published, stem suffix passed as the chain): {stem: AUTHOR chain to
score against}. The scorer and step 5 take a LABEL chain (cif_chain_mapping maps label -> auth), so the label_asym_id
whose ATOM records carry that auth_asym_id is passed (v3: stems are built from LABEL chains; 7KU5_A -> auth O,
9JKP_A -> auth B, 9HKF_B (label B = the identical auth-A copy, user 2026-11-24) -> auth B = label A).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import pandas as pd

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank, save_af2rank_scores
from proteinfoundation.prediction_pipeline.proteina_analysis import enrich_af2rank_output_dir

SUB = {"model_1_ptm": "af2rank_analysis", "model_2_ptm": "af2rank_analysis_model_2_ptm"}
KERNELS = {"cueq": (True, True), "plain": (False, False)}  # (use_cuequivariance_attention, ..._multiplicative_update)
OUT_PROD = "/orcd/pool/006/chenxiou/d36tc/af2rank"


def label_chain_for_auth(cif, auth):
    """The unique label_asym_id of the ATOM records whose auth_asym_id == auth (fails if not exactly one)."""
    d = MMCIF2Dict(cif)
    labels = {lab for g, lab, au in zip(d["_atom_site.group_PDB"], d["_atom_site.label_asym_id"],
                                        d["_atom_site.auth_asym_id"]) if g == "ATOM" and au == auth}
    assert len(labels) == 1, f"{cif}: auth chain {auth} has ATOM label chains {sorted(labels)}"
    return labels.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SUB), required=True)
    ap.add_argument("--labels", required=True, help="comma list of c2c labels")
    ap.add_argument("--c2c-pdb", default="/orcd/pool/006/chenxiou/d36tc/c2c_pdb")
    ap.add_argument("--out", default=OUT_PROD)
    ap.add_argument("--natives", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--expect-pools", type=int, required=True, help="number of (label, stem) pools over --labels")
    ap.add_argument("--kernels", choices=sorted(KERNELS), default="cueq", help="cueq = d36tc production; plain = D36ab")
    ap.add_argument("--stems", default="", help="optional comma list: score only these stems")
    ap.add_argument("--score-chains", default="", help="JSON {stem: author chain}; every pooled stem must be a key")
    args = ap.parse_args()
    cue_attn, cue_mul = KERNELS[args.kernels]
    prod = os.path.realpath(OUT_PROD)
    assert args.kernels == "cueq" or os.path.commonpath([os.path.realpath(args.out), prod]) != prod, \
        "--kernels plain must not write into the production d36tc af2rank dir"
    keep = set(args.stems.split(",")) if args.stems else None
    score_chains = json.load(open(args.score_chains)) if args.score_chains else None
    usalign = shutil.which("USalign")
    assert usalign, "USalign not on PATH"

    jobs = []
    for label in args.labels.split(","):
        dirs = sorted(glob.glob(os.path.join(args.c2c_pdb, label, "*", "*")))
        assert dirs, f"no c2c dirs for {label}"
        for d in dirs:
            stem = os.path.basename(d)
            if keep is not None and stem not in keep:
                continue
            pdbs = sorted(glob.glob(os.path.join(d, "*_gen.pdb")))
            assert len(pdbs) == args.n, f"{label}/{stem}: {len(pdbs)} gen PDBs, want {args.n}"
            entry, chain = stem.rsplit("_", 1)
            native = os.path.join(args.natives, f"{entry}.cif")
            assert os.path.exists(native), native
            if score_chains is not None:
                auth = score_chains[stem]
                chain = label_chain_for_auth(native, auth)
                print(f"[af2rank] {stem}: score vs auth chain {auth} = label chain {chain} (passed to the scorer)", flush=True)
            jobs.append((label, stem, chain, native, pdbs))
    assert len(jobs) == args.expect_pools, f"{len(jobs)} pools, want {args.expect_pools}"
    print(f"[af2rank] {args.model}: {len(jobs)} (label, stem) pools, {sum(len(j[4]) for j in jobs)} decoys", flush=True)

    scorer = None
    for label, stem, chain, native, pdbs in jobs:
        out_dir = os.path.join(args.out, label, stem, SUB[args.model])
        csv_path = os.path.join(out_dir, f"af2rank_scores_{stem}.csv")
        names = [os.path.basename(p) for p in pdbs]
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            preds = [os.path.join(out_dir, "predicted_structures", n) for n in names]
            done_kernels = set(df["kernels"]) if "kernels" in df.columns else {"cueq"}
            assert done_kernels == {args.kernels}, f"{csv_path}: written with kernels {done_kernels}, not {args.kernels}"
            if sorted(df["structure_file"]) == names and all(os.path.exists(p) for p in preds) \
                    and "gdt_template_pred" in df.columns:
                print(f"[af2rank] skip {label}/{stem} (complete)", flush=True)
                continue
        if scorer is None:
            scorer = OpenFoldAF2Rank(native, chain=chain, model_name=args.model, recycles=6,
                                     use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=cue_attn,
                                     use_cuequivariance_multiplicative_update=cue_mul, mask_sidechains=True,
                                     usalign_path=usalign)
        else:
            scorer.reset_reference(native, chain)
        scorer.set_segments(None)
        if score_chains is not None:
            # the 1:1 template mapping needs decoy length == reference chain length (else a silent kalign fallback)
            for p in pdbs:
                n_res = len({ln[22:27] for ln in open(p) if ln.startswith("ATOM") and ln[12:16].strip() == "CA"})
                assert n_res == len(scorer.reference_sequence), \
                    f"{label}/{stem} {os.path.basename(p)}: {n_res} residues vs reference {len(scorer.reference_sequence)}"
        os.makedirs(os.path.join(out_dir, "predicted_structures"), exist_ok=True)
        rows = []
        for p in pdbs:
            s = scorer.score_structure(p, decoy_chain="A", recycles=6,
                                       output_pdb=os.path.join(out_dir, "predicted_structures", os.path.basename(p)))
            assert "error" not in s, f"{label}/{stem} {os.path.basename(p)}: {s}"
            s.update({"protein_id": stem, "structure_file": os.path.basename(p), "structure_path": p,
                      "kernels": args.kernels})
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
