"""D36 W4 (Q6): production AF2Rank scorer on a fixed template set under one masking CELL and one MODEL per process.

Cells (template sequence mask x side-chain strip; ledger 6c):
  none   real template sequence, full side chains
  seqx   all-X sequence (production post-pipeline mask), full side chains
  sc     real sequence, side chains stripped to N/CA/C/O/CB (+Gly CB)
  both   all-X + stripped = the production recipe
  gapsc  AF2Rank paper's gap-token sequence (pre-pipeline) + stripped
Per target: OpenFoldAF2Rank(native cif, chain, recycles 6, DeepSpeed OFF) as d36tc_af2rank.py, whole chain; --kernels
cueq (default = d36tc production: cuEq attention + mult-update ON) or plain (both OFF = the D36ab native-template driver,
the configuration control (ii) validates); score_structure(..., output_pdb=...), then step 5 =
enrich_af2rank_output_dir (file-based, sequence-paired USalign -TMscore 5). Resumable per (cell, model, pid). Templates and fixed labels from the manifest.
"""
import argparse
import csv
import os
import shutil
import sys

import pandas as pd

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank, save_af2rank_scores
from proteinfoundation.prediction_pipeline.proteina_analysis import enrich_af2rank_output_dir

CELLS = {"none": (False, "none"), "seqx": (False, "x"), "sc": (True, "none"), "both": (True, "x"),
         "gapsc": (True, "gap")}
MODELS = ("model_1_ptm", "model_2_ptm")
KERNELS = {"cueq": (True, True), "plain": (False, False)}  # (use_cuequivariance_attention, ..._multiplicative_update)


def check_done(df, rows_in, p, kernels):
    """Post-conditions of a finished target; also re-run on resume so a failed target is never skipped."""
    assert "kernels" in df.columns and (df["kernels"] == kernels).all(), f"{p}: csv kernels differ from --kernels {kernels}"
    assert len(df) == len(rows_in) and \
        df[["ptm", "plddt", "tm_template_pred", "tm_ref_pred", "tm_ref_template"]].notna().all().all(), \
        f"{p}: incomplete step-5 csv"
    lab = {r["template_id"]: float(r["tm_native"]) for r in rows_in}
    dlab = max(abs(t - lab[i]) for i, t in zip(df["template_id"], df["tm_ref_template"]))
    assert dlab < 1e-3, f"{p}: step-5 tm_ref_template differs from the manifest label by {dlab}"
    return dlab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=sorted(CELLS), required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--pids", required=True, help="comma list of targets")
    ap.add_argument("--manifest", required=True, help="templates_manifest.csv (staged paths + fixed labels)")
    ap.add_argument("--out", required=True, help="output root; writes <out>/<cell>/<model>/<pid>/")
    ap.add_argument("--natives", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    ap.add_argument("--template-ids", default=None, help="optional comma list restricting the templates (controls)")
    ap.add_argument("--kernels", choices=sorted(KERNELS), default="cueq", help="cueq = d36tc production; plain = D36ab")
    args = ap.parse_args()
    usalign = shutil.which("USalign")
    assert usalign, "USalign not on PATH"
    mask_sc, seq_mask = CELLS[args.cell]
    cue_attn, cue_mul = KERNELS[args.kernels]
    keep_ids = set(args.template_ids.split(",")) if args.template_ids else None

    man = {}
    for r in csv.DictReader(open(args.manifest)):
        if keep_ids is None or r["template_id"] in keep_ids:
            man.setdefault(r["pid"], []).append(r)
    pids = args.pids.split(",")
    for p in pids:
        assert p in man, f"{p}: no templates in manifest"
        tdirs = {os.path.dirname(r["path"]) for r in man[p]}
        assert len(tdirs) == 1, f"{p}: templates in {len(tdirs)} dirs"
    n_tot = sum(len(man[p]) for p in pids)
    print(f"[w4] cell {args.cell} (mask_sidechains={mask_sc}, template_seq_mask={seq_mask}) {args.model}: "
          f"kernels {args.kernels} (cuEq attention={cue_attn}, mult-update={cue_mul}): {len(pids)} targets, {n_tot} templates",
          flush=True)

    scorer = None
    for p in pids:
        entry, chain = p.rsplit("_", 1)
        native = os.path.join(args.natives, f"{entry}.cif")
        assert os.path.exists(native), native
        rows_in = sorted(man[p], key=lambda r: r["template_id"])
        out_dir = os.path.join(args.out, args.cell, args.model, p)
        csv_path = os.path.join(out_dir, f"af2rank_scores_{p}.csv")
        names = sorted(os.path.basename(r["path"]) for r in rows_in)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            preds = [os.path.join(out_dir, "predicted_structures", n) for n in names]
            if sorted(df["structure_file"]) == names and all(os.path.exists(x) for x in preds) \
                    and "gdt_template_pred" in df.columns:
                check_done(df, rows_in, p, args.kernels)
                print(f"[w4] skip {p} (complete)", flush=True)
                continue
        if scorer is None:
            scorer = OpenFoldAF2Rank(native, chain=chain, model_name=args.model, recycles=6,
                                     use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=cue_attn,
                                     use_cuequivariance_multiplicative_update=cue_mul, mask_sidechains=mask_sc,
                                     usalign_path=usalign, template_seq_mask=seq_mask)
        else:
            scorer.reset_reference(native, chain)
        scorer.set_segments(None)
        os.makedirs(os.path.join(out_dir, "predicted_structures"), exist_ok=True)
        rows = []
        for r in rows_in:
            pdb = r["path"]
            assert len(scorer.reference_sequence) == int(r["L"]), (p, r["template_id"])
            s = scorer.score_structure(pdb, decoy_chain="A", recycles=6,
                                       output_pdb=os.path.join(out_dir, "predicted_structures", os.path.basename(pdb)))
            assert "error" not in s, f"{p} {r['template_id']}: {s}"
            s.update({"protein_id": p, "structure_file": os.path.basename(pdb), "structure_path": pdb,
                      "template_id": r["template_id"], "cell": args.cell, "model": args.model,
                      "kernels": args.kernels})
            rows.append(s)
        save_af2rank_scores(rows, out_dir, p)
        enrich_af2rank_output_dir(p, os.path.dirname(rows_in[0]["path"]), out_dir, native, chain, usalign_path=usalign)
        df = pd.read_csv(csv_path)
        dlab = check_done(df, rows_in, p, args.kernels)
        print(f"[w4] {p} {args.cell} {args.model}: {len(df)} scored; max|label diff| {dlab:.5f}", flush=True)
    print("D36W4_AF2RANK_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
