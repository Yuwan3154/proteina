"""D36 W8 (B): re-score the published 'ours' AF2Rank step (ProteinEBM top-5 -> AF2Rank, variant af2rank_uniform_rc6)
with a chosen kernel set, through the production code path.

Production = sc_badafdb_rc6.sbatch / d36f_proteina16.sbatch -> run_af2rank_prediction.py: top-5 by ProteinEBM energy
(_select_topk_df), reference = _pick_reference(cif_dir), segments = _segments_by_protein(dataset csv, pdb,
ordered_segments, 30), --no-mask_inter_segment, OpenFoldAF2Rank(recycles 6, DeepSpeed off, mask_sidechains, the
run_af2rank_prediction defaults otherwise), _score_protein_with_scorer with the persisted cg2all all-atom decoys, then
step 5 = proteina_analysis.enrich_af2rank_output_dir (reference _find_reference_cif / _default_chain).
Inputs come READ-ONLY from the production tars: the top-5 CA samples, the persisted cg2all_topk_structures and the
stored af2rank CSVs (compared decoy by decoy). Nothing is written next to the production data.

  stage : extract the inputs of --pids into <out>/inputs/<pid>/ (run once, before the GPU processes)
  score : one model x one kernel set over --pids -> <out>/<kernels>/<pid>/<sub>/ + compare_<kernels>_<model>.csv
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank
from proteinfoundation.prediction_pipeline.proteina_analysis import (
    _default_chain, _find_reference_cif, enrich_af2rank_output_dir)
from proteinfoundation.prediction_pipeline.run_af2rank_prediction import (
    _load_protein_ids, _pick_reference, _score_protein_with_scorer, _segments_by_protein, _select_topk_df)
from proteinfoundation.prediction_pipeline.usalign_tabular import resolve_usalign_exe

SUB = {"model_1_ptm": "af2rank_analysis", "model_2_ptm": "af2rank_analysis_model_2_ptm"}
KERNELS = {"cueq": (True, True), "plain": (False, False)}  # (use_cuequivariance_attention, ..._multiplicative_update)
COLS = ["ptm", "plddt", "tm_template_pred", "tm_ref_pred", "tm_ref_template", "composite"]
EBM_SUB = "proteinebm_v2_cathmd_analysis"


def stage(args):
    for pid in args.pids.split(","):
        d = Path(args.out) / "inputs" / pid
        if (d / "STAGED").exists():
            print(f"[stage] {pid}: already staged", flush=True)
            continue
        (d / "stored").mkdir(parents=True, exist_ok=True)
        (d / "cg2all").mkdir(exist_ok=True)
        with tarfile.open(os.path.join(args.tar_dir, f"{pid}.tar"), "r:") as tf:
            def get(member, dest):
                with tf.extractfile(f"{pid}/{member}") as src, open(dest, "wb") as fh:
                    shutil.copyfileobj(src, fh)
            get(f"{EBM_SUB}/proteinebm_scores_{pid}.csv", d / "stored" / f"proteinebm_scores_{pid}.csv")
            for sub in SUB.values():
                get(f"{args.variant}/{sub}/af2rank_scores_{pid}.csv", d / "stored" / f"{sub}.csv")
                get(f"{args.variant}/{sub}/af2rank_summary_{pid}.json", d / "stored" / f"{sub}_summary.json")
            topk = _select_topk_df(pd.read_csv(d / "stored" / f"proteinebm_scores_{pid}.csv"), d, args.top_k)
            names = [Path(p).name for p in topk["structure_path"]]
            for sub in SUB.values():
                stored = sorted(pd.read_csv(d / "stored" / f"{sub}.csv")["structure_file"].astype(str))
                assert stored == sorted(names), f"{pid} {sub}: stored decoys {stored} != energy top-{args.top_k} {names}"
            for n in names:
                get(n, d / n)
                get(f"{args.variant}/cg2all_topk_structures/{Path(n).stem}_allatom.pdb",
                    d / "cg2all" / f"{Path(n).stem}_allatom.pdb")
        (d / "STAGED").write_text("\n".join(names) + "\n")
        print(f"[stage] {pid}: {len(names)} decoys {names}", flush=True)


def score(args):
    os.environ["USALIGN_PATH"] = resolve_usalign_exe(None, required=True)
    print(f"[score] USALIGN_PATH {os.environ['USALIGN_PATH']}", flush=True)
    seg, known = {}, set()
    for csv in args.dataset_csvs.split(","):
        ids = set(_load_protein_ids(csv, "pdb"))
        assert not ids & known, f"{csv}: protein ids repeated across dataset csvs"
        known |= ids
        seg.update(_segments_by_protein(csv, "pdb", "ordered_segments", args.segment_min_len))
    cue_attn, cue_mul = KERNELS[args.kernels]
    scorer, rows = None, []
    for pid in args.pids.split(","):
        d = Path(args.out) / "inputs" / pid
        assert (d / "STAGED").exists(), f"{pid}: not staged"
        assert pid in known, f"{pid}: not in any dataset csv (production segments unknown)"
        topk = _select_topk_df(pd.read_csv(d / "stored" / f"proteinebm_scores_{pid}.csv"), d, args.top_k)
        assert all(Path(p).exists() for p in topk["structure_path"]), f"{pid}: CA decoys missing"
        allatom = {str(p): str(d / "cg2all" / f"{Path(p).stem}_allatom.pdb") for p in topk["structure_path"]}
        assert all(os.path.exists(v) for v in allatom.values()), f"{pid}: cg2all decoys missing"
        reference, chain = _pick_reference(topk, args.cif_dir, pid)
        assert reference.endswith(".cif"), f"{pid}: no native cif under {args.cif_dir}"
        if scorer is None:
            scorer = OpenFoldAF2Rank(
                reference_pdb=reference, chain=chain, model_name=args.model, recycles=args.recycles,
                use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=cue_attn,
                use_cuequivariance_multiplicative_update=cue_mul, skip_ref_metrics=False,
                compile_inference_path=False, inference_attn_kernel="sdpa", compile_strategy="per_block",
                use_cueq_triangle_mul=False, mask_sidechains=True)
        else:
            scorer.reset_reference(reference, chain)
        scorer.set_segments(seg.get(pid))
        scorer._mask_inter_segment = False
        out_dir = Path(args.out) / args.kernels / pid / SUB[args.model]
        _score_protein_with_scorer(scorer=scorer, protein_id=pid, topk_df=topk, output_dir=out_dir,
                                   recycles=args.recycles, filter_existing=True, allatom_map=allatom)
        ref5, chain5 = _find_reference_cif(pid, args.cif_dir), _default_chain(pid)
        js = json.load(open(d / "stored" / f"{SUB[args.model]}_summary.json"))
        assert (os.path.basename(js["reference_structure"]), js["chain"]) == (os.path.basename(ref5), chain5), \
            f"{pid}: production step 5 used {js['reference_structure']} {js['chain']}, here {ref5} {chain5}"
        enrich_af2rank_output_dir(protein_id=pid, inference_dir=str(d), output_dir=str(out_dir),
                                  reference_cif=ref5, reference_chain=chain5)
        new = pd.read_csv(out_dir / f"af2rank_scores_{pid}.csv").set_index("structure_file")
        old = pd.read_csv(d / "stored" / f"{SUB[args.model]}.csv").set_index("structure_file")
        assert sorted(new.index) == sorted(old.index) and len(new) == args.top_k, f"{pid}: decoy sets differ"
        assert "error" not in new.columns or new["error"].isna().all(), f"{pid}: scorer errors {new['error'].tolist()}"
        for f in new.index:
            r = {"pid": pid, "structure_file": f, "model": args.model, "kernels": args.kernels,
                 "n_segments": len(seg[pid]) if pid in seg else 0}
            for c in COLS:
                r[f"{c}_new"], r[f"{c}_stored"] = float(new.at[f, c]), float(old.at[f, c])
                assert (r[f"{c}_new"] != r[f"{c}_new"]) == (r[f"{c}_stored"] != r[f"{c}_stored"]), f"{pid} {f} {c}: NaN on one side"
            rows.append(r)
        print(f"[score] {pid} {args.model} {args.kernels}: max |new - stored| ptm "
              f"{max(abs(x['ptm_new'] - x['ptm_stored']) for x in rows if x['pid'] == pid):.6f}", flush=True)
    out_csv = Path(args.out) / f"compare_{args.kernels}_{args.model}_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    for c in COLS:
        print(f"[score] {args.kernels} {args.model} {c}: max |new - stored| {(df[c + '_new'] - df[c + '_stored']).abs().max():.6f}")
    print(f"D36W8_OURS_DONE {len(rows)} rows -> {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "score"])
    ap.add_argument("--pids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tar-dir", required=True, help="production seq_cond_segment dir (read-only)")
    ap.add_argument("--variant", default="af2rank_uniform_rc6")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", choices=sorted(SUB))
    ap.add_argument("--kernels", choices=sorted(KERNELS))
    ap.add_argument("--tag", default="", help="suffix of the compare csv (one per job)")
    ap.add_argument("--recycles", type=int, default=6)
    ap.add_argument("--segment-min-len", type=int, default=30)
    ap.add_argument("--dataset-csvs", default="")
    ap.add_argument("--cif-dir", default="")
    args = ap.parse_args()
    if args.cmd == "stage":
        stage(args)
    else:
        assert args.model and args.kernels and args.dataset_csvs and args.cif_dir, "score needs model/kernels/csvs/cif"
        score(args)


if __name__ == "__main__":
    sys.exit(main())
