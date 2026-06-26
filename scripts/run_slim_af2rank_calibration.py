"""WS1: AF2Rank-style confidence-calibration test of the slim AlphaFold2 model.

For each target, score a set of decoys (decoy-as-template, single-seq) with the slim
model and/or stock model_1_ptm, then measure per-target Spearman(confidence, decoy
quality). The slim model's Evoformer reps changed but its heads are stock-AF2 (frozen),
so the composite (pTM x pLDDT) may be miscalibrated -- this quantifies it.

Metric (AF2Rank calibration): mean over targets of Spearman(composite, tm_ref_template),
where tm_ref_template = TM-score(decoy, native). Also reports Spearman for the pTM and
pLDDT components separately.

Layout (parameterized -- confirm with the actual Rosetta-decoy paths at launch):
  --decoy_root/<target_id>/*.pdb     one subdir of decoy PDBs per target
  --native_dir/<target_id>.pdb|.cif  native reference per target

The model is loaded ONCE per mode; reset_reference() swaps the native between targets.
"""
import argparse
import gc
import glob
import os

import pandas as pd
import torch
from scipy.stats import spearmanr

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank


def find_native(native_dir, tid):
    for ext in (".pdb", ".cif"):
        p = os.path.join(native_dir, tid + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no native ({tid}.pdb/.cif) in {native_dir}")


def list_targets(decoy_root, targets_file):
    if targets_file:
        return [ln.strip() for ln in open(targets_file) if ln.strip()]
    return sorted(
        d for d in os.listdir(decoy_root) if os.path.isdir(os.path.join(decoy_root, d))
    )


def list_decoys(decoy_root, tid, max_decoys):
    decoys = sorted(glob.glob(os.path.join(decoy_root, tid, "*.pdb")))
    return decoys[:max_decoys] if max_decoys else decoys


def build_scorer(mode, ref_pdb, chain, args):
    kw = dict(
        reference_pdb=ref_pdb,
        chain=chain,
        model_name="model_1_ptm",
        recycles=args.recycles,
        skip_ref_metrics=False,
        usalign_path=args.usalign_path,
        use_mlm=args.use_mlm,
    )
    if mode == "slim":
        kw.update(
            slim_ckpt_path=args.slim_ckpt_path,
            evoformer_keep_block_indices=args.keep,
            use_ema=args.use_ema,
        )
    return OpenFoldAF2Rank(**kw)


def run_mode(mode, targets, args):
    rows = []
    scorer = None
    for ti, tid in enumerate(targets):
        native = find_native(args.native_dir, tid)
        if scorer is None:
            scorer = build_scorer(mode, native, args.chain, args)
        else:
            scorer.reset_reference(native, args.chain)
        decoys = list_decoys(args.decoy_root, tid, args.max_decoys)
        for dp in decoys:
            s = scorer.score_structure(dp, decoy_chain=args.chain, recycles=args.recycles)
            rows.append({
                "mode": mode,
                "target": tid,
                "decoy": os.path.basename(dp),
                "composite": s.get("composite"),
                "ptm": s.get("ptm"),
                "plddt": s.get("plddt"),
                "pae_mean": s.get("pae_mean"),
                "tm_ref_template": s.get("tm_ref_template"),
                "tm_ref_pred": s.get("tm_ref_pred"),
                "tm_io": s.get("tm_template_pred"),  # TM(decoy, output) — the AF2Rank composite's 3rd factor
            })
        print(f"[{mode}] {ti + 1}/{len(targets)} {tid}: {len(decoys)} decoys", flush=True)
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def per_target_spearman(df, score_col, truth_col="tm_ref_template"):
    rhos = []
    for tid, g in df.groupby("target"):
        ok = g[[score_col, truth_col]].dropna()
        if len(ok) >= 3 and ok[score_col].nunique() > 1 and ok[truth_col].nunique() > 1:
            rhos.append((tid, spearmanr(ok[score_col], ok[truth_col]).correlation))
    return pd.DataFrame(rhos, columns=["target", f"rho_{score_col}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoy_root", required=True, help="per-target subdirs of decoy *.pdb")
    ap.add_argument("--native_dir", required=True, help="native {target_id}.pdb/.cif per target")
    ap.add_argument("--targets_file", default=None, help="optional list of target ids (default: subdirs of decoy_root)")
    ap.add_argument("--slim_ckpt_path", default=None)
    ap.add_argument("--keep", default="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47")
    ap.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--recycles", type=int, default=1)  # AF2Rank protocol (total iters = recycles+1)
    ap.add_argument("--use_mlm", action=argparse.BooleanOptionalAction, default=False)  # AF2Rank: masked_msa at config default (~0.15); False = forced 0
    ap.add_argument("--chain", default="A")
    ap.add_argument("--max_decoys", type=int, default=0, help="0 = all")
    ap.add_argument("--modes", default="slim,stock", help="comma list: slim,stock")
    ap.add_argument("--usalign_path", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = list_targets(args.decoy_root, args.targets_file)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "slim" in modes and not args.slim_ckpt_path:
        raise ValueError("modes includes 'slim' but --slim_ckpt_path not given")

    summary = []
    for mode in modes:
        df = run_mode(mode, targets, args)
        df.to_csv(os.path.join(args.out_dir, f"calibration_decoys_{mode}.csv"), index=False)
        for col in ("composite", "ptm", "plddt"):
            st = per_target_spearman(df, col)
            mean_rho = st[f"rho_{col}"].mean()
            summary.append({
                "mode": mode,
                "score": col,
                "mean_per_target_spearman": mean_rho,
                "n_targets": len(st),
            })
            print(f"[{mode}] mean per-target Spearman({col}, tm_ref_template) = "
                  f"{mean_rho:.4f} over {len(st)} targets", flush=True)

    out_summary = os.path.join(args.out_dir, "calibration_summary.csv")
    pd.DataFrame(summary).to_csv(out_summary, index=False)
    print("WROTE", out_summary, flush=True)


if __name__ == "__main__":
    main()
