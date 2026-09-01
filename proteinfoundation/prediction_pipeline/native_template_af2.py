#!/usr/bin/env python3
"""Fold each target from its OWN native structure used as an AF2 template (full_template mode).

The GT-TEMPLATE control for the distogram-only protocol. `contactebm_distogram_af2.py` feeds AF2 a
[L,L,39] distogram as a synthetic template, which zeroes the template unit-vector and torsion-angle
channels; this script feeds the real mmCIF instead, so those channels carry true geometry. Everything
else -- query sequence/mask (read from the SAME .pt), model set, recycles, seed, confidence readout
and output layout -- is identical, making the pair a one-variable contrast between "the distogram of
the native" and "the native itself" as the template representation.

Writes one PDB per (target, model) plus a JSONL row, in the layout `score_d36h.py` already reads.

Usage:
  python -m proteinfoundation.prediction_pipeline.native_template_af2 \
      --samples_dir <dir of *.pt> --natives_dir <dir of *.cif> --out_dir <dir> \
      --params_dir <dir with params_model_1_ptm.npz> --model_name model_1_ptm,model_2_ptm --recycles 6
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch

from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


def _write_pdb(atom37, atom37_mask, aatype, out_pdb):
    write_prot_to_pdb(
        atom37,
        out_pdb,
        aatype=aatype,
        atom37_mask=atom37_mask,
        residue_index=np.arange(atom37.shape[0], dtype=np.int32) + 1,
        chain_index=np.zeros(atom37.shape[0], dtype=np.int32),
        overwrite=True,
        no_indexing=True,
    )


def _confidence(out, n_res):
    plddt = out["plddt"]
    assert plddt.shape[0] == n_res, f"plddt length {plddt.shape[0]} != n_res {n_res}"
    pae = out["predicted_aligned_error"]
    scores = {
        "plddt": float(plddt.mean().item()) / 100.0,
        "ptm": float(out["ptm_score"].item()),
        "pae_mean": float(pae.mean().item()),
    }
    scores["composite"] = scores["ptm"] * scores["plddt"]
    return scores, plddt.detach().cpu().numpy(), pae.detach().cpu().numpy()


def _native_cif(vid, natives_dir, pdb_mirror):
    entry = vid.rsplit("_", 1)[0]
    p = os.path.join(natives_dir, entry + ".cif")
    if os.path.exists(p):
        return p
    p = os.path.join(pdb_mirror, entry[1:3], entry + ".cif")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"{vid}: no native cif under {natives_dir} or {pdb_mirror}")


def main(args):
    kalign = shutil.which("kalign")
    if kalign is None:
        raise RuntimeError("kalign is required for full_template featurization but is not on PATH")

    os.makedirs(args.out_dir, exist_ok=True)
    pdb_dir = os.path.join(args.out_dir, "pdb")
    conf_dir = os.path.join(args.out_dir, "confidence")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "af2_confidence.jsonl")

    model_names = [m.strip() for m in args.model_name.split(",") if m.strip()]
    done = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["sample"], r.get("model", model_names[0])))

    files = sorted(p for p in os.listdir(args.samples_dir) if p.endswith(".pt"))
    if args.limit is not None:
        files = files[: args.limit]
    print(f"{len(files)} targets | models {model_names} | recycles {args.recycles} "
          f"| kalign {kalign} | {len(done)} rows already done", flush=True)

    for model_name in model_names:
        params = os.path.join(args.params_dir, f"params_{model_name}.npz")
        infer = OpenFoldTemplateInference(
            model_name=model_name,
            jax_params_path=params,
            use_deepspeed_evoformer_attention=args.deepspeed_attn,
            max_recycling_iters=args.recycles,
        )
        tag = model_name.replace("model_", "m").replace("_ptm", "")
        print(f"=== AF2 pass: {model_name} (tag {tag}) recycles={args.recycles} ===", flush=True)

        for ei, fname in enumerate(files):
            d = torch.load(os.path.join(args.samples_dir, fname), map_location="cpu",
                           weights_only=False)
            vid = d["id"]
            if (vid, 0, model_name) in done:
                continue
            rmask = d["residue_mask"].float()
            n_res = int(rmask.sum().item())
            aatype = d["aatype"].long()
            seq_aatype = aatype[:n_res].numpy().astype(np.int32)
            cif = _native_cif(vid, args.natives_dir, args.pdb_mirror)
            chain = vid.rsplit("_", 1)[1]

            batch = infer.build_batch(
                distogram_probs=None,
                residue_type=aatype.unsqueeze(0).to(infer.device),
                mask=rmask.unsqueeze(0).to(infer.device),
                template_mode="full_template",
                template_mmcif_path=cif,
                template_chain_id=chain,
                kalign_binary_path=kalign,
                mask_template_aatype=args.mask_template_aatype,
                seed=args.seed,
                skip_template_alignment=False,
            )
            with torch.no_grad():
                out = infer.model(batch)

            atom37 = out["final_atom_positions"].detach().cpu().numpy()
            if np.isnan(atom37).any():
                raise ValueError(f"{vid} {model_name}: AF2 produced NaN final_atom_positions")
            atom37_mask = out["final_atom_mask"].detach().cpu().numpy()
            scores, plddt_per_res, pae = _confidence(out, n_res)

            out_pdb = os.path.join(pdb_dir, f"{vid}_{tag}_s0.pdb")
            _write_pdb(atom37, atom37_mask, seq_aatype, out_pdb)
            np.savez_compressed(os.path.join(conf_dir, f"{vid}_{tag}_s0.npz"),
                                plddt=plddt_per_res, pae=pae)
            row = {"id": vid, "sample": 0, "model": model_name, "n_res": n_res,
                   "template_cif": cif, "template_chain": chain, "pdb": out_pdb, **scores}
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{tag} {ei + 1}/{len(files)}] {vid}: L={n_res} "
                  f"plddt {scores['plddt']:.3f} ptm {scores['ptm']:.3f} "
                  f"pae {scores['pae_mean']:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True,
                    help="dir of per-target .pt with id / aatype / residue_mask -- point it at the "
                         "GT-distogram control's payload so the query is byte-identical")
    ap.add_argument("--natives_dir", required=True)
    ap.add_argument("--pdb_mirror", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--params_dir", required=True)
    ap.add_argument("--model_name", default="model_1_ptm")
    ap.add_argument("--recycles", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mask_template_aatype", action="store_true", default=False,
                    help="AF2Rank's setting; OFF by default so this matches the distogram-only run, "
                         "which shows AF2 the real query sequence in the template row")
    ap.add_argument("--deepspeed_attn", action="store_true", default=False)
    main(ap.parse_args())
