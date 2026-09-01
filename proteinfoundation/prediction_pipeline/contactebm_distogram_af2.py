#!/usr/bin/env python3
"""Fold ContactEBM's sampled distograms with the distogram-only AF2 template protocol.

Consumer side of ContactEBM path (b): each per-target .pt written by
contact_ebm.scripts.sample_distogram holds a [S, L, L, 39] distogram-probability tensor in the AF2
template binning (3.25-50.75 A, 39 bins -- the same convention openfold's
config.model.template.distogram declares), so it is injectable unchanged as a synthetic template.
Nothing new is trained here: the structure comes from AF2's own structure module and the confidence
from AF2's own pretrained plddt / ptm / PAE heads.

Writes one PDB per (target, sample, model) plus a JSONL row of the confidence readout.

--model_name accepts a COMMA-SEPARATED list and --recycles sets AF2's trunk iteration count, so
this arm can be made NFE-identical to proteina's af2rank stage (top-k through model_1_ptm AND
model_2_ptm at recycles=6 => 14 trunk passes per sample). At their defaults the behaviour is the
single-model, openfold-default-recycles path the original 66-target run used.

Usage:
  python -m proteinfoundation.prediction_pipeline.contactebm_distogram_af2 \
      --samples_dir <dir of *.pt> --out_dir <dir> --params_dir <dir with params_model_1_ptm.npz>
"""

import argparse
import json
import os

import numpy as np
import torch

import openfold.np.residue_constants as rc
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
    """AF2's own pretrained readout. No padding is applied on this path (single-sample forward with
    compilation off), so the non-mask-aware compute_tm/compute_plddt see exactly the real residues."""
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


def main(args):
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
                # rows written before the multi-model change carry no "model" key
                done.add((r["id"], r["sample"], r.get("model", model_names[0])))

    files = sorted(p for p in os.listdir(args.samples_dir) if p.endswith(".pt"))
    if args.limit is not None:
        files = files[: args.limit]
    print(f"{len(files)} sample files | models {model_names} | recycles {args.recycles} "
          f"| {len(done)} rows already done", flush=True)

    for model_name in model_names:
        params = os.path.join(args.params_dir, f"params_{model_name}.npz")
        # recycles=None keeps openfold's data.common.max_recycling_iters default (3 -> 4 iterations)
        infer = OpenFoldTemplateInference(
            model_name=model_name,
            jax_params_path=params,
            use_deepspeed_evoformer_attention=args.deepspeed_attn,
            max_recycling_iters=args.recycles,
        )
        tag = model_name.replace("model_", "m").replace("_ptm", "")
        print(f"=== AF2 pass: {model_name} (tag {tag}) recycles={args.recycles} ===", flush=True)

        for fi, fname in enumerate(files):
            d = torch.load(os.path.join(args.samples_dir, fname), map_location="cpu",
                           weights_only=False)
            vid = d["id"]
            rmask = d["residue_mask"].float()
            n_res = int(rmask.sum().item())
            aatype = d["aatype"].long()
            seq_aatype = aatype[:n_res].numpy().astype(np.int32)
            n_samples = d["distogram_probs"].shape[0]

            for s_i in range(n_samples):
                if (vid, s_i, model_name) in done:
                    continue
                dgram = d["distogram_probs"][s_i].float().unsqueeze(0).to(infer.device)
                # sample_distogram already softmaxes; the float16 round-trip perturbs the sum, and
                # openfold's feats.py documents the injected tensor as summing to 1 over bins.
                dgram = dgram / dgram.sum(dim=-1, keepdim=True)
                rt = aatype.unsqueeze(0).to(infer.device)
                mk = rmask.unsqueeze(0).to(infer.device)

                out = infer(dgram, rt, mk, template_mode="distogram_only", seed=args.seed)
                atom37 = out["final_atom_positions"].detach().cpu().numpy()
                if np.isnan(atom37).any():
                    raise ValueError(f"{vid} sample {s_i} {model_name}: AF2 produced NaN "
                                     f"final_atom_positions")
                atom37_mask = out["final_atom_mask"].detach().cpu().numpy()
                scores, plddt_per_res, pae = _confidence(out, n_res)

                out_pdb = os.path.join(pdb_dir, f"{vid}_{tag}_s{s_i}.pdb")
                _write_pdb(atom37, atom37_mask, seq_aatype, out_pdb)
                np.savez_compressed(os.path.join(conf_dir, f"{vid}_{tag}_s{s_i}.npz"),
                                    plddt=plddt_per_res, pae=pae)
                row = {"id": vid, "sample": s_i, "model": model_name, "n_res": n_res,
                       "pdb": out_pdb, **scores}
                with open(out_jsonl, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{tag} {fi + 1}/{len(files)}] {vid} s{s_i}: L={n_res} "
                      f"plddt {scores['plddt']:.3f} ptm {scores['ptm']:.3f} "
                      f"pae {scores['pae_mean']:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--params_dir", required=True)
    ap.add_argument("--model_name", default="model_1_ptm",
                    help="one name, or a comma-separated list run as successive passes")
    ap.add_argument("--recycles", type=int, default=None,
                    help="AF2 max_recycling_iters; the trunk runs recycles+1 iterations. "
                         "None = openfold default (3). proteina's af2rank arm uses 6.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--deepspeed_attn", action="store_true", default=False)
    main(ap.parse_args())
