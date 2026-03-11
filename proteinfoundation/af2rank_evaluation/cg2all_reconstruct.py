#!/usr/bin/env python3
"""
cg2all batch reconstruction script.

Standalone script intended to run in the cue_openfold environment to reconstruct
CA-only PDB files to all-atom PDBs using cg2all. Designed to be called as a
subprocess from other environments (e.g. colabdesign) that lack PyTorch/cg2all.

Usage:
    conda run -n cue_openfold python cg2all_reconstruct.py \
        --inputs inputs.json \
        --output_dir /tmp/cg2all_out/ \
        --output_map output_map.json

inputs.json: JSON list of input CA-only PDB file paths
output_map.json: written on success, maps input_path -> reconstructed_path
"""

import argparse
import json
import os
import sys
import tempfile

import torch
import dgl
import cg2all.lib.libcg
import cg2all.lib.libmodel
from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import PredictionData, create_trajectory_from_batch
from cg2all.lib.libter import patch_termini


def main():
    parser = argparse.ArgumentParser(description="Batch cg2all CA->all-atom reconstruction")
    parser.add_argument("--inputs", required=True, help="JSON file containing list of input PDB paths")
    parser.add_argument("--output_dir", required=True, help="Directory to write reconstructed PDBs")
    parser.add_argument("--output_map", required=True, help="JSON file to write input->output path mapping")
    args = parser.parse_args()

    with open(args.inputs) as f:
        pdb_files = json.load(f)

    if not pdb_files:
        with open(args.output_map, 'w') as f:
            json.dump({}, f)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = MODEL_HOME / "CalphaBasedModel.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["hyper_parameters"]
    cg_model = cg2all.lib.libcg.CalphaBasedModel
    config = cg2all.lib.libmodel.set_model_config(config, cg_model, flattened=False)
    model = cg2all.lib.libmodel.Model(config, cg_model, compute_loss=False)
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()

    graphs = []
    valid_indices = []
    for i, pdb_file in enumerate(pdb_files):
        try:
            ds = PredictionData(pdb_file, cg_model, radius=config.globals.radius)
            g = ds[0]
            graphs.append(g)
            valid_indices.append(i)
        except Exception as e:
            print(f"Warning: cg2all failed to load {pdb_file}: {e}", file=sys.stderr)

    result = {}
    if graphs:
        batched = dgl.batch(graphs).to(device)
        with torch.no_grad():
            R = model.forward(batched)[0]["R"]
        traj_s, _ = create_trajectory_from_batch(batched, R)

        for idx, traj in zip(valid_indices, traj_s):
            output = patch_termini(traj)
            out_name = os.path.splitext(os.path.basename(pdb_files[idx]))[0] + "_allatom.pdb"
            out_path = os.path.join(args.output_dir, out_name)
            output.save(out_path)
            result[pdb_files[idx]] = out_path

    with open(args.output_map, 'w') as f:
        json.dump(result, f)

    print(f"Reconstructed {len(result)}/{len(pdb_files)} structures", file=sys.stderr)


if __name__ == "__main__":
    main()
