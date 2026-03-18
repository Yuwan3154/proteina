#!/usr/bin/env python3
"""
cg2all batch reconstruction script.

Standalone script intended to run in the cue_openfold environment to reconstruct
CA-only PDB files to all-atom PDBs using cg2all. Designed to be called as a
subprocess from other environments (e.g. colabdesign) that lack PyTorch/cg2all.

Usage:
    /path/to/cue_openfold/bin/python cg2all_reconstruct.py \
        --inputs inputs.json \
        --output_dir /tmp/cg2all_out/ \
        --output_map output_map.json

inputs.json: JSON list of input CA-only PDB file paths
output_map.json: written on success, maps input_path -> reconstructed_path
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import torch
import dgl
import cg2all.lib.libcg
import cg2all.lib.libmodel
from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import PredictionData, create_trajectory_from_batch
from cg2all.lib.libter import patch_termini


# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------

# Google Drive file IDs and Zenodo record for each cg2all model variant.
_GDRIVE_IDS = {
    "CalphaBasedModel": "1uzsVPB_0t0RDp2P8qJ44LzE3JiVowtTx",
    "ResidueBasedModel": "1KsxfB0B90YQQd1iBzw3buznHIwzN_0sA",
    "SidechainModel": "1kd_wq-Jq6z4CpRNLwkeAgPLQOl_xL64q",
    "CalphaCMModel": "1kLrmeO2F0WXvy0ujq0H4U5drjnuxNy8d",
    "CalphaSCModel": "1sFW-g1_1fOYUtKVi7I8898Xs-NiKgG5R",
    "BackboneModel": "17OZDDCiwo7M8egPgRIlMfHujOT-oy0Fz",
    "MainchainModel": "1Q6Xlop_u1hQdLwTlHHdCDxWTC34I8TQg",
    "Martini": "1GiEtLiIOotLrj--7-jJI8aRE10duQoBE",
    "Martini3": "1oqz8BVheg534BydxPL6bFc-J2OeHOcfn",
    "PRIMO": "1FW_QFijewI-z48GC-aDEjHMO_8g1syTH",
    "CalphaBasedModel-FIX": "16FfIW72BDy-RT46kgVoRsGYCcpHOeee1",
    "CalphaCMModel-FIX": "1xdDT-6kkkNiXcg3WxJm1gkw7wDj07Mw9",
    "CalphaSCModel-FIX": "1ODp46hxHlfDiSVSbNwrVQvk1xugtuAH9",
    "BackboneModel-FIX": "1uosDHt20KokQBMqyZylO0m8VEONcEuK6",
    "MainchainModel-FIX": "1TaOn42s-3HPlxB4sJ8V21g8rO447F4_v",
}
_ZENODO_RECORD = "8393343"


def download_ckpt_file(model_type: str, ckpt_fn: Path, fix_atom: bool = False) -> None:
    """Download a cg2all checkpoint file from Google Drive (preferred) or Zenodo.

    Args:
        model_type: Base model name, e.g. ``"CalphaBasedModel"`` or ``"BackboneModel"``.
        ckpt_fn:    Destination path for the ``.ckpt`` file.
        fix_atom:   If True, appends ``"-FIX"`` to the model_type key (fix-atom variant).
    """
    if fix_atom:
        model_type = f"{model_type}-FIX"

    if model_type not in _GDRIVE_IDS:
        raise ValueError(f"Unknown cg2all model type: {model_type!r}. "
                         f"Known types: {sorted(_GDRIVE_IDS)}")

    ckpt_fn = Path(ckpt_fn)
    ckpt_fn.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
        sys.stdout.write(f"Downloading from Google Drive ... {ckpt_fn}\n")
        sys.stdout.flush()
        gdown.download(id=_GDRIVE_IDS[model_type], output=str(ckpt_fn), quiet=True)
    except Exception:
        import requests
        sys.stdout.write(f"Downloading from Zenodo ... {ckpt_fn}\n")
        sys.stdout.flush()
        url = f"https://zenodo.org/record/{_ZENODO_RECORD}/files/{ckpt_fn.name}"
        with open(ckpt_fn, "wb") as fout:
            fout.write(requests.get(url).content)


def ensure_ckpt(model_type: str, fix_atom: bool = False) -> Path:
    """Return path to checkpoint, downloading it first if not present."""
    suffix = "-FIX" if fix_atom else ""
    ckpt_path = MODEL_HOME / f"{model_type}{suffix}.ckpt"
    if not ckpt_path.exists():
        sys.stdout.write(
            f"cg2all checkpoint not found at {ckpt_path}; downloading...\n"
        )
        sys.stdout.flush()
        download_ckpt_file(model_type, ckpt_path, fix_atom=fix_atom)
    return ckpt_path


def _fix_pdb_model_number(pdb_path: str) -> None:
    """Rewrite MODEL 0 → MODEL 1 in cg2all PDB output.

    MDAnalysis (used by cg2all) writes 0-indexed MODEL records, but the PDB
    format standard and OpenFold's template reader expect 1-indexed models.
    """
    with open(pdb_path) as f:
        text = f.read()
    patched = re.sub(r"^MODEL(\s+)0(\s*)$", r"MODEL\g<1>1\2", text, flags=re.MULTILINE)
    if patched != text:
        with open(pdb_path, "w") as f:
            f.write(patched)


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
    ckpt_path = ensure_ckpt("CalphaBasedModel")
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

    CG2ALL_BATCH_SIZE = 32
    result = {}
    for start in range(0, len(graphs), CG2ALL_BATCH_SIZE):
        chunk_graphs = graphs[start:start + CG2ALL_BATCH_SIZE]
        chunk_indices = valid_indices[start:start + CG2ALL_BATCH_SIZE]
        batched = dgl.batch(chunk_graphs).to(device)
        with torch.no_grad():
            R = model.forward(batched)[0]["R"]
        traj_s, _ = create_trajectory_from_batch(batched, R)
        for idx, traj in zip(chunk_indices, traj_s):
            output = patch_termini(traj)
            out_name = os.path.splitext(os.path.basename(pdb_files[idx]))[0] + "_allatom.pdb"
            out_path = os.path.join(args.output_dir, out_name)
            output.save(out_path)
            _fix_pdb_model_number(out_path)
            result[pdb_files[idx]] = out_path

    with open(args.output_map, 'w') as f:
        json.dump(result, f)

    print(f"Reconstructed {len(result)}/{len(pdb_files)} structures", file=sys.stderr)


if __name__ == "__main__":
    main()
