#!/usr/bin/env python3
"""
ProteinEBM Scorer (single-pass energy at t=0.1)

This script scores Proteina-generated decoy PDBs with ProteinEBM and writes results to:
  <inference_output_dir>/proteinebm_analysis/

It is designed to be used as a subprocess from the existing evaluation pipeline.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from ml_collections import ConfigDict

from protein_ebm.data.protein_utils import residues_to_features
from protein_ebm.model.boltz_utils import center_random_augmentation
from protein_ebm.model.ebm import ProteinEBM
from protein_ebm.model.r3_diffuser import R3Diffuser


def load_proteinebm_model(config_path: str, checkpoint_path: str, device: torch.device) -> Tuple[ProteinEBM, ConfigDict]:
    config_path = os.path.abspath(os.path.expanduser(config_path))
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config)

    diffuser = R3Diffuser(config.diffuser)
    model = ProteinEBM(config.model, diffuser).to(device)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model")}
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def _pdb_to_chain_residues(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("decoy", pdb_path)
    chain = list(structure.get_chains())[0]
    return [r for r in chain.get_residues() if is_aa(r)]


def _extract_ca_coords(structure_path: str, chain_id: str | None) -> np.ndarray:
    """
    Extract CA coordinates from a PDB or CIF file for a specific chain.

    - If chain_id is None and there is exactly 1 chain, use it.
    - If chain_id is provided but not present:
        - if there is exactly 1 chain, use it
        - else raise with available chain IDs
    """
    is_cif = structure_path.lower().endswith(".cif")
    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("ref", structure_path)
    model = list(structure.get_models())[0]
    chains = list(model.get_chains())
    chain_ids = [c.id for c in chains]

    # Heuristic chain mappings (copied from AF2Rank scorer logic) for cases where
    # the dataset's "A/B/C" chain IDs don't match author-provided CIF chain IDs.
    chain_mappings: dict[str, list[str]] = {
        "A": ["E", "N", "X", "1", "a"],
        "B": ["F", "O", "Y", "2", "b"],
        "C": ["G", "P", "Z", "3", "c"],
    }

    if chain_id is None:
        if len(chains) != 1:
            raise ValueError(f"chain_id not provided and structure has multiple chains: {chain_ids}")
        chain = chains[0]
    else:
        if chain_id in chain_ids:
            chain = next(c for c in chains if c.id == chain_id)
        else:
            # Try common chain remappings (A->E, etc.)
            mapped = None
            for alt in chain_mappings.get(chain_id, []):
                if alt in chain_ids:
                    mapped = alt
                    break
            if mapped is not None:
                chain = next(c for c in chains if c.id == mapped)
            elif len(chains) == 1:
                chain = chains[0]
            else:
                raise ValueError(f"Chain '{chain_id}' not found in {structure_path}. Available chains: {chain_ids}")

    coords: list[np.ndarray] = []
    for res in chain.get_residues():
        if not is_aa(res):
            continue
        if "CA" not in res:
            continue
        coords.append(res["CA"].get_coord())

    if not coords:
        raise ValueError(f"No CA atoms found in {structure_path} (chain={chain.id})")

    return np.asarray(coords, dtype=np.float32)


def _parse_usalign_output(lines: list[str]) -> Dict[str, float]:
    parse_float = lambda x: float(x.split("=")[1].split()[0])
    out = {"rms": 0.0, "tms": 0.0, "gdt": 0.0}
    for line in lines:
        line = line.rstrip()
        if line.startswith("RMSD"):
            out["rms"] = parse_float(line)
        elif line.startswith("TM-score"):
            out["tms"] = parse_float(line)
        elif line.startswith("GDT-TS-score"):
            out["gdt"] = parse_float(line)
    return out


def tmscore_ca_coords(x: np.ndarray, y: np.ndarray, tmscore_exe: str = "USalign") -> Dict[str, float]:
    """
    Compute TM-score / RMSD / GDT-TS for two CA-traces using USalign (preferred) or TMscore.
    Requires the executable to be available in PATH.
    """
    exe = shutil.which(tmscore_exe) or shutil.which("USalign") or shutil.which("TMscore")
    if exe is None:
        raise FileNotFoundError("Neither USalign nor TMscore found in PATH")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f2:
        for f, z in zip([f1, f2], [x, y]):
            for k, c in enumerate(z):
                f.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                    % (k + 1, "CA", "ALA", "A", k + 1, float(c[0]), float(c[1]), float(c[2]), 1, 0)
                )
            f.flush()

    cmd = [exe, f1.name, f2.name]
    if os.path.basename(exe) == "USalign":
        cmd += ["-TMscore", "1"]

    output = subprocess.check_output(cmd, text=True)
    os.unlink(f1.name)
    os.unlink(f2.name)
    return _parse_usalign_output(output.splitlines())


def _spearmanr(x: List[float], y: List[float]) -> float:
    """
    Compute Spearman correlation without SciPy.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have same shape")
    if x_arr.size < 2:
        return float("nan")

    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

        # average ranks for ties
        sorted_a = a[order]
        i = 0
        while i < len(sorted_a):
            j = i + 1
            while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
                j += 1
            if j - i > 1:
                avg = (i + 1 + j) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    rx = rankdata(x_arr)
    ry = rankdata(y_arr)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def plot_proteinebm_results(results: List[Dict[str, object]], output_dir: str, protein_id: str) -> Dict[str, str]:
    """
    Write per-protein diagnostic plots (analogous to AF2Rank folder PNGs).

    Produces:
      - proteinebm_<protein_id>_score_vs_true_quality.png  (score=-energy vs tm_ref_template)
      - proteinebm_<protein_id>_energy_hist.png
      - proteinebm_<protein_id>_tm_hist.png
    """
    energies = np.asarray([float(r["energy"]) for r in results], dtype=np.float64)
    tm = np.asarray([float(r["tm_ref_template"]) for r in results], dtype=np.float64)
    score = -energies

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: score vs true quality
    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.scatter(tm, score, s=18, alpha=0.5)
    rho = _spearmanr(tm.tolist(), score.tolist()) if len(tm) > 1 else float("nan")
    plt.title(f"ProteinEBM: score(-energy) vs TM(ref, decoy)\n{protein_id} | Spearman ρ={rho:.3f}")
    plt.xlabel("TM-score (Reference vs Decoy) [tm_ref_template]")
    plt.ylabel("ProteinEBM score (-energy, higher is better)")
    plt.grid(True, alpha=0.3)
    score_plot = os.path.join(output_dir, f"proteinebm_{protein_id}_score_vs_true_quality.png")
    plt.tight_layout()
    plt.savefig(score_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: energy histogram
    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.hist(energies, bins=30, alpha=0.8)
    plt.title(f"ProteinEBM energy histogram\n{protein_id}")
    plt.xlabel("Energy (lower is better)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    energy_hist = os.path.join(output_dir, f"proteinebm_{protein_id}_energy_hist.png")
    plt.tight_layout()
    plt.savefig(energy_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: TM histogram
    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.hist(tm, bins=30, alpha=0.8)
    plt.title(f"TM(ref, decoy) histogram\n{protein_id}")
    plt.xlabel("TM-score (Reference vs Decoy)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    tm_hist = os.path.join(output_dir, f"proteinebm_{protein_id}_tm_hist.png")
    plt.tight_layout()
    plt.savefig(tm_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "score_vs_true_quality": os.path.abspath(score_plot),
        "energy_hist": os.path.abspath(energy_hist),
        "tm_hist": os.path.abspath(tm_hist),
    }


def build_input_feats_from_pdb(
    pdb_path: str,
    model: ProteinEBM,
    t: float,
    template_self_condition: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    residues = _pdb_to_chain_residues(pdb_path)
    atom_positions, atom_mask, aatype, residue_idx = residues_to_features(residues, strict=True)

    atom_positions = atom_positions.to(device)
    atom_mask = atom_mask.to(device)
    aatype = aatype.to(device)
    residue_idx = residue_idx.to(device)

    n_res = atom_positions.shape[0]
    residue_mask = torch.ones((1, n_res), dtype=torch.float32, device=device)
    chain_encoding = torch.zeros((1, n_res), dtype=torch.long, device=device)
    external_contacts = torch.zeros((1, n_res), dtype=torch.long, device=device)

    if model.diffuse_sidechain:
        # Flatten atom37 -> augment -> reshape to [B, N, 37*3]
        flat_coords = atom_positions.reshape(1, n_res * 37, 3)
        flat_mask = atom_mask.reshape(1, n_res * 37)
        aug_coords = center_random_augmentation(flat_coords, flat_mask)
        aug_coords = aug_coords * flat_mask[..., None]
        coords = aug_coords.reshape(1, n_res, 37 * 3)
        atom_mask_input = atom_mask.unsqueeze(0)  # [1, N, 37]
    else:
        ca_coords = atom_positions[:, 1, :].unsqueeze(0)  # [1, N, 3]
        ca_mask = atom_mask[:, 1].unsqueeze(0)  # [1, N]
        coords = center_random_augmentation(ca_coords, ca_mask)
        atom_mask_input = torch.ones((1, n_res), dtype=torch.float32, device=device)

    # Match ProteinEBM `score_decoys.py` behavior for ensemble_size=0:
    # - r_noisy is the clean coords (augmented) even when t>0
    # - selfcond_coords is the template coords when enabled
    r_noisy = coords
    selfcond_coords = coords if template_self_condition else torch.zeros_like(coords)

    input_feats = {
        "aatype": aatype.unsqueeze(0),  # [1, N]
        "mask": residue_mask,  # [1, N]
        "residue_idx": residue_idx.unsqueeze(0),  # [1, N]
        "chain_encoding": chain_encoding,  # [1, N]
        "external_contacts": external_contacts,  # [1, N]
        "selfcond_coords": selfcond_coords,  # [1, N, 3] or [1, N, 37*3]
        "r_noisy": r_noisy,  # [1, N, 3] or [1, N, 37*3]
        "atom_mask": atom_mask_input,  # [1, N] or [1, N, 37]
        "t": torch.tensor([t], dtype=torch.float32, device=device),  # [1]
    }
    return input_feats


def score_decoy_pdb(
    pdb_path: str,
    model: ProteinEBM,
    t: float,
    template_self_condition: bool,
    device: torch.device,
) -> float:
    input_feats = build_input_feats_from_pdb(
        pdb_path=pdb_path,
        model=model,
        t=t,
        template_self_condition=template_self_condition,
        device=device,
    )

    with torch.no_grad():
        out = model.compute_energy(input_feats)
    return float(out["energy"].detach().cpu().item())


def run_proteinebm_scoring_for_protein(
    protein_id: str,
    inference_output_dir: str,
    output_dir: str,
    reference_cif: str,
    reference_chain: str | None,
    proteinebm_config: str,
    proteinebm_checkpoint: str,
    t: float,
    template_self_condition: bool,
) -> Tuple[str, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = load_proteinebm_model(
        config_path=proteinebm_config,
        checkpoint_path=proteinebm_checkpoint,
        device=device,
    )

    inference_dir = Path(inference_output_dir)
    pdb_files = sorted(inference_dir.glob(f"{protein_id}_*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No decoy PDB files found under {inference_output_dir} matching {protein_id}_*.pdb")

    os.makedirs(output_dir, exist_ok=True)
    scores_csv_path = os.path.join(output_dir, f"proteinebm_scores_{protein_id}.csv")
    summary_path = os.path.join(output_dir, f"proteinebm_summary_{protein_id}.json")

    ref_coords = _extract_ca_coords(reference_cif, chain_id=reference_chain)

    start_time = time.time()
    results: List[Dict[str, object]] = []

    for pdb_path in pdb_files:
        energy = score_decoy_pdb(
            pdb_path=str(pdb_path),
            model=model,
            t=t,
            template_self_condition=template_self_condition,
            device=device,
        )
        decoy_coords = _extract_ca_coords(str(pdb_path), chain_id=None)
        tm_out = tmscore_ca_coords(ref_coords, decoy_coords)
        results.append(
            {
                "protein_id": protein_id,
                "structure_file": pdb_path.name,
                "structure_path": str(pdb_path),
                "t": t,
                "energy": energy,
                "tm_ref_template": tm_out["tms"],
                "rmsd_ref_template": tm_out["rms"],
                "gdt_ref_template": tm_out["gdt"],
            }
        )

    # Write CSV
    with open(scores_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["protein_id", "structure_file", "structure_path", "t", "energy"],
        )
        writer.fieldnames = [
            "protein_id",
            "structure_file",
            "structure_path",
            "t",
            "energy",
            "tm_ref_template",
            "rmsd_ref_template",
            "gdt_ref_template",
        ]
        writer.writeheader()
        writer.writerows(results)

    successful = results
    spearman_rho_energy = None
    max_tm_ref_template = None
    tm_ref_template_at_min_energy = None
    top_1_tm_ref_template = None
    top_5_tm_ref_template = None

    if len(successful) > 1:
        tm_vals = [float(r["tm_ref_template"]) for r in successful]
        score_vals = [-float(r["energy"]) for r in successful]  # higher is better

        spearman_rho_energy = _spearmanr(tm_vals, score_vals)
        max_tm_ref_template = float(max(tm_vals))

        energies = [float(r["energy"]) for r in successful]
        best_idx = int(np.argmin(np.asarray(energies)))
        tm_ref_template_at_min_energy = float(tm_vals[best_idx])
        top_1_tm_ref_template = tm_ref_template_at_min_energy

        top5_idx = np.argsort(np.asarray(energies))[:5]
        top_5_tm_ref_template = float(max([tm_vals[int(i)] for i in top5_idx]))

    runtime_s = time.time() - start_time
    plot_paths = plot_proteinebm_results(results, output_dir=output_dir, protein_id=protein_id)
    summary = {
        "protein_id": protein_id,
        "total_structures": len(results),
        "successful_scores": len(results),
        "runtime_seconds": runtime_s,
        "t": t,
        "reference_structure": os.path.abspath(os.path.expanduser(reference_cif)),
        "chain": reference_chain,
        "proteinebm_config": os.path.abspath(os.path.expanduser(proteinebm_config)),
        "proteinebm_checkpoint": os.path.abspath(os.path.expanduser(proteinebm_checkpoint)),
        "template_self_condition": template_self_condition,
        "scores_csv": os.path.abspath(scores_csv_path),
        "plots": plot_paths,
        # AF2Rank-compatible summary fields (template-quality style)
        "spearman_correlation_rho_composite": spearman_rho_energy,  # (-energy) vs tm_ref_template
        "spearman_correlation_rho_energy": spearman_rho_energy,
        "max_tm_ref_template": max_tm_ref_template,
        "tm_ref_template_at_max_composite": tm_ref_template_at_min_energy,  # best score == min energy
        "tm_ref_template_at_min_energy": tm_ref_template_at_min_energy,
        "top_1_tm_ref_template": top_1_tm_ref_template,
        "top_5_tm_ref_template": top_5_tm_ref_template,
        "status": "completed",
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return scores_csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="ProteinEBM scoring for Proteina decoys (single-pass energy)")
    parser.add_argument("--protein_id", required=True, help="Protein identifier (e.g. 1a2y_C)")
    parser.add_argument("--reference_cif", required=True, help="Path to native/reference CIF (for TMscore ground-truth)")
    parser.add_argument("--chain", default=None, help="Chain ID in reference CIF (default: taken from protein_id if possible)")
    parser.add_argument("--inference_output_dir", required=True, help="Directory containing decoy PDB files")
    parser.add_argument("--output_dir", required=True, help="Directory to write proteinebm_analysis outputs")
    parser.add_argument("--proteinebm_config", required=True, help="Path to ProteinEBM base_pretrain.yaml")
    parser.add_argument("--proteinebm_checkpoint", required=True, help="Path to ProteinEBM checkpoint (.pt)")
    parser.add_argument("--t", type=float, default=0.1, help="Diffusion time t for scoring (default: 0.1)")
    parser.add_argument(
        "--proteinebm_template_self_condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use template coordinates for self-conditioning input (default: True)",
    )

    args = parser.parse_args()

    print(f"Starting ProteinEBM scoring for {args.protein_id}", flush=True)
    chain = args.chain
    if chain is None and "_" in args.protein_id:
        chain = args.protein_id.split("_", 1)[1]
    scores_csv, summary_json = run_proteinebm_scoring_for_protein(
        protein_id=args.protein_id,
        inference_output_dir=args.inference_output_dir,
        output_dir=args.output_dir,
        reference_cif=args.reference_cif,
        reference_chain=chain,
        proteinebm_config=args.proteinebm_config,
        proteinebm_checkpoint=args.proteinebm_checkpoint,
        t=args.t,
        template_self_condition=args.proteinebm_template_self_condition,
    )
    print(f"Completed ProteinEBM scoring for {args.protein_id}", flush=True)
    print(f"Scores: {scores_csv}", flush=True)
    print(f"Summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()



